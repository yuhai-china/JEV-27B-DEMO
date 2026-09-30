"""Polymarket forecasting with JEV-27B: System 1 alone vs. System 1 + news vs. System 2 reasoning vs. System 1 + 2.

Pipeline for one market (question, resolution rules, forecast date D):
  1. news    headlines published before D (Google News RSS, `before:` operator + pubDate filter — no hindsight)
  2. S2      the same engine's System 2 (Qwen3.8-27B, thinking) analyses base rates, timelines and evidence and states a
             probability ("Probability: 0.xx")
  3. S1+2    System 1 reads question + rules + headlines + System 2's analysis and returns the calibrated P(Yes)

python forecast.py backtest   # resolved markets, forecast 7 days before resolution; Brier vs. the market price then
python forecast.py live       # live markets, forecast as of today, next to the current market price
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import email.utils
import json
import math
import os
import re
import sys
import urllib.parse
import xml.etree.ElementTree as ET

import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from jev_client import chat, decide_many  # noqa: E402

GAMMA = "https://gamma-api.polymarket.com/markets"
CLOB = "https://clob.polymarket.com/prices-history"
HERE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "Mozilla/5.0"}
INSTR = "Estimate the probability that the event in state resolves Yes, using only information available as of the forecast date."
CRIT = ["yes: The specified event/conjunction occurs.", "no: The specified event/conjunction does not occur."]
SPORTS = re.compile(r"\bvs\.?\b|starting 11|\bwin on \d{4}|o/u|spread|\bko\b|tko|total (goals|points)|map \d|mls|nfl|nba|mlb|nhl|uefa|premier league|"
                    r"champions league|grand prix|\bset \d|match|tournament|open\b|super bowl|world series|stanley cup", re.I)


def yes_no(m):
    try:
        o = json.loads(m.get("outcomes") or "[]"); p = [float(x) for x in json.loads(m.get("outcomePrices") or "[]")]
    except (ValueError, TypeError):
        return None
    return p[0] if o == ["Yes", "No"] and len(p) == 2 else None


def iso(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00").replace(" ", "T")) if s else None


def news_before(question: str, before: dt.date, n: int = 8) -> list[dict]:
    q = re.sub(r"^(will|is|are|does|do|did|has|have)\s+", "", question.strip().rstrip("?"), flags=re.I)
    q = re.sub(r"\b(by|before|on|in)\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b.*$", "", q, flags=re.I)
    q = " ".join(q.split()[:12])
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode({"q": f"{q} before:{before.isoformat()}", "hl": "en-US", "gl": "US", "ceid": "US:en"})
    try:
        root = ET.fromstring(requests.get(url, headers=UA, timeout=20).content)
    except Exception:
        return []
    out = []
    for it in root.iter("item"):
        try:
            d = email.utils.parsedate_to_datetime(it.findtext("pubDate")).date()
        except Exception:
            continue
        if d < before:
            out.append({"date": d.isoformat(), "title": it.findtext("title"), "source": (it.find("source").text if it.find("source") is not None else None)})
    return sorted(out, key=lambda x: x["date"], reverse=True)[:n]


def market_price_at(m, when: dt.datetime):
    try:
        tok = json.loads(m["clobTokenIds"])[0]
        h = requests.get(CLOB, params={"market": tok, "interval": "max", "fidelity": 360}, timeout=20).json()["history"]
    except Exception:
        return None
    before = [x["p"] for x in h if x["t"] <= when.timestamp()]
    return before[-1] if before else None


def base_state(m, as_of: str, news=None, notes=None):
    s = {"forecast_as_of": as_of, "resolution_date": (m.get("endDate") or "")[:10],
         "information_policy": ("Use only the frozen information below: the question, its resolution rules" +
                                (", headlines published before the forecast date" if news is not None else "") +
                                (" and an analyst's notes" if notes else "") + ". No browsing. The market price is not shown."),
         "event": {"question": m["question"], "resolution_rules": (m.get("description") or "")[:1500]}}
    if news is not None:
        s["news_before_forecast_date"] = news
    if notes:
        s["analyst_notes"] = notes
    return s


def system2(m, as_of: str, news: list[dict]) -> tuple[str, float | None]:
    heads = "\n".join(f"- {h['date']} · {h['source']}: {h['title']}" for h in news) or "- (no headlines found)"
    prompt = (f"You are forecasting a prediction market as of {as_of}. You cannot browse; use only what is below and general knowledge.\n\n"
              f"Question: {m['question']}\nResolves: {(m.get('endDate') or '')[:10]}\nResolution rules: {(m.get('description') or '')[:1500]}\n\n"
              f"Headlines published before {as_of}:\n{heads}\n\n"
              "Reason about the base rate, how much time is left, what would have to happen, and what the headlines imply. "
              "Then write a short analysis (at most 6 bullet points) and finish with a line 'Probability: 0.xx' for Yes.")
    out = chat(prompt, thinking=True, max_tokens=3000)
    final = out.split("</think>")[-1].strip()
    mm = re.findall(r"Probability:\s*([01](?:\.\d+)?)", final) or re.findall(r"Probability:\s*([01](?:\.\d+)?)", out)
    return final[-1500:], (float(mm[-1]) if mm else None)


def run(markets: list[tuple[dict, str]], workers: int = 16) -> list[dict]:
    """markets = [(market, forecast_date_iso)] -> per-market dict with all four forecasts."""
    with cf.ThreadPoolExecutor(8) as ex:
        news = list(ex.map(lambda x: news_before(x[0]["question"], dt.date.fromisoformat(x[1])), markets))
    with cf.ThreadPoolExecutor(workers) as ex:
        s2 = list(ex.map(lambda x: system2(x[0][0], x[0][1], x[1]), zip(markets, news)))
    reqs = []
    for (m, d), nw, (notes, _) in zip(markets, news, s2):
        reqs += [("choice", base_state(m, d), INSTR, CRIT), ("choice", base_state(m, d, nw), INSTR, CRIT),
                 ("choice", base_state(m, d, nw, notes), INSTR, CRIT)]
    out = decide_many(reqs)
    res = []
    for i, ((m, d), nw, (notes, p2)) in enumerate(zip(markets, news, s2)):
        res.append({"question": m["question"], "forecast_date": d, "resolution_date": (m.get("endDate") or "")[:10],
                    "headlines": nw, "system2_analysis": notes, "p_system2": p2,
                    "p_s1": list(out[3 * i].values())[0], "p_s1_news": list(out[3 * i + 1].values())[0], "p_s1_s2": list(out[3 * i + 2].values())[0]})
    return res


def brier(ps, ys):
    return sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ys)


def logloss(ps, ys):
    return -sum(math.log(max(1e-4, p if y else 1 - p)) for p, y in zip(ps, ys)) / len(ys)


def backtest(n=100, horizon_days=7, closed_after="2026-03-01", select_only=False):
    """Resolved non-sports markets (politics, economy, crypto, geopolitics, tech, world, business, science), closed after
    `closed_after` (after the model's knowledge), volume >= $50k, <= 3 per event; forecast date = close - horizon."""
    cands, seen = [], set()
    for tag in ["politics", "economy", "crypto", "geopolitics", "tech", "world", "business", "science"]:
        for off in (0, 100, 200):
            evs = requests.get("https://gamma-api.polymarket.com/events", params={"closed": "true", "tag_slug": tag, "limit": 100, "offset": off,
                                                                                  "order": "endDate", "ascending": "false"}, timeout=60).json()
            if not isinstance(evs, list):
                break
            for e in evs:
                for m in e.get("markets", []):
                    if m.get("id") in seen:
                        continue
                    seen.add(m.get("id"))
                    m["_event"] = e.get("slug")
                    cands.append(m)
    per_event, picked = {}, []
    cands = [m for m in cands if m.get("closedTime") and m["closedTime"][:10] >= closed_after]
    for m in sorted(cands, key=lambda m: m["closedTime"], reverse=True):
        y = yes_no(m)
        if y not in (0.0, 1.0) or m.get("umaResolutionStatus") != "resolved" or SPORTS.search(m["question"]):
            continue
        if float(m.get("volumeNum") or m.get("volume") or 0) < 50000:
            continue
        closed = iso(m["closedTime"]); fd = closed - dt.timedelta(days=horizon_days)
        if iso(m.get("startDate") or m.get("createdAt") or "2100-01-01") > fd:
            continue
        if per_event.get(m["_event"], 0) >= 3:
            continue
        per_event[m["_event"]] = per_event.get(m["_event"], 0) + 1
        picked.append((m, fd.date().isoformat(), y, market_price_at(m, fd)))
        if len(picked) >= n:
            break
    print(f"candidates {len(cands)}, picked {len(picked)} from {len(per_event)} events", flush=True)
    picked = picked[:n]
    json.dump([{"market": {k: m.get(k) for k in ("id", "question", "description", "endDate", "closedTime", "clobTokenIds", "volumeNum", "_event")},
                "forecast_date": d, "outcome": y, "market_price_at_forecast": mp} for m, d, y, mp in picked],
              open(os.path.join(HERE, "superseded", "picks_backtest.json"), "w"), indent=1, ensure_ascii=False)
    if select_only:
        return picked
    res = run([(m, d) for m, d, _, _ in picked])
    for r, (m, d, y, mp) in zip(res, picked):
        r["outcome"] = y; r["market_price_at_forecast"] = mp; r["volume"] = float(m.get("volumeNum") or 0)
    json.dump(res, open(os.path.join(HERE, "superseded", "results_backtest.json"), "w"), indent=1, ensure_ascii=False)
    return res


def live(n=15):
    today = dt.date.today().isoformat()
    ms = requests.get(GAMMA, params={"active": "true", "closed": "false", "limit": 300, "order": "volume24hr", "ascending": "false"}, timeout=60).json()
    ms = [m for m in ms if yes_no(m) is not None and not SPORTS.search(m["question"])][:n]
    res = run([(m, today) for m in ms])
    for r, m in zip(res, ms):
        r["market_price_now"] = yes_no(m)
    json.dump(res, open(os.path.join(HERE, "results_live.json"), "w"), indent=1, ensure_ascii=False)
    return res


def summarize(res):
    ys = [r["outcome"] for r in res]
    rows = [("always 0.5", [0.5] * len(ys)), (f"base rate of this sample ({sum(ys)/len(ys):.2f}, hindsight)", [sum(ys) / len(ys)] * len(ys)),
            ("JEV System 1 — question only", [r["p_s1"] for r in res]), ("JEV System 1 + headlines", [r["p_s1_news"] for r in res]),
            ("JEV System 2 (thinking) — stated probability", [r["p_system2"] if r["p_system2"] is not None else 0.5 for r in res]),
            ("**JEV System 1 + 2** (headlines + System 2 analysis → System 1)", [r["p_s1_s2"] for r in res])]
    mk = [r for r in res if r.get("market_price_at_forecast") is not None]
    print(f"{len(ys)} resolved markets · {100*sum(ys)/len(ys):.0f}% resolved Yes · forecast 7 days before resolution\n")
    print("| forecaster | Brier ↓ | log loss ↓ | direction ✓ |\n|---|---:|---:|---:|")
    for name, ps in rows:
        acc = sum(int((p > 0.5) == (y == 1.0)) for p, y in zip(ps, ys)) / len(ys)
        print(f"| {name} | {brier(ps, ys):.3f} | {logloss(ps, ys):.3f} | {100*acc:.0f}% |")
    if mk:
        ym = [r["outcome"] for r in mk]
        print(f"\nOn the {len(mk)} markets with a price history: market price 7 days before resolution Brier {brier([r['market_price_at_forecast'] for r in mk], ym):.3f}; "
              f"JEV System 1 + 2 Brier {brier([r['p_s1_s2'] for r in mk], ym):.3f}")


# ---------------------------------------------------------------------------------------------------------------------
# Option 3: JEV as a corrector of the market price (market price at the forecast date is part of the input)
INSTR_MKT = ("Estimate the probability that the event in state resolves Yes, starting from the market's probability and adjusting only "
             "as far as the information available as of the forecast date justifies.")


def mkt_state(m, as_of, price, news=None, notes=None):
    s = {"forecast_as_of": as_of, "resolution_date": (m.get("endDate") or "")[:10],
         "information_policy": "Use only the frozen information below. No browsing.",
         "event": {"question": m["question"], "resolution_rules": (m.get("description") or "")[:1500]},
         "prediction_market": {"probability_yes": round(price, 3), "as_of": as_of, "note": "last traded price of the Yes share"}}
    if news is not None:
        s["news_before_forecast_date"] = news
    if notes:
        s["analyst_notes"] = notes
    return s


def system2_mkt(m, as_of, price, news):
    heads = "\n".join(f"- {h['date']} · {h['source']}: {h['title']}" for h in news) or "- (no headlines found)"
    prompt = (f"You are adjusting a prediction-market forecast as of {as_of}. You cannot browse.\n\n"
              f"Question: {m['question']}\nResolves: {(m.get('endDate') or '')[:10]}\nResolution rules: {(m.get('description') or '')[:1500]}\n\n"
              f"Current market probability of Yes: {price:.3f}\n\nHeadlines published before {as_of}:\n{heads}\n\n"
              "Markets are usually well calibrated. Decide whether the headlines contain information the price may not yet reflect, "
              "and how strongly. Write a short analysis (at most 5 bullet points) and finish with a line 'Probability: 0.xx' for Yes.")
    out = chat(prompt, thinking=True, max_tokens=3000)
    final = out.split("</think>")[-1].strip()
    mm = re.findall(r"Probability:\s*([01](?:\.\d+)?)", final) or re.findall(r"Probability:\s*([01](?:\.\d+)?)", out)
    return final[-1500:], (float(mm[-1]) if mm else None)


def backtest_market():
    picks = json.load(open(os.path.join(HERE, "superseded", "picks_backtest.json")))
    prev = {r["question"]: r for r in json.load(open(os.path.join(HERE, "superseded", "results_backtest.json")))}
    picks = [p for p in picks if p["market_price_at_forecast"] is not None]
    news = [prev[p["market"]["question"]]["headlines"] if p["market"]["question"] in prev
            else news_before(p["market"]["question"], dt.date.fromisoformat(p["forecast_date"])) for p in picks]
    print(f"{len(picks)} markets with a price; headlines reused for {sum(p['market']['question'] in prev for p in picks)}", flush=True)
    with cf.ThreadPoolExecutor(16) as ex:
        s2 = list(ex.map(lambda x: system2_mkt(x[0]["market"], x[0]["forecast_date"], x[0]["market_price_at_forecast"], x[1]), zip(picks, news)))
    reqs = []
    for p, nw, (notes, _) in zip(picks, news, s2):
        m, d, mp = p["market"], p["forecast_date"], p["market_price_at_forecast"]
        reqs += [("choice", mkt_state(m, d, mp), INSTR_MKT, CRIT), ("choice", mkt_state(m, d, mp, nw), INSTR_MKT, CRIT),
                 ("choice", mkt_state(m, d, mp, nw, notes), INSTR_MKT, CRIT)]
    out = decide_many(reqs)
    res = []
    for i, (p, nw, (notes, p2)) in enumerate(zip(picks, news, s2)):
        res.append({"question": p["market"]["question"], "forecast_date": p["forecast_date"], "outcome": p["outcome"],
                    "market": p["market_price_at_forecast"], "headlines": nw, "system2_analysis": notes, "p_system2": p2,
                    "p_s1_mkt": list(out[3 * i].values())[0], "p_s1_mkt_news": list(out[3 * i + 1].values())[0],
                    "p_s1_s2_mkt": list(out[3 * i + 2].values())[0]})
    json.dump(res, open(os.path.join(HERE, "superseded", "results_backtest_market.json"), "w"), indent=1, ensure_ascii=False)
    return res


def summarize_market(res):
    ys = [r["outcome"] for r in res]; mk = [r["market"] for r in res]
    rows = [("market price 7 days before resolution", mk),
            ("JEV System 1 + market price", [r["p_s1_mkt"] for r in res]),
            ("JEV System 1 + market price + headlines", [r["p_s1_mkt_news"] for r in res]),
            ("JEV System 2 (thinking) + market price + headlines — stated probability", [r["p_system2"] if r["p_system2"] is not None else r["market"] for r in res]),
            ("**JEV System 1 + 2** + market price + headlines", [r["p_s1_s2_mkt"] for r in res])]
    print(f"{len(ys)} resolved markets · {100*sum(ys)/len(ys):.0f}% resolved Yes · forecast 7 days before resolution · market price in the input\n")
    print("| forecaster | Brier ↓ | log loss ↓ | direction ✓ | moved toward the outcome (of moves > 0.02) |\n|---|---:|---:|---:|---:|")
    for name, ps in rows:
        acc = sum(int((p > 0.5) == (y == 1.0)) for p, y in zip(ps, ys)) / len(ys)
        moves = [(p - m_, y) for p, m_, y in zip(ps, mk, ys) if abs(p - m_) > 0.02]
        good = sum(int((d > 0) == (y == 1.0)) for d, y in moves)
        mv = f"{good}/{len(moves)}" if name != rows[0][0] else "—"
        print(f"| {name} | {brier(ps, ys):.3f} | {logloss(ps, ys):.3f} | {100*acc:.0f}% | {mv} |")


# ---------------------------------------------------------------------------------------------------------------------
# Clean design: fixed calendar forecast dates. The first backtest anchored the forecast date on each market's *actual*
# close time (close - 7 days). "Will X happen by D" markets that resolve Yes usually close early, so that anchor leaks the
# outcome: 59% of Yes markets but 33% of No markets closed >= 3 days before their scheduled end, and the market looked
# like it under-priced Yes. Here every market that was open on a fixed date is forecast on that date.
FIXED_DATES = ["2026-04-01", "2026-05-01", "2026-06-01", "2026-07-01", "2026-08-01"]


def closed_candidates(end_min: dt.datetime, end_max: dt.datetime, pages=3):
    """Highest-volume closed events whose scheduled end falls in [end_min, end_max], across non-sports tags."""
    cands, seen = [], set()
    for tag in ["politics", "economy", "crypto", "geopolitics", "tech", "world", "business", "science", "finance", "elections"]:
        for off in range(0, 100 * pages, 100):
            try:
                evs = requests.get("https://gamma-api.polymarket.com/events", params={
                    "closed": "true", "tag_slug": tag, "limit": 100, "offset": off, "order": "volume", "ascending": "false",
                    "end_date_min": end_min.strftime("%Y-%m-%dT%H:%M:%SZ"), "end_date_max": end_max.strftime("%Y-%m-%dT%H:%M:%SZ")}, timeout=60).json()
            except Exception:
                break
            if not isinstance(evs, list) or not evs:
                break
            for e in evs:
                for m in e.get("markets", []):
                    if not isinstance(m, dict) or m.get("id") in seen:
                        continue
                    seen.add(m.get("id"))
                    m["_event"] = e.get("slug")
                    cands.append(m)
    return cands


def price_at(m, when: dt.datetime):
    """Last traded Yes price at or before `when` (explicit time window: interval=max is empty for older closed markets)."""
    try:
        tok = json.loads(m["clobTokenIds"])[0]
        h = requests.get(CLOB, params={"market": tok, "startTs": int((when - dt.timedelta(days=4)).timestamp()),
                                       "endTs": int(when.timestamp()), "fidelity": 60}, timeout=20).json()["history"]
    except Exception:
        return None
    before = [x["p"] for x in h if x["t"] <= when.timestamp()]
    return before[-1] if before else None


def backtest_fixed(per_date=30, horizon_max_days=45):
    used, picks = set(), []
    for d in FIXED_DATES:
        fd = dt.datetime.fromisoformat(d + "T12:00:00+00:00")
        cands = closed_candidates(fd, fd + dt.timedelta(days=horizon_max_days))
        pool = []
        for m in cands:
            y = yes_no(m)
            if y not in (0.0, 1.0) or m.get("umaResolutionStatus") != "resolved" or SPORTS.search(m["question"]) or m["id"] in used:
                continue
            if float(m.get("volumeNum") or m.get("volume") or 0) < 50000 or not m.get("closedTime") or not m.get("endDate"):
                continue
            start, closed, end = iso(m.get("startDate") or m.get("createdAt") or "2100-01-01"), iso(m["closedTime"]), iso(m["endDate"])
            if start < fd - dt.timedelta(days=1) and closed > fd + dt.timedelta(days=1) and fd < end <= fd + dt.timedelta(days=horizon_max_days):
                pool.append(m)
        per_event, n = {}, 0
        for m in sorted(pool, key=lambda m: -float(m.get("volumeNum") or 0)):
            if per_event.get(m["_event"], 0) >= 2:
                continue
            price = price_at(m, fd)
            if price is None:
                continue
            per_event[m["_event"]] = per_event.get(m["_event"], 0) + 1
            used.add(m["id"])
            picks.append({"market": {k: m.get(k) for k in ("id", "question", "description", "endDate", "closedTime", "volumeNum", "_event")},
                          "forecast_date": d, "outcome": yes_no(m), "market_price": price})
            n += 1
            if n >= per_date:
                break
        print(f"{d}: {len(cands)} candidates, pool {len(pool)}, picked {n}", flush=True)
    json.dump(picks, open(os.path.join(HERE, "picks_fixed.json"), "w"), indent=1, ensure_ascii=False)
    res = run([(p["market"], p["forecast_date"]) for p in picks])
    reqs = []
    for p, r in zip(picks, res):
        m, d, mp = p["market"], p["forecast_date"], p["market_price"]
        reqs += [("choice", mkt_state(m, d, mp), INSTR_MKT, CRIT), ("choice", mkt_state(m, d, mp, r["headlines"]), INSTR_MKT, CRIT)]
    out = decide_many(reqs)
    for i, (p, r) in enumerate(zip(picks, res)):
        r.update({"event": p["market"]["_event"], "outcome": p["outcome"], "market": p["market_price"],
                  "p_s1_mkt": list(out[2 * i].values())[0], "p_s1_mkt_news": list(out[2 * i + 1].values())[0]})
    json.dump(res, open(os.path.join(HERE, "results_fixed.json"), "w"), indent=1, ensure_ascii=False)
    return res


def summarize_fixed(res):
    import random
    ys = [r["outcome"] for r in res]; n = len(res)
    mk = [min(max(r["market"], 1e-3), 1 - 1e-3) for r in res]
    lg = lambda p: math.log(p / (1 - p)); sg = lambda z: 1 / (1 + math.exp(-z))
    # "push toward Yes" control, leave-one-event-out
    evs = sorted(set(r["event"] for r in res))
    shift = [None] * n
    for e in evs:
        tr = [i for i in range(n) if res[i]["event"] != e]
        a = min((x / 20 for x in range(-40, 41)), key=lambda a: sum((sg(a + lg(mk[i])) - ys[i]) ** 2 for i in tr))
        for i in range(n):
            if res[i]["event"] == e:
                shift[i] = sg(a + lg(mk[i]))
    rows = [("always 0.5", [0.5] * n), (f"base rate of this sample ({sum(ys)/n:.2f}, hindsight)", [sum(ys) / n] * n),
            ("JEV System 1 — question only", [r["p_s1"] for r in res]), ("JEV System 1 + headlines", [r["p_s1_news"] for r in res]),
            ("JEV System 2 (thinking) — stated probability", [r["p_system2"] if r["p_system2"] is not None else 0.5 for r in res]),
            ("JEV System 1 + 2", [r["p_s1_s2"] for r in res]),
            ("market price on the forecast date", [r["market"] for r in res]),
            ("control: market price shifted by a constant (fitted leave-one-event-out)", shift),
            ("JEV System 1 + market price", [r["p_s1_mkt"] for r in res]),
            ("JEV System 1 + market price + headlines", [r["p_s1_mkt_news"] for r in res])]
    print(f"{n} resolved markets · {len(evs)} events · forecast dates {', '.join(FIXED_DATES)} · {100*sum(ys)/n:.0f}% resolved Yes\n")
    print("| forecaster | Brier ↓ | log loss ↓ | direction ✓ |\n|---|---:|---:|---:|")
    table = {}
    for name, ps in rows:
        acc = sum(int((p > 0.5) == (y == 1.0)) for p, y in zip(ps, ys)) / n
        table[name] = brier(ps, ys)
        print(f"| {name} | {brier(ps, ys):.3f} | {logloss(ps, ys):.3f} | {100*acc:.0f}% |")
    random.seed(0)
    by_ev = {e: [i for i in range(n) if res[i]["event"] == e] for e in evs}
    for key, lab in (("p_s1_mkt", "System 1 + market price"), ("p_s1_mkt_news", "System 1 + market price + headlines")):
        diffs = []
        for _ in range(5000):
            idx = [i for e in random.choices(evs, k=len(evs)) for i in by_ev[e]]
            diffs.append(sum((res[i]["market"] - ys[i]) ** 2 - (res[i][key] - ys[i]) ** 2 for i in idx) / len(idx))
        diffs.sort()
        g = sum((r["market"] - y) ** 2 - (r[key] - y) ** 2 for r, y in zip(res, ys)) / n
        print(f"\nBrier gain of {lab} over the market: {g:+.4f}, 95% event-clustered bootstrap interval [{diffs[125]:+.4f}, {diffs[4875]:+.4f}]")
    for y in (1.0, 0.0):
        idx = [i for i in range(n) if ys[i] == y]
        print(f"resolved {'Yes' if y else 'No'} (n={len(idx)}): mean JEV − market = {sum(res[i]['p_s1_mkt'] - res[i]['market'] for i in idx) / len(idx):+.3f}")
    return table


def charts():
    """Charts + markdown tables for the README from the two cached backtests."""
    import contextlib, io, random
    sys.path.insert(0, os.path.join(HERE, "..", "common"))
    from charts import JEV, BASE, ALT, WARN, style
    a = json.load(open(os.path.join(HERE, "superseded", "results_backtest.json")))
    b = json.load(open(os.path.join(HERE, "superseded", "results_backtest_market.json")))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print("## A. Without the market price\n"); summarize(a)
        print("\n## B. With the market price in the input (JEV as a market corrector)\n"); summarize_market(b)
    ys = [r["outcome"] for r in b]
    random.seed(0)
    diffs = []
    for _ in range(10000):
        idx = [random.randrange(len(b)) for _ in b]
        diffs.append(sum((b[i]["market"] - ys[i]) ** 2 - (b[i]["p_s1_mkt"] - ys[i]) ** 2 for i in idx) / len(idx))
    diffs.sort()
    gain = sum((r["market"] - y) ** 2 - (r["p_s1_mkt"] - y) ** 2 for r, y in zip(b, ys)) / len(b)
    boot = (f"\nBootstrap (10,000 resamples of the {len(b)} markets): Brier improvement of *System 1 + market price* over the market "
            f"= {gain:+.4f}, 95% interval [{diffs[250]:+.4f}, {diffs[9750]:+.4f}], share of resamples > 0: {sum(d > 0 for d in diffs) / len(diffs):.2f}")
    print(buf.getvalue() + boot)
    open(os.path.join(HERE, "sample_output.md"), "w").write(buf.getvalue() + boot + "\n")

    plt = style()
    ya = [r["outcome"] for r in a]
    rows_a = [("always 0.5", [0.5] * len(ya), BASE), ("System 1\nquestion only", [r["p_s1"] for r in a], "#B197FC"),
              ("System 1\n+ headlines", [r["p_s1_news"] for r in a], "#9775FA"),
              ("System 2\nstated prob.", [r["p_system2"] if r["p_system2"] is not None else 0.5 for r in a], ALT),
              ("System 1 + 2", [r["p_s1_s2"] for r in a], JEV)]
    mk_a = [r for r in a if r.get("market_price_at_forecast") is not None]
    rows_b = [("market price", [r["market"] for r in b], BASE), ("System 1\n+ market", [r["p_s1_mkt"] for r in b], JEV),
              ("System 1\n+ market\n+ headlines", [r["p_s1_mkt_news"] for r in b], "#9775FA"),
              ("System 2\n+ market\n+ headlines", [r["p_system2"] if r["p_system2"] is not None else r["market"] for r in b], ALT),
              ("System 1 + 2\n+ market\n+ headlines", [r["p_s1_s2_mkt"] for r in b], "#B197FC")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.3))
    for ax, rows, y, title in ((a1, rows_a, ya, f"A. Forecasting alone ({len(a)} markets)"), (a2, rows_b, ys, f"B. Correcting the market price ({len(b)} markets)")):
        vals = [brier(ps, y) for _, ps, _ in rows]
        ax.bar([r[0] for r in rows], vals, color=[r[2] for r in rows])
        for i, v in enumerate(vals):
            ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)
        ax.set_ylabel("Brier score (lower is better)"); ax.set_title(title); ax.tick_params(axis="x", labelsize=8)
    a1.axhline(brier([r["market_price_at_forecast"] for r in mk_a], [r["outcome"] for r in mk_a]), color=WARN, ls="--", lw=1)
    a1.text(-0.45, brier([r["market_price_at_forecast"] for r in mk_a], [r["outcome"] for r in mk_a]) - 0.022,
            f"market price ({brier([r['market_price_at_forecast'] for r in mk_a], [r['outcome'] for r in mk_a]):.3f})", color=WARN, ha="left", fontsize=8)
    fig.suptitle("Polymarket, forecast 7 days before resolution — resolved markets closed after 1 March 2026", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "..", "assets", "polymarket_brier.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for y, col, lab in ((1.0, ALT, "resolved Yes"), (0.0, WARN, "resolved No")):
        g = [r for r in b if r["outcome"] == y]
        ax.scatter([r["market"] for r in g], [r["p_s1_mkt"] for r in g], s=22, color=col, alpha=0.8, label=lab)
    ax.plot([0, 1], [0, 1], color=BASE, lw=1, ls="--"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("market price 7 days before resolution"); ax.set_ylabel("JEV System 1 given the market price")
    ax.set_title("How JEV adjusts the market\n(green above / orange below the diagonal = moved the right way)", fontsize=10); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "..", "assets", "polymarket_adjustments.png")); plt.close(fig)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "backtest"
    if mode == "charts":
        charts()
        sys.exit()
    if mode == "fixed":
        summarize_fixed(backtest_fixed())
        sys.exit()
    if mode == "fixed-summary":
        summarize_fixed(json.load(open(os.path.join(HERE, "results_fixed.json"))))
        sys.exit()
    if mode == "backtest":
        summarize(backtest())
    elif mode == "live":
        for r in live():
            print(f"{r['question'][:70]:70s}  market {r['market_price_now']:.2f} | S1 {r['p_s1']:.2f} | S1+news {r['p_s1_news']:.2f} | S2 {r['p_system2']} | S1+2 {r['p_s1_s2']:.2f}")
    elif mode == "picks":
        print(len(backtest(select_only=True)), "picks cached")
    elif mode == "market":
        summarize_market(backtest_market())
    elif mode == "market-summary":
        summarize_market(json.load(open(os.path.join(HERE, "superseded", "results_backtest_market.json"))))
    elif mode == "summary":
        summarize(json.load(open(os.path.join(HERE, "superseded", "results_backtest.json"))))
