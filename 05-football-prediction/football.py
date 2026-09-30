"""Football (soccer) match prediction with JEV-27B — System 1, System 1 + news, System 2, System 1 + 2, and bookmaker odds.

Matches: the 2026/27 season so far in the Premier League, La Liga, Serie A, Bundesliga and Ligue 1 (football-data.co.uk),
i.e. after the model's knowledge. Everything shown to the model is from before kick-off: last season's final table,
this season's table and form up to the match, last season's head-to-head, and pre-match headlines (Google News
`before:` the match date). Bookmaker probabilities (Pinnacle, margin removed) are the reference forecaster.

python football.py            # run everything, save results.json, print the scoreboard
python football.py summary    # re-print from results.json
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import csv
import datetime as dt
import email.utils
import io
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

HERE = os.path.dirname(os.path.abspath(__file__))
LEAGUES = {"E0": "Premier League", "SP1": "La Liga", "I1": "Serie A", "D1": "Bundesliga", "F1": "Ligue 1"}
UA = {"User-Agent": "Mozilla/5.0"}
Q = "Which full-time result (after 90 minutes) will this football match end with?"


def load(season, code):
    r = requests.get(f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv", headers=UA, timeout=30, allow_redirects=True)
    rows = list(csv.DictReader(io.StringIO(r.content.decode("utf-8-sig", errors="replace"))))
    out = []
    for x in rows:
        if not x.get("HomeTeam") or x.get("FTR") not in ("H", "D", "A"):
            continue
        d = dt.datetime.strptime(x["Date"], "%d/%m/%Y" if len(x["Date"]) == 10 else "%d/%m/%y").date()
        out.append({**x, "date": d})
    return sorted(out, key=lambda x: x["date"])


def table(matches):
    t = collections.defaultdict(lambda: {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "Pts": 0, "form": []})
    for m in matches:
        h, a, hg, ag = m["HomeTeam"], m["AwayTeam"], int(m["FTHG"]), int(m["FTAG"])
        for team, gf, ga in ((h, hg, ag), (a, ag, hg)):
            s = t[team]; s["P"] += 1; s["GF"] += gf; s["GA"] += ga
            r = "W" if gf > ga else ("D" if gf == ga else "L")
            s[r] += 1; s["Pts"] += {"W": 3, "D": 1, "L": 0}[r]; s["form"].append(r)
    order = sorted(t, key=lambda k: (-t[k]["Pts"], -(t[k]["GF"] - t[k]["GA"]), -t[k]["GF"]))
    for i, k in enumerate(order):
        t[k]["pos"] = i + 1
    return t


def implied(row):
    for a, b, c in (("PSCH", "PSCD", "PSCA"), ("PSH", "PSD", "PSA"), ("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A")):
        try:
            o = [float(row[a]), float(row[b]), float(row[c])]
        except (KeyError, ValueError, TypeError):
            continue
        inv = [1 / x for x in o]
        return [x / sum(inv) for x in inv], [float(row.get(k) or 0) or v for k, v in zip(("AvgH", "AvgD", "AvgA"), o)]
    return None, None


def news(home, away, before, n=6):
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode({"q": f"{home} vs {away} before:{before.isoformat()}", "hl": "en-GB", "gl": "GB", "ceid": "GB:en"})
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
        if before - dt.timedelta(days=10) <= d < before:
            out.append({"date": d.isoformat(), "title": it.findtext("title")})
    return sorted(out, key=lambda x: x["date"], reverse=True)[:n]


def build():
    matches = []
    for code, name in LEAGUES.items():
        cur, prev = load("2627", code), load("2526", code)
        pt = table(prev)
        base = collections.Counter(m["FTR"] for m in prev)
        for i, m in enumerate(cur):
            before = [x for x in cur if x["date"] < m["date"]]
            ct = table(before)
            h, a = m["HomeTeam"], m["AwayTeam"]
            def team(tn):
                last = ({"final_position": pt[tn]["pos"], "points_per_game": round(pt[tn]["Pts"] / pt[tn]["P"], 2)} if tn in pt
                        else "not in this league last season (promoted)")
                now = ct.get(tn)
                cur_s = ({"played": now["P"], "points": now["Pts"], "position": now["pos"], "goals_for": now["GF"], "goals_against": now["GA"],
                          "last5": "".join(now["form"][-5:])} if now and now["P"] else "no matches yet this season")
                return {"name": tn, "last_season": last, "this_season_so_far": cur_s}
            h2h = [f"{x['HomeTeam']} {x['FTHG']}-{x['FTAG']} {x['AwayTeam']}" for x in prev if {x["HomeTeam"], x["AwayTeam"]} == {h, a}]
            probs, odds = implied(m)
            if probs is None:
                continue
            matches.append({"league": name, "date": m["date"].isoformat(), "home": team(h), "away": team(a), "head_to_head_last_season": h2h,
                            "result": m["FTR"], "score": f"{m['FTHG']}-{m['FTAG']}", "bookmaker": probs, "avg_odds": odds,
                            "league_base_rates": [base["H"] / len(prev), base["D"] / len(prev), base["A"] / len(prev)]})
    return matches


def state(m, headlines=None, notes=None, odds=False):
    s = {"competition": m["league"], "match_date": m["date"], "home_team": m["home"], "away_team": m["away"],
         "head_to_head_last_season": m["head_to_head_last_season"] or "none"}
    if headlines is not None:
        s["pre_match_headlines"] = headlines or "none found"
    if notes:
        s["analyst_notes"] = notes
    if odds:
        s["bookmaker_probabilities"] = {"home win": round(m["bookmaker"][0], 3), "draw": round(m["bookmaker"][1], 3), "away win": round(m["bookmaker"][2], 3)}
    return s


def opts(m):
    return [f"home win: {m['home']['name']}", "draw", f"away win: {m['away']['name']}"]


def system2(m, heads):
    prompt = ("You are a football analyst. Estimate the full-time (90 minutes) result probabilities for this match, using only the data below.\n\n"
              + json.dumps(state(m, heads), ensure_ascii=False, indent=1) +
              "\n\nConsider home advantage, team strength last season and this season, form, and any team news in the headlines. "
              "Write at most 5 short bullet points, then finish with exactly one line: 'Home: 0.xx, Draw: 0.xx, Away: 0.xx'.")
    out = chat(prompt, thinking=True, max_tokens=2500)
    final = out.split("</think>")[-1].strip()
    mm = re.findall(r"Home:\s*([01](?:\.\d+)?)\s*,\s*Draw:\s*([01](?:\.\d+)?)\s*,\s*Away:\s*([01](?:\.\d+)?)", final)
    p = [float(x) for x in mm[-1]] if mm else None
    if p and sum(p) > 0:
        p = [x / sum(p) for x in p]
    return final[-1200:], p


def run():
    ms = build()
    print(f"{len(ms)} matches", flush=True)
    with cf.ThreadPoolExecutor(8) as ex:
        heads = list(ex.map(lambda m: news(m["home"]["name"], m["away"]["name"], dt.date.fromisoformat(m["date"])), ms))
    with cf.ThreadPoolExecutor(24) as ex:
        s2 = list(ex.map(lambda x: system2(*x), zip(ms, heads)))
    reqs = []
    for m, hd, (notes, _) in zip(ms, heads, s2):
        o = opts(m)
        reqs += [("choice", state(m), Q, o), ("choice", state(m, hd), Q, o), ("choice", state(m, hd, notes), Q, o),
                 ("choice", state(m, hd, notes, odds=True), Q + " Start from the bookmaker probabilities and adjust only as far as the information justifies.", o)]
    out = decide_many(reqs)
    for i, (m, hd, (notes, p2)) in enumerate(zip(ms, heads, s2)):
        m["headlines"], m["system2_analysis"], m["p_system2"] = hd, notes, p2
        for j, k in enumerate(("p_s1", "p_s1_news", "p_s1_s2", "p_s1_s2_odds")):
            m[k] = list(out[4 * i + j].values())
    json.dump(ms, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False)
    return ms


def rps(p, r):
    o = {"H": [1, 0, 0], "D": [0, 1, 0], "A": [0, 0, 1]}[r]
    c1, c2 = p[0] - o[0], (p[0] + p[1]) - (o[0] + o[1])
    return (c1 ** 2 + c2 ** 2) / 2


IDX = {"H": 0, "D": 1, "A": 2}


def metrics(ps, ms):
    n = len(ms)
    r = sum(rps(p, m["result"]) for p, m in zip(ps, ms)) / n
    b = sum(sum((p[k] - (1 if IDX[m["result"]] == k else 0)) ** 2 for k in range(3)) for p, m in zip(ps, ms)) / n
    ll = -sum(math.log(max(1e-4, p[IDX[m["result"]]])) for p, m in zip(ps, ms)) / n
    acc = sum(int(max(range(3), key=lambda k: p[k]) == IDX[m["result"]]) for p, m in zip(ps, ms)) / n
    return r, b, ll, acc


def temper(p, t):
    z = [math.log(max(x, 1e-9)) / t for x in p]
    e = [math.exp(v - max(z)) for v in z]
    return [v / sum(e) for v in e]


def cv_temperature(ps, ms, seed=0):
    """2-fold cross-validated temperature scaling: fit T on one half (log loss), apply to the other half."""
    import random
    order = list(range(len(ms))); random.Random(seed).shuffle(order)
    folds, out = [order[::2], order[1::2]], [None] * len(ms)
    for f in (0, 1):
        tr, te = folds[1 - f], folds[f]
        t = min((x / 10 for x in range(5, 61)), key=lambda t: -sum(math.log(max(1e-9, temper(ps[i], t)[IDX[ms[i]["result"]]])) for i in tr))
        for i in te:
            out[i] = temper(ps[i], t)
    return out


def summary(ms):
    import contextlib, io
    rows = [("uniform 1/3", lambda m: [1 / 3] * 3), ("league base rates (last season)", lambda m: m["league_base_rates"]),
            ("JEV System 1 — stats only", lambda m: m["p_s1"]), ("JEV System 1 + headlines", lambda m: m["p_s1_news"]),
            ("JEV System 2 (thinking) — stated probabilities", lambda m: m["p_system2"] or [1 / 3] * 3),
            ("JEV System 1 + 2", lambda m: m["p_s1_s2"]),
            ("bookmaker (Pinnacle, margin removed)", lambda m: m["bookmaker"]),
            ("JEV System 1 + 2 + bookmaker probabilities", lambda m: m["p_s1_s2_odds"])]
    res = collections.Counter(m["result"] for m in ms)
    buf = io.StringIO()
    score = {}
    with contextlib.redirect_stdout(buf):
        print(f"{len(ms)} matches · home {res['H']} / draw {res['D']} / away {res['A']}\n")
        print("| forecaster | RPS ↓ | Brier ↓ | log loss ↓ | top pick correct |\n|---|---:|---:|---:|---:|")
        for name, f in rows:
            r, b, ll, acc = metrics([f(m) for m in ms], ms)
            score[name] = r
            print(f"| {name} | {r:.4f} | {b:.4f} | {ll:.4f} | {100*acc:.1f}% |")
        print("\nConfidence check and a one-number fix (temperature scaling, fitted on half of the matches and scored on the other half):\n")
        print("| forecaster | avg. probability of its top pick | top pick correct | RPS after calibration | log loss after calibration |\n|---|---:|---:|---:|---:|")
        cal = {}
        for name, key in (("System 1 — stats only", "p_s1"), ("System 1 + headlines", "p_s1_news"), ("System 1 + 2", "p_s1_s2"),
                          ("System 1 + 2 + bookmaker", "p_s1_s2_odds"), ("System 2", "p_system2")):
            ps = [m[key] or [1 / 3] * 3 for m in ms]
            cps = cv_temperature(ps, ms)
            r, _, ll, _ = metrics(cps, ms)
            cal[name] = r
            print(f"| {name} | {sum(max(p) for p in ps)/len(ps):.2f} | {metrics(ps, ms)[3]:.2f} | {r:.4f} | {ll:.4f} |")
        bk = [m["bookmaker"] for m in ms]
        print(f"| bookmaker | {sum(max(p) for p in bk)/len(bk):.2f} | {metrics(bk, ms)[3]:.2f} | {metrics(bk, ms)[0]:.4f} (as is) | {metrics(bk, ms)[2]:.4f} |")
        mix = [[0.7 * b_ + 0.3 * s for b_, s in zip(m["bookmaker"], m["p_system2"] or [1 / 3] * 3)] for m in ms]
        print(f"\nBlend 0.7 × bookmaker + 0.3 × System 2: RPS {metrics(mix, ms)[0]:.4f} (bookmaker alone {metrics(bk, ms)[0]:.4f})")
        # illustrative value betting (NOT evidence: small sample, no staking model, closing-line odds)
        print("\nIllustrative flat-stake betting at average odds (1 unit per bet). 250 matches is far too few: these numbers are luck, not an edge.\n")
        print("| strategy | bets | profit (units) | ROI |\n|---|---:|---:|---:|")
        for name, key, edge in (("JEV System 1 + 2, bet when p × odds > 1.05", "p_s1_s2", 1.05), ("JEV System 1 + 2 + bookmaker, bet when p × odds > 1.02", "p_s1_s2_odds", 1.02)):
            bets = profit = 0
            for m in ms:
                for k in range(3):
                    if m[key][k] * m["avg_odds"][k] > edge:
                        bets += 1; profit += (m["avg_odds"][k] - 1) if IDX[m["result"]] == k else -1
            print(f"| {name} | {bets} | {profit:+.1f} | {100*profit/max(bets,1):+.1f}% |")
        bets = profit = 0
        for m in ms:
            k = max(range(3), key=lambda k: m["bookmaker"][k]); bets += 1; profit += (m["avg_odds"][k] - 1) if IDX[m["result"]] == k else -1
        print(f"| always back the bookmaker favourite | {bets} | {profit:+.1f} | {100*profit/max(bets,1):+.1f}% |")
    print(buf.getvalue())
    open(os.path.join(HERE, "sample_output.md"), "w").write(buf.getvalue())
    charts(ms, score, cal)


def charts(ms, score, cal):
    sys.path.insert(0, os.path.join(HERE, "..", "common"))
    from charts import JEV, BASE, ALT, WARN, style
    plt = style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
    names = [("uniform 1/3", "uniform\n1/3", BASE, None), ("league base rates (last season)", "league\nbase rates", BASE, None),
             ("JEV System 1 — stats only", "System 1\nstats", "#B197FC", "System 1 — stats only"),
             ("JEV System 1 + headlines", "System 1\n+ headlines", "#9775FA", "System 1 + headlines"),
             ("JEV System 1 + 2", "System 1 + 2", JEV, "System 1 + 2"),
             ("JEV System 2 (thinking) — stated probabilities", "System 2\n(thinking)", ALT, "System 2"),
             ("bookmaker (Pinnacle, margin removed)", "bookmaker\n(Pinnacle)", WARN, None)]
    xs = range(len(names))
    a1.bar(xs, [score[n[0]] for n in names], color=[n[2] for n in names], label="as produced")
    a1.bar([x for x, n in zip(xs, names) if n[3]], [cal[n[3]] for n in names if n[3]], fill=False, hatch="///", edgecolor="#343A40", lw=0.8,
           label="after temperature calibration (2-fold CV)")
    for x, n in zip(xs, names):
        a1.text(x, score[n[0]] + 0.003, f"{score[n[0]]:.3f}", ha="center", fontsize=8)
    a1.axhline(score["bookmaker (Pinnacle, margin removed)"], color=WARN, ls="--", lw=1)
    a1.set_xticks(list(xs)); a1.set_xticklabels([n[1] for n in names], fontsize=8); a1.set_ylim(0.18, 0.31)
    a1.set_ylabel("RPS (lower is better)"); a1.legend(loc="upper left", fontsize=8)
    a1.set_title(f"{len(ms)} matches, 2026/27 top-five European leagues")
    for key, lab, col in (("p_s1_news", "System 1 + headlines (raw)", "#9775FA"), ("p_system2", "System 2", ALT), ("bookmaker", "bookmaker", WARN)):
        ps = [m[key] or [1 / 3] * 3 for m in ms]
        pts = []
        for lo, hi in ((0.3, 0.45), (0.45, 0.6), (0.6, 0.75), (0.75, 0.9), (0.9, 1.01)):
            g = [(p, m) for p, m in zip(ps, ms) if lo <= max(p) < hi]
            if len(g) >= 8:
                pts.append((sum(max(p) for p, _ in g) / len(g), sum(int(max(range(3), key=lambda k: p[k]) == IDX[m["result"]]) for p, m in g) / len(g), len(g)))
        a2.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=col, label=lab)
    a2.plot([0.3, 1], [0.3, 1], color=BASE, ls="--", lw=1)
    a2.set_xlabel("stated probability of the top pick"); a2.set_ylabel("how often the top pick happened")
    a2.set_title("Reliability: System 1 is over-confident on football"); a2.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "..", "assets", "football_rps.png")); plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summary(json.load(open(os.path.join(HERE, "results.json"))))
    else:
        summary(run())
