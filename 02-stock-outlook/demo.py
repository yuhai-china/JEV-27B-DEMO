"""Demo 02 — turning live market data + headlines into typed signals with JEV-27B System 1.

Illustration only: NOT back-tested, NOT investment advice. The interesting part is the *signal extraction*
(news tone, is the news actually about the company, a stance with a probability) at ~30 decisions per second,
which is what a research / monitoring pipeline would consume.

State = last 3 months of daily closes (Yahoo Finance) + the 5 latest headlines. Four decisions per ticker:
  * noul    P(closes higher 5 trading days from now)
  * choice  stance for the next week (strong sell ... strong buy)
  * score   how positive the news flow is (0-5)
  * noul    are the headlines materially about the company's own business (flags noise)

    python demo.py                 # default watch-list
    python demo.py TSLA NFLX ASML  # your own tickers
"""
import datetime as dt, json, os, statistics, sys, time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from charts import JEV, BASE, ALT, WARN, style  # noqa: E402
from jev_client import decide_many, expected_score  # noqa: E402

H = {"User-Agent": "Mozilla/5.0"}
TICKERS = sys.argv[1:] or ["NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "AMD"]
STANCES = ["strong sell", "sell", "hold", "buy", "strong buy"]


def snapshot(t):
    c = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{t}?range=3mo&interval=1d", headers=H, timeout=20).json()["chart"]["result"][0]
    closes = [x for x in c["indicators"]["quote"][0]["close"] if x]
    day = dt.datetime.fromtimestamp(c["timestamp"][-1], dt.timezone.utc).date().isoformat()
    ret = lambda n: round(100 * (closes[-1] / closes[-1 - n] - 1), 2)
    daily = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
    news = requests.get(f"https://query1.finance.yahoo.com/v1/finance/search?q={t}&newsCount=6&quotesCount=0", headers=H, timeout=20).json().get("news", [])
    heads = [{"title": n["title"], "publisher": n.get("publisher"),
              "date": dt.datetime.fromtimestamp(n["providerPublishTime"], dt.timezone.utc).date().isoformat()} for n in news[:5]]
    return {"ticker": t, "as_of": day, "last_close": round(closes[-1], 2),
            "return_pct": {"1d": ret(1), "5d": ret(5), "20d": ret(20), "60d": ret(min(60, len(closes) - 1))},
            "volatility_20d_annualized_pct": round(100 * statistics.pstdev(daily[-20:]) * (252 ** 0.5), 1),
            "vs_20d_average_pct": round(100 * (closes[-1] / statistics.mean(closes[-20:]) - 1), 2),
            "vs_50d_average_pct": round(100 * (closes[-1] / statistics.mean(closes[-50:]) - 1), 2),
            "latest_headlines": heads}


def analyse(snaps):
    reqs = []
    for s in snaps:
        reqs += [("noul", s, "Is this scenario one where: the stock closes higher five trading days from now?"),
                 ("choice", s, "Which trading stance fits this scenario for the next week?", STANCES),
                 ("score", s, "Rate how positive the recent news flow is for this stock on a 0-5 scale (0 = very negative, 5 = very positive)."),
                 ("noul", s, "Is this scenario one where: the latest headlines are materially about this company's own business or fundamentals?")]
    t0 = time.time()
    out = decide_many(reqs)
    rows = []
    for i, s in enumerate(snaps):
        up, st, tone, rel = out[4 * i: 4 * i + 4]
        best = max(st, key=st.get)
        rows.append({**s, "p_up_5d": up["true"], "stance": best, "stance_p": st[best], "stance_dist": st,
                     "news_tone": expected_score(tone), "headlines_on_topic": rel["true"]})
    return rows, len(reqs), time.time() - t0


def main():
    snaps = [snapshot(t) for t in TICKERS]
    rows, n, secs = analyse(snaps)
    json.dump({"n_decisions": n, "seconds": secs, "rows": rows}, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False)
    md = [f"**{n} decisions in {secs:.1f} s** · prices as of {rows[0]['as_of']}\n",
          "| ticker | close | 5d / 20d / 60d % | vol 20d % | P(up in 5 days) | stance (prob) | news tone 0-5 | headlines on-topic |",
          "|---|---:|---|---:|---:|---|---:|---:|"]
    for r in rows:
        x = r["return_pct"]
        md.append(f"| {r['ticker']} | {r['last_close']} | {x['5d']:+} / {x['20d']:+} / {x['60d']:+} | {r['volatility_20d_annualized_pct']} | "
                  f"{r['p_up_5d']:.2f} | {r['stance']} ({r['stance_p']:.2f}) | {r['news_tone']:.1f} | {r['headlines_on_topic']:.2f} |")
    md.append(f"\nHeadlines the model saw for {rows[0]['ticker']}:\n")
    md += [f"- {h['date']} · {h['publisher']}: {h['title']}" for h in rows[0]["latest_headlines"]]
    open(os.path.join(HERE, "sample_output.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    plt = style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8))
    tk = [r["ticker"] for r in rows]
    a1.bar(tk, [r["p_up_5d"] for r in rows], color=[JEV if r["p_up_5d"] >= 0.5 else WARN for r in rows])
    a1.axhline(0.5, color=BASE, ls="--", lw=1); a1.set_ylim(0, 1); a1.set_title("P(closes higher in 5 trading days)")
    for i, r in enumerate(rows):
        a1.text(i, r["p_up_5d"] + 0.02, f"{r['p_up_5d']:.2f}", ha="center", fontsize=8)
    a2.scatter([r["news_tone"] for r in rows], [r["return_pct"]["20d"] for r in rows], s=[40 + 160 * r["headlines_on_topic"] for r in rows], color=ALT)
    for r in rows:
        a2.annotate(r["ticker"], (r["news_tone"], r["return_pct"]["20d"]), textcoords="offset points", xytext=(5, 4), fontsize=8)
    a2.set_xlabel("news tone (0 = very negative, 5 = very positive)"); a2.set_ylabel("20-day return %")
    a2.set_title("News tone vs recent return (bubble = headlines on-topic)")
    fig.suptitle(f"Live signals from JEV-27B System 1 — {n} decisions in {secs:.1f} s (illustration, not advice)", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "..", "assets", "stock_signals.png")); plt.close(fig)


if __name__ == "__main__":
    main()
