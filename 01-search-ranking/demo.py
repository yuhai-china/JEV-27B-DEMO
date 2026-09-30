"""Demo 01 — search re-ranking with JEV-27B System 1.

First stage: BM25 top-20 candidates (BEIR TREC-COVID, bundled in data/). Second stage: JEV-27B reads every
(query, document) pair and returns
  * P(relevant)        — kind=choice yes/no (the same question the Decision Index uses for ToolRet / BRIGHT)
  * relevance 0-5      — kind=score, expected rating
The script re-ranks live, prints before/after tables with the human relevance labels (0/1/2) and writes
results.json + charts in ../assets/.

    python demo.py
"""
import json, math, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from charts import JEV, BASE, ALT, style  # noqa: E402
from jev_client import decide_many, expected_score  # noqa: E402

DATA = json.load(open(os.path.join(HERE, "data", "trec_covid_demo.json")))
BENCH = json.load(open(os.path.join(HERE, "data", "benchmark_ndcg.json")))
ASSETS = os.path.join(HERE, "..", "assets")
PROB_STATE_TASK = "Rank candidate documents/tools by relevance to this query."
PROB_TASK = "Assess whether this candidate is relevant/useful to the query. Use the full text below; return the probability of relevance."
PROB_OPTS = ["no: Not relevant/useful to the query.", "yes: Relevant/useful to the query."]
RATE_Q = "Rate how relevant the document is to the search query on a 0-5 scale (0 = irrelevant, 5 = perfectly relevant)."


def ndcg(ranked_labels, all_relevant, k=10):
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(ranked_labels[:k]))
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(all_relevant, reverse=True)[:k]))
    return dcg / idcg if idcg else 0.0


def rerank(query, cands):
    reqs = []
    for c in cands:
        reqs.append(("choice", {"query": query, "task": PROB_STATE_TASK}, json.dumps({"candidate": c["text"], "task": PROB_TASK}), PROB_OPTS))
        reqs.append(("score", {"query": query, "document": c["text"]}, RATE_Q))
    out = decide_many(reqs)
    for i, c in enumerate(cands):
        c["p_relevant"] = list(out[2 * i].values())[1]
        c["rating"] = expected_score(out[2 * i + 1])
    return cands


def main():
    results, lines = [], []
    t_all = time.time()
    for q in DATA["queries"]:
        cands = [dict(c) for c in q["candidates"]]
        t = time.time()
        rerank(q["query"], cands)
        dt = time.time() - t
        bm = sorted(cands, key=lambda c: -c["bm25"])
        jv = sorted(cands, key=lambda c: -c["p_relevant"])
        nb, nj = ndcg([c["label"] for c in bm], q["relevant_grades"]), ndcg([c["label"] for c in jv], q["relevant_grades"])
        results.append({"query": q["query"], "ndcg10_bm25": nb, "ndcg10_jev": nj, "seconds": dt,
                        "bm25_top5": [{k: c[k] for k in ("text", "label")} for c in bm[:5]],
                        "jev_top5": [{k: c[k] for k in ("text", "label", "p_relevant", "rating")} for c in jv[:5]]})
        lines.append(f"\n### “{q['query']}”\n\n{2*len(cands)} decisions in {dt:.1f} s · nDCG@10 **{nb:.3f} → {nj:.3f}**\n")
        lines.append("| # | BM25 order | label | JEV-27B order | label | P(relevant) | rating |\n|---|---|:-:|---|:-:|---:|---:|")
        for r in range(5):
            a, b = bm[r], jv[r]
            lines.append(f"| {r+1} | {a['text'][:60].replace('|','/')}… | {a['label']} | {b['text'][:60].replace('|','/')}… | {b['label']} | {b['p_relevant']:.2f} | {b['rating']:.1f} |")
    json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False)
    open(os.path.join(HERE, "sample_output.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nall queries: {time.time()-t_all:.1f} s")

    plt = style()
    # chart 1: full benchmarks
    names = [("random order (mean of 5)", "random order", "#DEE2E6"), ("BM25 (first stage)", "BM25", BASE),
             ("bge-reranker-v2-m3 [bge]", "bge-reranker-v2-m3", "#74C0FC"),
             ("jev-27b [yesno]", "JEV-27B · P(relevant)", JEV), ("jev-27b [score]", "JEV-27B · rating 0-5", ALT)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (ds, title) in zip(axes, (("trec-covid", "TREC-COVID · 50 queries, BM25 top-100\n(not a Decision Index task)"),
                                      ("nfcorpus", "NFCorpus · 323 queries, BM25 top-50\n(not a Decision Index task)"),
                                      ("esci", "Amazon ESCI · 300 queries, judged products\n(a Decision Index task family)"))):
        rows = [(lab, BENCH[ds][k]["ndcg@10"], col) for k, lab, col in names if k in BENCH[ds]]
        ax.bar([r[0] for r in rows], [r[1] for r in rows], color=[r[2] for r in rows])
        for i, r in enumerate(rows):
            ax.text(i, r[1] + 0.01, f"{r[1]:.3f}", ha="center", fontsize=9)
        ax.set_title(title, fontsize=10); ax.set_ylim(0, max(r[1] for r in rows) * 1.18); ax.set_ylabel("nDCG@10")
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
    fig.suptitle("Re-ranking quality, nDCG@10 — JEV-27B used as a generic relevance judge, no ranking-specific fine-tuning", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "search_benchmarks.png")); plt.close(fig)
    # chart 2: per query before/after
    fig, ax = plt.subplots(figsize=(10, 3.8))
    xs = range(len(results)); w = 0.38
    ax.bar([x - w / 2 for x in xs], [r["ndcg10_bm25"] for r in results], w, label="BM25", color=BASE)
    ax.bar([x + w / 2 for x in xs], [r["ndcg10_jev"] for r in results], w, label="JEV-27B re-ranked", color=JEV)
    ax.set_xticks(list(xs)); ax.set_xticklabels([r["query"][:34] + ("…" if len(r["query"]) > 34 else "") for r in results], rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("nDCG@10"); ax.set_ylim(0, 1.1); ax.legend(); ax.set_title("Live demo: BM25 top-20 re-ranked by JEV-27B")
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "search_live.png")); plt.close(fig)


if __name__ == "__main__":
    main()
