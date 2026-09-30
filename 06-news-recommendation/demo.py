"""Demo 06 — zero-shot news recommendation on MIND (Microsoft News Dataset).

JEV-27B has never been trained on MIND or on any click data. System 1 reads a user's recently clicked headlines and one
candidate article and returns P(this user clicks it) in one forward pass. That is the whole recommender: no training,
no user embeddings, no click history of the article needed, so brand-new articles are covered from the first second.

Two comparisons:
  1. zero-shot vs zero-shot: popularity so far, category match, title similarity (TF-IDF)
  2. trained vs trained, same data: a LightGBM ranker trained on MIND impressions, with and without JEV's zero-shot score
     as one extra feature

Data: MIND-small (train: 9-14 Nov 2019, evaluation: 15 Nov 2019), downloaded at run time from the Hugging Face mirror
huyva/MIND-small. MIND is released by Microsoft under the Microsoft Research License Terms (non-commercial research).

    python demo.py            # full run (~15 min on one GPU; JEV scores are cached in jev_scores.json)
    python demo.py eval       # re-evaluate from features.json (no JEV calls)
    python demo.py summary    # tables + chart from the cached results
"""
import bisect, collections, json, math, os, sys, time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from charts import JEV, BASE, ALT, style  # noqa: E402
from jev_client import decide_many  # noqa: E402

N_EVAL, N_TRAIN_STACK, N_TRAIN_BIG = 500, 1000, 4000
QUESTION = "Is this scenario one where: this user clicks on the candidate article?"
CACHE = os.path.join(HERE, "jev_scores.json")
NAMES = {"random": "random order", "pop": "popularity so far", "cat_share": "category match", "sim_max": "title similarity (TF-IDF)",
         "jev": "JEV-27B System 1, zero-shot", "lgb": "LightGBM ranker", "lgb_jev": "LightGBM ranker + JEV zero-shot score",
         "lgb_big": "LightGBM ranker, 4x more training data"}


def load_mind():
    from huggingface_hub import hf_hub_download
    get = lambda f: hf_hub_download("huyva/MIND-small", f, repo_type="dataset")
    cols = ["iid", "uid", "time", "hist_", "imps"]
    tb = pd.read_csv(get("train/behaviors.tsv"), sep="\t", header=None, names=cols)
    db = pd.read_csv(get("dev/behaviors.tsv"), sep="\t", header=None, names=cols)
    for b in (tb, db):
        b["t"] = pd.to_datetime(b.time, format="%m/%d/%Y %I:%M:%S %p")
        b["hist_"] = b["hist_"].fillna("").str.split()
        b["imps"] = b.imps.str.split()
    news = pd.concat([pd.read_csv(get(f"{s}/news.tsv"), sep="\t", header=None, quoting=3,
                                  names=["nid", "cat", "sub", "title", "abstract", "url", "te", "ae"]) for s in ("train", "dev")])
    news = news.drop_duplicates("nid").set_index("nid")
    news["abstract"] = news.abstract.fillna("")
    return tb, db, news


def load_eval_day():
    """Only what the live app needs: the evaluation-day impressions and all article texts."""
    from huggingface_hub import hf_hub_download
    get = lambda f: hf_hub_download("huyva/MIND-small", f, repo_type="dataset")
    db = pd.read_csv(get("dev/behaviors.tsv"), sep="\t", header=None, names=["iid", "uid", "time", "hist_", "imps"])
    db["hist_"] = db["hist_"].fillna("").str.split()
    db["imps"] = db.imps.str.split()
    news = pd.concat([pd.read_csv(get(f"{s}/news.tsv"), sep="\t", header=None, quoting=3,
                                  names=["nid", "cat", "sub", "title", "abstract", "url", "te", "ae"]) for s in ("train", "dev")])
    news = news.drop_duplicates("nid").set_index("nid")
    news["abstract"] = news.abstract.fillna("")
    return db, news


def rank_impression(db, news, iid):
    """Live: score every candidate of one impression with JEV System 1."""
    r = db[db.iid == iid].iloc[0]
    cands = [(x[:-2], int(x[-1])) for x in r.imps if x[:-2] in news.index]
    t = time.time()
    out = decide_many([("noul", jev_state(news, r["hist_"], nid), QUESTION) for nid, _ in cands])
    secs = time.time() - t
    ranked = sorted(((o["true"], nid, y) for o, (nid, y) in zip(out, cands)), reverse=True)
    history = [headline(news, h) for h in r["hist_"] if h in news.index][-10:]
    return history, ranked, secs


def headline(news, nid):
    r = news.loc[nid]
    return f"{r['cat']} › {r['sub']}: {r['title']}"


def jev_state(news, history, nid):
    r = news.loc[nid]
    return {"news_recently_clicked_by_this_user (oldest to newest)": [headline(news, h) for h in history if h in news.index][-20:],
            "candidate_article": {"category": f"{r['cat']} › {r['sub']}", "title": r["title"], "abstract": r["abstract"][:300]}}


def eligible(b):
    return b[(b["hist_"].str.len() >= 5) & (b.imps.str.len() <= 50) & b.imps.map(lambda xs: any(x.endswith("-1") for x in xs))]


def build():
    from sklearn.feature_extraction.text import TfidfVectorizer
    tb, db, news = load_mind()
    events = collections.defaultdict(list)          # click times per article, for "popularity so far"
    for b in (tb, db):
        for t, imps in zip(b.t, b.imps):
            for x in imps:
                if x.endswith("-1"):
                    events[x[:-2]].append(t.value)
    for v in events.values():
        v.sort()
    tfidf = TfidfVectorizer(min_df=2, stop_words="english").fit(news.title)
    pos = dict(zip(news.index, range(len(news))))
    TM = tfidf.transform(news.title)

    def rows_for(sample, split):
        out = []
        for r in sample.itertuples():
            hist = [h for h in r.hist_ if h in pos][-50:]
            hc = collections.Counter(news.loc[h, "cat"] for h in hist)
            hs = collections.Counter(news.loc[h, "sub"] for h in hist)
            HM = TM[[pos[h] for h in hist]]
            for x in r.imps:
                nid, y = x[:-2], int(x[-1])
                if nid not in pos:
                    continue
                sim = (HM @ TM[pos[nid]].T).toarray().ravel()
                out.append({"split": split, "iid": int(r.iid), "nid": nid, "y": y, "pop": math.log1p(bisect.bisect_left(events.get(nid, []), r.t.value)),
                            "cat_share": hc[news.loc[nid, "cat"]] / len(hist), "sub_share": hs[news.loc[nid, "sub"]] / len(hist),
                            "sim_max": float(sim.max()), "sim_mean": float(sim.mean()), "hist_len": len(hist), "cat": news.loc[nid, "cat"],
                            "_hist": r.hist_})
        return out

    ev = eligible(db).sample(N_EVAL, random_state=0)
    tr = eligible(tb).sample(N_TRAIN_BIG, random_state=0)
    rows = rows_for(ev, "eval") + rows_for(tr, "train")
    stack_iids = set(tr.iid.iloc[:N_TRAIN_STACK])
    for r in rows:
        r["stack"] = r["split"] == "train" and r["iid"] in stack_iids

    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    todo = [r for r in rows if (r["split"] == "eval" or r["stack"]) and f"{r['split']}:{r['iid']}:{r['nid']}" not in cache]
    print(f"{len(rows)} candidate rows, {len(todo)} JEV decisions to run", flush=True)
    t0 = time.time()
    for k in range(0, len(todo), 2000):
        chunk = todo[k: k + 2000]
        out = decide_many([("noul", jev_state(news, r["_hist"], r["nid"]), QUESTION) for r in chunk])
        cache.update({f"{r['split']}:{r['iid']}:{r['nid']}": o["true"] for r, o in zip(chunk, out)})
        json.dump(cache, open(CACHE, "w"))
        print(f"  {k + len(chunk)}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    for r in rows:
        r["jev"] = cache.get(f"{r['split']}:{r['iid']}:{r['nid']}")
        r.pop("_hist")
    json.dump({"jev_seconds": time.time() - t0, "rows": rows}, open(os.path.join(HERE, "features.json"), "w"))


def evaluate():
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    d = pd.DataFrame(json.load(open(os.path.join(HERE, "features.json")))["rows"])
    d["cat"] = d.cat.astype("category")
    ev, tr = d[d.split == "eval"].copy(), d[d.split == "train"]
    F = ["pop", "cat_share", "sub_share", "sim_max", "sim_mean", "hist_len", "cat"]

    def ranker(train, feats):
        train = train.sort_values("iid", kind="stable")
        m = lgb.LGBMRanker(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20, verbose=-1, random_state=0)
        m.fit(train[feats], train.y, group=train.groupby("iid", sort=True).size().values)
        return m.predict(ev[feats])
    stack = tr[tr["stack"]]
    ev["lgb"] = ranker(stack, F)
    ev["lgb_jev"] = ranker(stack, F + ["jev"])
    ev["lgb_big"] = ranker(tr, F)
    ev["random"] = np.random.RandomState(0).rand(len(ev))

    def metrics(g, col):
        y = g.y.values
        s = g[col].values + 1e-9 * np.random.RandomState(1).rand(len(g))
        ys = y[np.argsort(-s)]
        dcg = lambda k: sum(ys[i] / math.log2(i + 2) for i in range(min(k, len(ys))))
        idcg = lambda k: sum(1 / math.log2(i + 2) for i in range(min(k, int(y.sum()))))
        return [roc_auc_score(y, s), (ys / np.arange(1, len(ys) + 1)).sum() / y.sum(), dcg(5) / idcg(5), dcg(10) / idcg(10)]

    per = [{c: metrics(g, c) for c in NAMES} for _, g in ev.groupby("iid") if g.y.nunique() == 2]
    res = {c: np.mean([p[c] for p in per], axis=0).tolist() for c in NAMES}
    # paired bootstrap over impressions for the two headline comparisons
    rng = np.random.RandomState(0)
    boot = {}
    for a, b in (("jev", "cat_share"), ("lgb_jev", "lgb")):
        diff = np.array([p[a][0] - p[b][0] for p in per])
        bs = sorted(diff[rng.randint(0, len(diff), len(diff))].mean() for _ in range(5000))
        boot[f"{a}-{b}"] = [float(diff.mean()), float(bs[125]), float(bs[4875])]
    out = {"n_impressions": len(per), "n_candidates": int(len(ev)), "click_share": float(ev.y.mean()),
           "n_train_impressions_stack": int(stack.iid.nunique()), "n_train_impressions_big": int(tr.iid.nunique()), "metrics": res, "bootstrap_auc": boot}
    json.dump(out, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
    return out


def summary():
    s = json.load(open(os.path.join(HERE, "summary.json")))
    m, b = s["metrics"], s["bootstrap_auc"]
    row = lambda c, bold=False: (f"| {'**' if bold else ''}{NAMES[c]}{'**' if bold else ''} | " +
                                 " | ".join(f"{'**' if bold else ''}{v:.3f}{'**' if bold else ''}" for v in m[c]) + " |")
    md = [f"{s['n_impressions']} impressions from 15 Nov 2019 · {s['n_candidates']:,} candidate articles · {100 * s['click_share']:.1f}% clicked\n",
          "### 1. Zero-shot: no training on MIND\n", "| method | AUC | MRR | nDCG@5 | nDCG@10 |", "|---|---:|---:|---:|---:|"]
    md += [row(c) for c in ("random", "pop", "cat_share", "sim_max")] + [row("jev", True)]
    md.append(f"\nAUC gain of JEV over the best zero-shot baseline: {b['jev-cat_share'][0]:+.3f} (95% bootstrap interval {b['jev-cat_share'][1]:+.3f} to {b['jev-cat_share'][2]:+.3f})")
    md += [f"\n### 2. Trained on MIND ({s['n_train_impressions_stack']:,} training impressions, same data for both)\n",
           "| method | AUC | MRR | nDCG@5 | nDCG@10 |", "|---|---:|---:|---:|---:|", row("lgb"), row("lgb_jev", True),
           f"| {NAMES['lgb_big']} ({s['n_train_impressions_big']:,} impressions) | " + " | ".join(f"{v:.3f}" for v in m["lgb_big"]) + " |"]
    md.append(f"\nAUC gain from adding the JEV feature: {b['lgb_jev-lgb'][0]:+.3f} (95% bootstrap interval {b['lgb_jev-lgb'][1]:+.3f} to {b['lgb_jev-lgb'][2]:+.3f})")
    open(os.path.join(HERE, "sample_output.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    plt = style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    zs = ["random", "pop", "cat_share", "sim_max", "jev"]
    a1.bar([NAMES[c].replace(" (", "\n(").replace(", zero-shot", "\nzero-shot") for c in zs], [m[c][0] for c in zs], color=[BASE] * 4 + [JEV])
    for i, c in enumerate(zs):
        a1.text(i, m[c][0] + 0.004, f"{m[c][0]:.3f}", ha="center", fontsize=9)
    a1.set_ylim(0.45, max(m[c][0] for c in zs) + 0.04); a1.set_ylabel("AUC (per impression)"); a1.tick_params(axis="x", labelsize=8)
    a1.set_title("Zero-shot: nobody trained on MIND")
    tt = ["lgb", "lgb_jev", "lgb_big"]
    labels = [f"LightGBM\n{s['n_train_impressions_stack']:,} impressions", f"LightGBM + JEV score\n{s['n_train_impressions_stack']:,} impressions",
              f"LightGBM\n{s['n_train_impressions_big']:,} impressions"]
    a2.bar(labels, [m[c][0] for c in tt], color=[BASE, JEV, "#CED4DA"])
    for i, c in enumerate(tt):
        a2.text(i, m[c][0] + 0.004, f"{m[c][0]:.3f}", ha="center", fontsize=9)
    a2.axhline(m["jev"][0], color=JEV, ls="--", lw=1)
    a2.text(2.45, m["jev"][0] + 0.004, "JEV alone, zero-shot", color=JEV, ha="right", fontsize=8)
    a2.set_ylim(0.45, max(m[c][0] for c in tt + ["jev"]) + 0.04); a2.tick_params(axis="x", labelsize=8)
    a2.set_title("Trained on MIND: add JEV as one feature")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "..", "assets", "news_recommendation.png")); plt.close(fig)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "all":
        build()
    if mode in ("all", "eval"):
        evaluate()
    summary()
