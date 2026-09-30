"""Demo 07 — zero-shot short-video recommendation (TikTok-style feeds): JEV-27B looks at the video covers.

JEV-27B has never been trained on MicroLens, on recommendation, or on any click data. System 1 looks at the covers of the
last 5 videos a user watched and at one candidate cover, and returns P(this user clicks it) in one forward pass.
Needs the multimodal server: bash common/serve_jev27b_mm.sh

Data: MicroLens-100k (Westlake University, https://github.com/westlake-repl/MicroLens): 100,000 users of a short-video
app, raw cover images. Downloaded at run time from the official site; not redistributed here (this folder stores only
user/video IDs and scores).
Task: for 200 random users, the last video they watched is hidden among 19 videos other users watched within +/- 3 days
(what a feed would be showing at that time). Every method ranks the 20 candidates.

    python demo.py            # download (~700 MB), score with JEV (3 x 4,000 decisions), tables + chart
    python demo.py summary    # from results.json
"""
import collections, concurrent.futures as cf, json, math, os, random, sys, time, zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from charts import JEV, BASE, ALT, style  # noqa: E402
import jev_client as j  # noqa: E402

DATA = os.environ.get("MICROLENS_DIR", os.path.expanduser("~/.cache/jev-demo/microlens"))
SITE = "https://recsys.westlake.edu.cn/MicroLens-100k-Dataset/"
N_USERS, N_NEG, N_HIST = 200, 19, 5
Q = "Is this scenario one where: this user clicks on the candidate video?"
NAMES = {"random": "random order", "pop": "popularity so far", "title_sim": "title similarity (TF-IDF)",
         "jev_titles": "JEV-27B, titles only", "jev_images": "JEV-27B, covers only", "jev_both": "JEV-27B, covers + titles",
         "itemcf": "item-based collaborative filtering"}


def download():
    import requests
    os.makedirs(DATA, exist_ok=True)
    for f in ("MicroLens-100k_pairs.csv", "MicroLens-100k_title_en.csv", "MicroLens-100k_covers.zip"):
        if not os.path.exists(os.path.join(DATA, f)):
            print("downloading", f, flush=True)
            with requests.get(SITE + f, stream=True, timeout=600) as r, open(os.path.join(DATA, f), "wb") as out:
                for chunk in r.iter_content(1 << 20):
                    out.write(chunk)
    if not os.path.isdir(os.path.join(DATA, "covers")):
        zipfile.ZipFile(os.path.join(DATA, "MicroLens-100k_covers.zip")).extractall(os.path.join(DATA, "covers"))


def cover(i):
    """Cover resized to at most 448 px (about 144 vision tokens)."""
    from PIL import Image
    out = os.path.join(DATA, "covers_448", f"{i}.jpg")
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        im = Image.open(os.path.join(DATA, "covers", f"{i}.jpg")).convert("RGB")
        im.thumbnail((448, 448))
        im.save(out, quality=85)
    return out


def titles():
    return pd.read_csv(os.path.join(DATA, "MicroLens-100k_title_en.csv"), header=None, names=["item", "title"]).set_index("item").title.to_dict()


def prepare():
    """Deterministic candidate lists: 200 users x (1 hidden last video + 19 videos others watched within +/- 3 days)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    p = pd.read_csv(os.path.join(DATA, "MicroLens-100k_pairs.csv")).sort_values(["user", "timestamp"])
    tt = titles()
    seqs = p.groupby("user").agg(list)
    seqs = seqs[seqs["item"].str.len() >= N_HIST + 1]
    rng = random.Random(0)
    users = rng.sample(list(seqs.index), N_USERS)
    ts = p.sort_values("timestamp")
    T, I = ts.timestamp.values, ts["item"].values
    day = 86_400_000
    pop_times = collections.defaultdict(list)
    for it, t in zip(I, T):
        pop_times[it].append(t)
    test = set(users)
    cooc = collections.defaultdict(collections.Counter)   # co-occurrence in OTHER users' recent histories
    for u, row in seqs.iterrows():
        if u not in test:
            its = row["item"][-20:]
            for a in its:
                for b in its:
                    if a != b:
                        cooc[a][b] += 1
    rows = []
    for u in users:
        items, times = seqs.loc[u, "item"], seqs.loc[u, "timestamp"]
        target, t_target, hist = items[-1], times[-1], items[-1 - N_HIST:-1]
        lo, hi = np.searchsorted(T, t_target - 3 * day), np.searchsorted(T, t_target + 3 * day)
        seen = set(items)
        pool = [x for x in I[lo:hi] if x not in seen]
        negs = []
        while len(negs) < N_NEG:
            x = rng.choice(pool)
            if x not in negs:
                negs.append(x)
        for c in [target] + negs:
            rows.append({"user": int(u), "item": int(c), "y": int(c == target), "hist_items": [int(h) for h in hist],
                         "pop": math.log1p(np.searchsorted(np.array(pop_times[c]), t_target)), "itemcf": sum(cooc[h][c] for h in hist)})
    df = pd.DataFrame(rows)
    tf = TfidfVectorizer(min_df=1).fit([str(tt.get(i, "")) for i in set(df["item"]) | {h for hs in df.hist_items for h in hs}])
    df["title_sim"] = [float((tf.transform([str(tt.get(r.item, ""))]) @ tf.transform([str(tt.get(h, "")) for h in r.hist_items]).T).max())
                       for r in df.itertuples()]
    json.dump({str(u): {"history": g.hist_items.iloc[0], "candidates": g["item"].tolist(), "clicked": int(g[g.y == 1]["item"].iloc[0])}
               for u, g in df.groupby("user")}, open(os.path.join(HERE, "candidates.json"), "w"))
    return df


def request(hist, item, mode, tt):
    """One System 1 request. mode: titles | images | both."""
    if mode == "titles":
        return ("noul", {"videos_recently_watched_by_this_user (oldest to newest)": [tt.get(h, "") for h in hist],
                         "candidate_video_title": tt.get(item, "")}, Q)
    parts = ["Short-video app. Covers" + (" and titles" if mode == "both" else "") +
             " of the videos this user watched most recently (oldest to newest):\n"]
    for k, h in enumerate(hist):
        parts += [f"{k + 1}. "] + ([f"“{tt.get(h, '')}” "] if mode == "both" else []) + [{"image": cover(h)}, "\n"]
    parts += ["Candidate video" + (f" “{tt.get(item, '')}”" if mode == "both" else "") + ": ", {"image": cover(item)}]
    return ("mm", parts, Q)


def score(reqs, workers=16):
    with cf.ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda a: (j.decide_mm("noul", a[1], a[2]) if a[0] == "mm" else j.decide(*a))["true"], reqs))


def run():
    download()
    df = prepare()
    tt = titles()
    for mode in ("titles", "images", "both"):
        t0 = time.time()
        df[f"jev_{mode}"] = score([request(r.hist_items, r.item, mode, tt) for r in df.itertuples()])
        print(f"{mode}: {len(df)} decisions in {time.time() - t0:.0f}s", flush=True)
    df.drop(columns=["hist_items"]).to_json(os.path.join(HERE, "results.json"), orient="records")


def rank_user(user):
    """Live, for the app: rank one user's 20 candidates by their covers."""
    c = json.load(open(os.path.join(HERE, "candidates.json")))[str(user)]
    tt = titles()
    t0 = time.time()
    ps = score([request(c["history"], i, "images", tt) for i in c["candidates"]])
    ranked = sorted(zip(ps, c["candidates"]), reverse=True)
    return c["history"], ranked, c["clicked"], time.time() - t0


def summary():
    df = pd.DataFrame(json.load(open(os.path.join(HERE, "results.json"))))
    df["random"] = np.random.RandomState(0).rand(len(df))

    def m(g, col):
        s = g[col].values + 1e-9 * np.random.RandomState(1).rand(len(g))
        rank = int((s > s[g.y.values == 1][0]).sum()) + 1
        return [(len(g) - rank) / (len(g) - 1), float(rank == 1), float(rank <= 5), 1 / math.log2(rank + 1) if rank <= 10 else 0.0, 1 / rank]
    per = {c: np.array([m(g, c) for _, g in df.groupby("user")]) for c in NAMES}
    res = {c: per[c].mean(0).tolist() for c in NAMES}
    rng = np.random.RandomState(0)
    boot = {}
    for a, b in (("jev_images", "jev_titles"), ("jev_images", "title_sim"), ("jev_images", "itemcf")):
        d = per[a][:, 0] - per[b][:, 0]
        bs = sorted(d[rng.randint(0, len(d), len(d))].mean() for _ in range(5000))
        boot[f"{a}-{b}"] = [float(d.mean()), float(bs[125]), float(bs[4875])]
    json.dump({"metrics": res, "bootstrap_auc": boot, "n_users": int(df.user.nunique())}, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
    row = lambda c, bold=False: f"| {'**' if bold else ''}{NAMES[c]}{'**' if bold else ''} | " + " | ".join(
        f"{'**' if bold else ''}{v:.3f}{'**' if bold else ''}" for v in res[c]) + " |"
    md = [f"{df.user.nunique()} users · 20 candidates each (1 hidden last video + 19 videos others watched within ±3 days)\n",
          "### Zero-shot (no interaction data used)\n", "| method | AUC | HR@1 | HR@5 | NDCG@10 | MRR |", "|---|---:|---:|---:|---:|---:|"]
    md += [row(c) for c in ("random", "pop", "title_sim", "jev_titles")] + [row("jev_images", True), row("jev_both")]
    md += ["\n### Reference: learned from other users' interaction logs\n", "| method | AUC | HR@1 | HR@5 | NDCG@10 | MRR |", "|---|---:|---:|---:|---:|---:|", row("itemcf")]
    md += ["\nPaired bootstrap of the AUC difference (95% interval):"] + [f"- {NAMES[k.split('-')[0]]} − {NAMES[k.split('-')[1]]}: {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for k, v in boot.items()]
    open(os.path.join(HERE, "sample_output.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    plt = style()
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    order = ["random", "pop", "title_sim", "jev_titles", "jev_both", "jev_images", "itemcf"]
    labels = ["random", "popularity\nso far", "title\nsimilarity", "JEV\ntitles only", "JEV\ncovers + titles", "JEV\ncovers only", "item-based CF\n(uses 59,045 users' logs)"]
    cols = [BASE, BASE, BASE, "#B197FC", "#9775FA", JEV, "#CED4DA"]
    ax.bar(labels, [res[c][0] for c in order], color=cols)
    for i, c in enumerate(order):
        ax.text(i, res[c][0] + 0.006, f"{res[c][0]:.3f}", ha="center", fontsize=9)
    ax.axvline(5.5, color="#ADB5BD", ls=":", lw=1)
    ax.text(2.5, 0.765, "zero-shot: no interaction data", ha="center", fontsize=9, color="#495057")
    ax.text(6, 0.765, "reference", ha="center", fontsize=9, color="#495057")
    ax.set_ylim(0.45, 0.79); ax.set_ylabel("AUC (per user)"); ax.tick_params(axis="x", labelsize=8)
    ax.set_title("Zero-shot short-video recommendation: JEV-27B looking at covers matches collaborative filtering")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "..", "assets", "image_recommendation.png")); plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "summary":
        run()
    summary()
