"""Demo 09 — multimodal judge: which answer about an image is better? (VL-RewardBench)

Given an image, a user's question and two candidate answers, JEV-27B-VL System 1 returns P(answer A is better) in one forward
pass (asked in both orders and averaged). Its decision head was trained on text only: judging over images is zero-shot.
Needs the multimodal server: bash common/serve_jev27b_mm.sh

VL-RewardBench (CVPR 2025, 1,247 human-verified pairs): general questions from real users, visual hallucination detection,
and multimodal knowledge / math reasoning. Comparison rows come from the official leaderboard (26 models, updated 2025-05-11).

    python demo.py            # ~3 minutes on one GPU
    python demo.py summary    # tables + chart from results.json
"""
import concurrent.futures as cf, io, json, os, re, sys, time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from charts import JEV, BASE, ALT, style  # noqa: E402
import jev_client as j  # noqa: E402

DATA = os.environ.get("VLRB_DIR", os.path.expanduser("~/.cache/jev-demo/vlrewardbench"))
Q = ("Which response answers the user's question about the image better: more accurate to what is actually in the image, "
     "correct, helpful, and free of hallucinated details?")
OPTS = ["Response A", "Response B"]
# official leaderboard (Google sheet linked from https://huggingface.co/spaces/MMInstruction/VL-RewardBench): general, hallucination, reasoning, overall, macro
LEADERBOARD = [("Skywork-VL-Reward-7B", 65.6, 80.2, 61.3, 73.3, 69.0), ("Gemini 2.0 Flash", 50.8, 72.6, 70.1, 68.8, 64.5),
               ("Gemini 1.5 Pro", 50.8, 72.5, 64.2, 67.2, 62.5), ("GPT-4o", 49.1, 67.6, 70.5, 65.8, 62.4),
               ("Gemini 1.5 Flash", 47.8, 59.6, 58.4, 57.6, 55.3), ("Llama-3.2-90B-Vision", 42.6, 57.3, 61.7, 56.2, 53.9),
               ("Claude 3.5 Sonnet", 43.4, 55.0, 62.3, 55.3, 53.6), ("Qwen2-VL-72B", 38.1, 32.8, 58.0, 39.5, 43.0)]
cut = lambda s, n=4000: s if len(s) <= n else s[:n] + " …[truncated]"


def load():
    from huggingface_hub import hf_hub_download
    from PIL import Image
    df = pd.read_parquet(hf_hub_download("MMInstruction/VL-RewardBench", "data/test-00000-of-00001.parquet", repo_type="dataset"))
    os.makedirs(os.path.join(DATA, "img"), exist_ok=True)
    paths = []
    for r in df.itertuples():
        p = os.path.join(DATA, "img", f"{r.id}.jpg")
        if not os.path.exists(p):
            im = Image.open(io.BytesIO(r.image["bytes"])).convert("RGB")
            im.thumbnail((768, 768))
            im.save(p, quality=90)
        paths.append(p)
    df["path"] = paths
    pre = df.id.map(lambda s: re.split(r"[_\-:/0-9]", str(s))[0].lower())
    df["domain"] = pre.map(lambda p: "hallucination" if p in ("hallucination", "rlaif", "rlhf") else "reasoning" if p in ("mathverse", "mmmu") else "general")
    return df


def parts(path, question, a, b):
    return ["User's question about the image: " + question + "\nImage: ", {"image": path}, "\n\n[Response A]\n" + cut(a) + "\n\n[Response B]\n" + cut(b)]


def judge(path, question, a, b):
    """P(a is better than b), both presentation orders averaged."""
    p1 = j.decide_mm("choice", parts(path, question, a, b), Q, OPTS)["Response A"]
    p2 = j.decide_mm("choice", parts(path, question, b, a), Q, OPTS)["Response B"]
    return (p1 + p2) / 2, p1, p2


def run():
    df = load()
    t0 = time.time()
    with cf.ThreadPoolExecutor(8) as ex:
        out = list(ex.map(lambda r: judge(r.path, r.query, r.response[0], r.response[1]), df.itertuples()))
    secs = time.time() - t0
    rows = [{"id": r.id, "domain": r.domain, "first_is_better": bool(list(r.human_ranking)[0] < list(r.human_ranking)[1]),
             "p_first_better": p, "p_order_ab": a, "p_order_ba": b, "order_agree": bool((a > 0.5) == (b > 0.5))}
            for r, (p, a, b) in zip(df.itertuples(), out)]
    json.dump({"seconds": secs, "rows": rows}, open(os.path.join(HERE, "results.json"), "w"), indent=1)


def summary():
    d = json.load(open(os.path.join(HERE, "results.json")))
    r = pd.DataFrame(d["rows"])
    r["correct"] = (r.p_first_better > 0.5) == r.first_is_better
    dom = {k: 100 * r[r.domain == k].correct.mean() for k in ("general", "hallucination", "reasoning")}
    overall, macro = 100 * r.correct.mean(), float(np.mean(list(dom.values())))
    rng = np.random.RandomState(0)
    c = r.correct.values.astype(float)
    bs = sorted(c[rng.randint(0, len(c), len(c))].mean() for _ in range(5000))
    agree = 100 * r.order_agree.mean()
    board = sorted(LEADERBOARD + [("JEV-27B-VL System 1", dom["general"], dom["hallucination"], dom["reasoning"], overall, macro)], key=lambda x: -x[4])
    md = [f"{len(r):,} pairs · {2 * len(r):,} System 1 decisions in {d['seconds']:.0f} s · overall accuracy 95% interval "
          f"[{100 * bs[125]:.1f}, {100 * bs[4875]:.1f}] · verdict unchanged when A and B are swapped: {agree:.0f}%\n",
          "| judge | general | hallucination | reasoning | **overall** | macro average |", "|---|---:|---:|---:|---:|---:|"]
    for n, g, h, rs, o, m in board:
        b = "**" if n.startswith("JEV") else ""
        md.append(f"| {b}{n}{b} | {g:.1f} | {h:.1f} | {rs:.1f} | {b}{o:.1f}{b} | {b}{m:.1f}{b} |")
    open(os.path.join(HERE, "sample_output.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    plt = style()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    names, vals = [x[0] for x in board][::-1], [x[4] for x in board][::-1]
    ax.barh(names, vals, color=[JEV if n.startswith("JEV") else ALT if "Skywork" in n else BASE for n in names])
    for i, v in enumerate(vals):
        ax.text(v + 0.4, i, f"{v:.1f}", va="center", fontsize=9)
    ax.axvline(50, color="#ADB5BD", ls=":", lw=1)
    ax.set_xlim(30, 85); ax.set_xlabel("overall accuracy on VL-RewardBench (%)"); ax.tick_params(axis="y", labelsize=9)
    ax.set_title("Multimodal judge: JEV-27B-VL System 1 vs the VL-RewardBench leaderboard")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "..", "assets", "multimodal_judge.png")); plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "summary":
        run()
    summary()
