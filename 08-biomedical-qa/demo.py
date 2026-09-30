"""Demo 08 — biomedical research questions (PubMedQA), zero-shot, one forward pass.

PubMedQA asks a research question about a PubMed abstract (e.g. "Do preoperative statins reduce atrial fibrillation after
coronary artery bypass grafting?") and expects yes / no / maybe. Reasoning-required setting: the model sees the question
and the abstract without its conclusion. JEV-27B System 1 answers with a probability for each of the three options in one
forward pass, with no training on PubMedQA and no examples in the prompt.

Official test set: the 500 expert-labelled questions used by the PubMedQA leaderboard.

    python demo.py            # ~30 s of model time on one GPU
    python demo.py summary    # tables + chart from results.json
"""
import json, os, sys, time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from charts import JEV, BASE, style  # noqa: E402
from jev_client import decide, decide_many  # noqa: E402

Q = "Based on this abstract, what is the answer to the research question?"
OPTS = ["yes", "no", "maybe"]
# PubMedQA leaderboard, reasoning-required setting (https://pubmedqa.github.io, last update 2024-04-28): accuracy %
LEADERBOARD = [("GPT-4 (Medprompt)", 82.0), ("Med-PaLM 2", 81.8), ("MEDITRON 70B", 81.6), ("Claude 3", 79.7), ("Flan-PaLM 540B (3-shot)", 79.0),
               ("Human performance", 78.0), ("Galactica 120B", 77.6), ("GPT-4 (Nori et al. 2023)", 75.2), ("PubMedGPT 2.7B", 74.4),
               ("PMC-LLaMA 7B", 73.4), ("BioLinkBERT large", 72.2), ("BioBERT", 68.1)]


def load():
    import requests
    from huggingface_hub import hf_hub_download
    df = pd.read_parquet(hf_hub_download("qiaojin/PubMedQA", "pqa_labeled/train-00000-of-00001.parquet", repo_type="dataset"))
    test = requests.get("https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json", timeout=60).json()
    return df[df.pubid.astype(str).isin(test)].reset_index(drop=True)


def state(row):
    c = row["context"]
    return {"research_question": row["question"], "abstract": "\n".join(f"{l}: {t}" for l, t in zip(c["labels"], c["contexts"]))}


def answer(row):
    return decide("choice", state(row), Q, OPTS)


def run():
    df = load()
    t0 = time.time()
    out = decide_many([("choice", state(r), Q, OPTS) for _, r in df.iterrows()], workers=16)
    secs = time.time() - t0
    rows = [{"pubid": int(r.pubid), "expert": r.final_decision, "jev": max(o, key=o.get), "probs": o} for r, o in zip(df.itertuples(), out)]
    json.dump({"seconds": secs, "rows": rows}, open(os.path.join(HERE, "results.json"), "w"), indent=1)


def summary():
    from sklearn.metrics import f1_score
    d = json.load(open(os.path.join(HERE, "results.json")))
    r = pd.DataFrame(d["rows"])
    acc, f1 = 100 * (r.jev == r.expert).mean(), 100 * f1_score(r.expert, r.jev, average="macro")
    board = sorted(LEADERBOARD + [("JEV-27B System 1 (zero-shot, one pass)", acc)], key=lambda x: -x[1])
    md = [f"{len(r)} expert-labelled questions (official test set) · {len(r)} decisions in {d['seconds']:.0f} s\n",
          f"**Accuracy {acc:.1f}% · macro-F1 {f1:.1f}%**\n", "| model | accuracy (%) |", "|---|---:|"]
    md += [f"| {'**' if n.startswith('JEV') else ''}{n}{'**' if n.startswith('JEV') else ''} | {'**' if n.startswith('JEV') else ''}{a:.1f}{'**' if n.startswith('JEV') else ''} |" for n, a in board]
    cm = pd.crosstab(r.expert, r.jev).reindex(index=OPTS, columns=OPTS, fill_value=0)
    md += ["\nConfusion (rows = expert answer, columns = JEV):\n", "| expert \\ JEV | yes | no | maybe |", "|---|---:|---:|---:|"]
    md += [f"| {k} | " + " | ".join(str(int(v)) for v in cm.loc[k]) + " |" for k in OPTS]
    open(os.path.join(HERE, "sample_output.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    plt = style()
    fig, ax = plt.subplots(figsize=(9, 5))
    names, vals = [n for n, _ in board][::-1], [a for _, a in board][::-1]
    cols = [JEV if n.startswith("JEV") else "#FFA94D" if n.startswith("Human") else BASE for n in names]
    ax.barh(names, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(v + 0.2, i, f"{v:.1f}", va="center", fontsize=8)
    ax.set_xlim(60, 86); ax.set_xlabel("accuracy on the PubMedQA test set (%)"); ax.tick_params(axis="y", labelsize=8)
    ax.set_title("PubMedQA: zero-shot, one forward pass, at human-expert level")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "..", "assets", "pubmedqa.png")); plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "summary":
        run()
    summary()
