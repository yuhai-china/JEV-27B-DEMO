"""Demo 10 — agent judge: did the web agent really finish the task? (AgentRewardBench)

JEV-27B-VL System 1 reads the user's goal, the agent's action history and final message, looks at the final screenshot, and
returns P(task completed) in one forward pass. Zero-shot. Needs the multimodal server: bash common/serve_jev27b_mm.sh

AgentRewardBench (McGill-NLP, 2025): 1,302 web-agent trajectories (WebArena, VisualWebArena, WorkArena, AssistantBench) by
GPT-4o, Claude 3.7 Sonnet, Llama 3.3 and Qwen2.5-VL agents, with expert success labels. Leaderboard numbers are read from the
official leaderboard space (McGill-NLP/agent-reward-bench-leaderboard).

    python demo.py            # download the needed files (~0.5 GB), judge all 1,302 trajectories (~4 min), tables + chart
    python demo.py summary    # from results.json
"""
import concurrent.futures as cf, json, os, re, sys, time

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from charts import JEV, ALT, WARN, style  # noqa: E402
import jev_client as j  # noqa: E402

R = "McGill-NLP/agent-reward-bench"
DATA = os.environ.get("ARB_DIR", os.path.expanduser("~/.cache/jev-demo/agent-reward-bench"))
TXT_JUDGE = "gpt-4o-mini-noscreen-noaxtree"      # its prompt files carry the goal, action history and the agent's final message as text
Q = "Is this scenario one where: the agent successfully completed the user's task (the goal was fully achieved, or the requested information was correctly given to the user)?"


def prepare():
    from PIL import Image
    os.makedirs(os.path.join(DATA, "shots"), exist_ok=True)
    files = [s.rfilename for s in HfApi().dataset_info(R).siblings]
    jfiles = [f for f in files if f.startswith("judgments/") and f"/{TXT_JUDGE}/" in f and f.endswith(".json")]
    last = {}
    for f in files:
        m = re.match(r"screenshots/([^/]+)/([^/]+)/([^/]+)/screenshot_step_(\d+)\.png$", f)
        if m and int(m.group(4)) > last.get(m.groups()[:3], (-1, ""))[0]:
            last[m.groups()[:3]] = (int(m.group(4)), f)

    def one(f):
        _, bench, agent, _, name = f.split("/")
        task = name[:-5]
        d = json.load(open(hf_hub_download(R, f, repo_type="dataset")))
        user = [m for m in d["chat_messages"]["regular"] if m["role"] == "user"][-1]["content"]
        text = user if isinstance(user, str) else " ".join(c.get("text", "") for c in user if isinstance(c, dict))
        shot = os.path.join(DATA, "shots", f"{bench}_{agent}_{task}.jpg")
        if not os.path.exists(shot):
            im = Image.open(hf_hub_download(R, last[(bench, agent, task)][1], repo_type="dataset")).convert("RGB")
            im.thumbnail((1024, 1024)); im.save(shot, quality=88)
        return {"benchmark": bench, "agent": agent, "task_id": task, "text": text, "shot": shot}

    with cf.ThreadPoolExecutor(32) as ex:
        return list(ex.map(one, jfiles))


def clip(t, head=1500, tail=7000):
    return t if len(t) <= head + tail else t[:head] + "\n[... middle steps omitted ...]\n" + t[-tail:]


def judge(text, screenshot):
    """P(the agent completed the task)."""
    parts = ["Evaluating a web-browsing AI agent.\n" + clip(text), "\n\nFinal screenshot of the web page: ", {"image": screenshot}]
    try:
        return j.decide_mm("noul", parts, Q)["true"]
    except Exception:          # rare over-long prompts: retry with a shorter history
        return j.decide_mm("noul", [parts[0][:6000]] + parts[1:], Q)["true"]


def run():
    rows = prepare()
    ann = pd.read_csv(hf_hub_download(R, "data/annotations.csv", repo_type="dataset"))
    ann = ann[ann.trajectory_success != "Unsure"]
    lab = ann.groupby(["benchmark", "task_id", "model_name"]).trajectory_success.agg(lambda s: s.value_counts().index[0]).to_dict()
    with cf.ThreadPoolExecutor(16) as ex:
        ps = list(ex.map(lambda r: judge(r["text"], r["shot"]), rows))
    out = [{"benchmark": r["benchmark"], "agent": r["agent"], "task_id": r["task_id"], "p": p, "label": lab.get((r["benchmark"], r["task_id"], r["agent"]))}
           for r, p in zip(rows, ps)]
    out = [o | {"y": o["label"] == "Successful"} for o in out if o["label"]]
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1)


def summary():
    df = pd.read_json(os.path.join(HERE, "results.json"))
    y = df.y.values.astype(bool)
    order = np.argsort(-df.p.values)
    cum = np.cumsum(y[order])

    def prec_at(rec):
        k = int(np.ceil(rec / 100 * y.sum())); n = int(np.searchsorted(cum, k)) + 1
        return 100 * k / n
    board = [json.loads(l) for l in open(hf_hub_download("McGill-NLP/agent-reward-bench-leaderboard", "results.jsonl", repo_type="space")) if l.strip()]
    pred = df.p.values > 0.5
    tp = (pred & y).sum()
    from sklearn.metrics import roc_auc_score
    md = [f"{len(df):,} trajectories · {100 * y.mean():.1f}% successful (expert labels) · AUROC {roc_auc_score(y, df.p):.3f}\n",
          f"JEV at threshold 0.5: precision {100 * tp / pred.sum():.1f}, recall {100 * tp / y.sum():.1f}\n",
          "| judge | its precision | its recall | JEV precision at the same recall | difference |", "|---|---:|---:|---:|---:|"]
    for b in sorted(board, key=lambda x: -x["Overall"]):
        jp = prec_at(b["Recall"])
        md.append(f"| {b['Judge']} | {b['Overall']:.1f} | {b['Recall']:.1f} | **{jp:.1f}** | {jp - b['Overall']:+.1f} |")
    open(os.path.join(HERE, "sample_output.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    prec = cum / np.arange(1, len(y) + 1) * 100; rec = cum / y.sum() * 100
    plt = style()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(rec, prec, color=JEV, lw=2.2, label="JEV-27B-VL System 1 (one pass, zero-shot)")
    ax.scatter([100 * tp / y.sum()], [100 * tp / pred.sum()], color=JEV, s=70, zorder=5, edgecolor="white")
    ax.annotate("JEV at threshold 0.5", (100 * tp / y.sum(), 100 * tp / pred.sum()), textcoords="offset points", xytext=(8, 6), fontsize=8, color=JEV)
    for b in board:
        col = WARN if b["Judge"] == "Rule-based" else ALT if ("7B" in b["Judge"] or "WebJudge" in b["Judge"]) else "#495057"
        ax.scatter(b["Recall"], b["Overall"], color=col, s=28, zorder=4)
        if b["Judge"] in ("Rule-based", "WebJudge (o4-mini)", "WebJudge-7B", "World-State-Model-7B", "GPT-4o (A)", "Claude 3.7 S. (S)", "Qwen2.5-VL (S)", "NNetNav (Llama-3.3 70B)"):
            ax.annotate(b["Judge"], (b["Recall"], b["Overall"]), textcoords="offset points", xytext=(5, -10), fontsize=7, color=col)
    ax.set_xlim(30, 100); ax.set_ylim(45, 95); ax.set_xlabel("recall of successful trajectories (%)"); ax.set_ylabel("precision (%)")
    ax.set_title("AgentRewardBench: did the web agent really finish the task?\nJEV's curve vs every judge on the leaderboard (dots)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "..", "assets", "agent_judge.png")); plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "summary":
        run()
    summary()
