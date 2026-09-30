"""Demo 05 — hallucination guard: System 2 answers, System 1 decides whether the answer can be trusted.

System 2 answers a factual question directly (no thinking, a few words). System 1 then reads (question, answer) and returns
P(answer is correct) in one forward pass. Compared with asking System 2 how confident it is.
Data: 1,000 random TriviaQA questions (validation split, rc.nocontext; Apache-2.0).

    python demo.py            # run (~2-3 min) and summarise
    python demo.py summary    # tables + charts from results.json
"""
import concurrent.futures as cf, json, os, random, re, string, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from charts import JEV, BASE, ALT, style  # noqa: E402
from jev_client import chat_raw, decide, reply_parts  # noqa: E402

N = 1000
CHECK_Q = "Is this scenario one where: the proposed answer is factually correct?"


def answer(question):
    """System 2, no thinking: a short direct answer."""
    r = chat_raw(f"Answer this trivia question with the answer only (a few words, no explanation).\n\n{question}", thinking=False, max_tokens=40)
    return reply_parts(r)[1].strip().split("\n")[0][:200]


def trust(question, ans):
    """System 1: P(the answer is correct)."""
    return decide("noul", {"question": question, "proposed_answer": ans}, CHECK_Q)["true"]


def stated_confidence(question, ans):
    """Baseline: ask System 2 how confident it is."""
    txt = reply_parts(chat_raw(f"Question: {question}\nProposed answer: {ans}\n\nHow likely is it that the proposed answer is correct? "
                               "Reply with a single probability between 0 and 1, nothing else.", thinking=False, max_tokens=10))[1]
    m = re.findall(r"[01](?:\.\d+)?", txt)
    return float(m[0]) if m else None


def norm(s):
    s = "".join(ch if ch not in set(string.punctuation) else " " for ch in s.lower())
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in s.split())


def is_correct(ans, aliases):
    """Lenient match against the TriviaQA aliases: exact / contained either way / token overlap (F1 >= 0.6)."""
    a = norm(ans)
    at = set(a.split())
    for al in map(norm, aliases):
        if not al or not a:
            continue
        if al == a or re.search(r"\b" + re.escape(al) + r"\b", a) or re.search(r"\b" + re.escape(a) + r"\b", al):
            return True
        bt = set(al.split())
        if at & bt and 2 * len(at & bt) / (len(at) + len(bt)) >= 0.6:
            return True
    return False


def run():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    tq = pd.read_parquet(hf_hub_download("mandarjoshi/trivia_qa", "rc.nocontext/validation-00000-of-00001.parquet", repo_type="dataset"))
    idx = random.Random(0).sample(range(len(tq)), N)

    def one(i):
        r = tq.iloc[i]
        ans = answer(r.question)
        t = time.time()
        p1 = trust(r.question, ans)
        dt = time.time() - t
        aliases = list(r.answer["aliases"]) + list(r.answer["normalized_aliases"]) + [r.answer["value"]]
        return {"question": r.question, "answer": ans, "gold": r.answer["value"], "correct": is_correct(ans, aliases),
                "p_system1": p1, "p_system2_stated": stated_confidence(r.question, ans), "system1_seconds": dt}

    t0 = time.time()
    with cf.ThreadPoolExecutor(24) as ex:
        res = list(ex.map(one, idx))
    print(f"{N} questions in {time.time() - t0:.0f}s")
    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False)


def auroc(scores, labels):
    """P(a random correct answer gets a higher score than a random wrong one); ties count half."""
    import bisect
    pos = sorted(s for s, l in zip(scores, labels) if l)
    neg = [s for s, l in zip(scores, labels) if not l]
    total = 0.0
    for v in neg:
        lo, hi = bisect.bisect_left(pos, v), bisect.bisect_right(pos, v)
        total += (len(pos) - hi) + 0.5 * (hi - lo)
    return total / (len(pos) * len(neg))


def summary():
    res = json.load(open(os.path.join(HERE, "results.json")))
    y = [r["correct"] for r in res]
    s1 = [r["p_system1"] for r in res]
    s2 = [r["p_system2_stated"] if r["p_system2_stated"] is not None else 0.5 for r in res]
    n = len(res)
    covs = [0.3, 0.5, 0.7, 0.9, 1.0]

    def sel(sc, cov):
        order = sorted(range(n), key=lambda i: -sc[i])
        k = int(round(cov * n))
        return sum(y[i] for i in order[:k]) / k

    md = [f"{n} TriviaQA questions · System 2 answers directly · accuracy of its answers: **{sum(y) / n:.1%}**\n",
          "| | System 1 check (one forward pass) | System 2 asked for its confidence |", "|---|---:|---:|",
          f"| separates right from wrong answers (AUROC, 1.0 = perfect) | **{auroc(s1, y):.3f}** | {auroc(s2, y):.3f} |",
          f"| average confidence (actual accuracy {sum(y) / n:.1%}) | {sum(s1) / n:.1%} | {sum(s2) / n:.1%} |",
          f"| distinct confidence values used | {len(set(round(x, 3) for x in s1))} | {len(set(s2))} |"]
    md += ["\nAnswer only the questions the check trusts most, and say \"I'm not sure\" otherwise:\n",
           "| questions answered | accuracy with System 1 check | accuracy with System 2's own confidence |", "|---:|---:|---:|"]
    for c in covs:
        b = "**" if c < 1 else ""
        md.append(f"| {c:.0%} | {b}{sel(s1, c):.1%}{b} | {sel(s2, c):.1%} |")
    md += ["\nHow far System 1's number can be trusted:\n", "| System 1 says | answers | actually correct |", "|---|---:|---:|"]
    bins = []
    for lo, hi in ((0.0, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 0.97), (0.97, 1.01)):
        g = [i for i in range(n) if lo <= s1[i] < hi]
        if g:
            bins.append((lo, hi, len(g), sum(y[i] for i in g) / len(g), sum(s1[i] for i in g) / len(g)))
            md.append(f"| {lo:.2f} – {min(hi, 1):.2f} | {len(g)} | {bins[-1][3]:.1%} |")
    med = sorted(r["system1_seconds"] for r in res)[n // 2]
    md.append(f"\nMedian time of the System 1 check: {1000 * med:.0f} ms (24 questions in flight)")
    open(os.path.join(HERE, "sample_output.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    plt = style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2))
    cs = [i / 20 for i in range(4, 21)]
    a1.plot([100 * c for c in cs], [sel(s1, c) for c in cs], "-o", color=JEV, ms=3, label="System 1 check")
    a1.plot([100 * c for c in cs], [sel(s2, c) for c in cs], "-o", color=ALT, ms=3, label="System 2's own confidence")
    a1.axhline(sum(y) / n, color=BASE, ls="--", lw=1)
    a1.text(21, sum(y) / n + 0.006, f"answer everything: {sum(y) / n:.0%}", color="#495057", fontsize=8)
    a1.set_xlabel("questions answered (%), the rest get \"I'm not sure\""); a1.set_ylabel("accuracy of the answers given")
    a1.set_title("Answer only when the check trusts the answer"); a1.legend(fontsize=8, loc="upper right")
    a2.bar([f"{lo:.2f}–{min(hi, 1):.2f}\n(n={k})" for lo, hi, k, _, _ in bins], [a for *_, a, _ in bins], color=JEV)
    for i, (*_, a, _) in enumerate(bins):
        a2.text(i, a + 0.02, f"{a:.0%}", ha="center", fontsize=9)
    a2.set_ylim(0, 1.12); a2.set_xlabel("System 1 confidence that the answer is correct"); a2.set_ylabel("actually correct")
    a2.set_title("System 1's number means what it says"); a2.tick_params(axis="x", labelsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "..", "assets", "hallucination_guard.png")); plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "summary":
        run()
    summary()
