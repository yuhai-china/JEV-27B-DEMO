"""Demo 04 — System 1 → System 2: answer fast when confident, think only when needed (same engine, same weights).

JEV-27B serves two "systems" from one vLLM engine:
  System 1  the decision head: one forward pass, calibrated probabilities (~0.1 s)
  System 2  the unmodified Qwen3.8-27B with step-by-step thinking (seconds to minutes)
Because System 1's confidence is calibrated, it can decide *when* to escalate: if its top option is below a threshold,
the question goes to System 2.

Measured on 120 multiple-choice questions with known answers (30 each from GSM8K, AQuA-RAT, ARC-Challenge,
CommonsenseQA; data/questions.json):

    python demo.py            # run everything (System 2 on all 120 questions takes ~10 minutes), then summarise
    python demo.py summary    # re-print tables / charts from results.json
    python demo.py examples   # three hand-written escalation examples
"""
import concurrent.futures as cf, json, os, re, sys, time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from charts import JEV, BASE, ALT, WARN, style  # noqa: E402
import jev_client  # noqa: E402
from jev_client import decide, split_thinking  # noqa: E402

QS = json.load(open(os.path.join(HERE, "data", "questions.json")))
OUT = os.path.join(HERE, "results.json")
L = "ABCDEFGHIJKLMNOP"
THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]


def s2_prompt(q, opts, context=""):
    body = (context + "\n\n" if context else "") + q + "\n\n" + "\n".join(f"{L[i]}) {o}" for i, o in enumerate(opts))
    return body + "\n\nThink it through step by step. On the last line write exactly: Answer: <letter>"


def system2(q, opts, context=""):
    t = time.time()
    r = requests.post(f"{jev_client.URL}/v1/chat/completions", json={
        "model": jev_client.REPO, "messages": [{"role": "user", "content": s2_prompt(q, opts, context)}], "max_tokens": 8000,
        "temperature": 0.6, "top_p": 0.95, "chat_template_kwargs": {"enable_thinking": True}}, timeout=1800).json()
    reasoning, final = split_thinking(r["choices"][0]["message"]["content"])
    m = re.findall(r"Answer:\s*\**\(?([A-P])\b", final) or re.findall(r"\b([A-P])\)?\s*$", final.strip())
    pick = L.index(m[-1]) if m and L.index(m[-1]) < len(opts) else None
    return {"pick": pick, "seconds": time.time() - t, "tokens": r["usage"]["completion_tokens"], "final": final[-300:]}


def run():
    res = []
    for q in QS:  # System 1, one request at a time → real single-request latency
        t = time.time()
        p = decide("choice", {}, q["question"], q["options"])
        pr = list(p.values())
        res.append({**q, "s1_probs": pr, "s1_pick": max(range(len(pr)), key=lambda i: pr[i]), "s1_seconds": time.time() - t})
    print(f"System 1 done: accuracy {sum(r['s1_pick'] == r['answer'] for r in res) / len(res):.3f}")
    t0 = time.time()
    with cf.ThreadPoolExecutor(12) as ex:
        for r, s2 in zip(res, ex.map(lambda q: system2(q["question"], q["options"]), res)):
            r["s2"] = s2
    print(f"System 2 done in {time.time() - t0:.0f} s")
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)


def summary():
    res = json.load(open(OUT))
    n = len(res)
    s1c = lambda r: r["s1_pick"] == r["answer"]
    s2c = lambda r: r["s2"]["pick"] == r["answer"]
    conf = lambda r: max(r["s1_probs"])
    rows = []
    for t in THRESHOLDS:
        esc = [conf(r) < t for r in res]
        acc = sum(s2c(r) if e else s1c(r) for r, e in zip(res, esc)) / n
        per_q = sorted(r["s1_seconds"] + (r["s2"]["seconds"] if e else 0) for r, e in zip(res, esc))
        tok = sum(r["s2"]["tokens"] if e else 0 for r, e in zip(res, esc)) / n
        rows.append({"threshold": t, "escalated": sum(esc) / n, "accuracy": acc, "mean_seconds": sum(per_q) / n,
                     "median_seconds": per_q[n // 2], "p90_seconds": per_q[int(0.9 * n)], "mean_thinking_tokens": tok})
    s1_acc, s2_acc = rows[0]["accuracy"], rows[-1]["accuracy"]
    md = [f"{n} questions · System 1 median latency {sorted(r['s1_seconds'] for r in res)[n // 2] * 1000:.0f} ms · "
          f"System 2 median {sorted(r['s2']['seconds'] for r in res)[n // 2]:.1f} s (12 requests in parallel), "
          f"median {sorted(r['s2']['tokens'] for r in res)[n // 2]} thinking tokens\n",
          "| policy | answered by System 2 | accuracy | median wait | 90th-percentile wait | mean wait | generated tokens per question |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        name = ("System 1 only" if r["threshold"] == 0 else "System 2 only" if r["threshold"] > 1 else f"escalate when System 1 confidence < {r['threshold']:.2f}")
        bold = "**" if r["threshold"] == 0.7 else ""
        md.append(f"| {bold}{name}{bold} | {r['escalated']:.0%} | {bold}{r['accuracy']:.3f}{bold} | {r['median_seconds']:.2f} s | {r['p90_seconds']:.1f} s | "
                  f"{r['mean_seconds']:.1f} s | {r['mean_thinking_tokens']:.0f} |")
    md += ["\nPer source (threshold 0.70):\n", "| source | System 1 | System 2 | gated | sent to System 2 |", "|---|---:|---:|---:|---:|"]
    per = {}
    for s in dict.fromkeys(r["source"] for r in res):
        g = [r for r in res if r["source"] == s]
        e = [conf(r) < 0.7 for r in g]
        per[s] = {"s1": sum(map(s1c, g)) / len(g), "s2": sum(map(s2c, g)) / len(g),
                  "gated": sum(s2c(r) if x else s1c(r) for r, x in zip(g, e)) / len(g), "esc": sum(e) / len(g)}
        md.append(f"| {s} | {per[s]['s1']:.2f} | {per[s]['s2']:.2f} | {per[s]['gated']:.2f} | {per[s]['esc']:.0%} |")
    md += ["\nIs System 1's confidence trustworthy? (why gating works)\n", "| System 1 confidence | questions | System 1 accuracy |", "|---|---:|---:|"]
    for lo, hi in ((0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.99), (0.99, 1.01)):
        g = [r for r in res if lo <= conf(r) < hi]
        if g:
            md.append(f"| {lo:.2f} – {min(hi, 1):.2f} | {len(g)} | {sum(map(s1c, g)) / len(g):.2f} |")
    md.append(f"\nParse failures in System 2 answers: {sum(r['s2']['pick'] is None for r in res)}")
    open(os.path.join(HERE, "sample_output.md"), "w").write("\n".join(md) + "\n")
    json.dump({"policies": rows, "per_source": per}, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
    print("\n".join(md))

    plt = style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    xs, ys = [100 * r["escalated"] for r in rows], [r["accuracy"] for r in rows]
    a1.plot(xs, ys, "-o", color=JEV)
    for r, x, y in zip(rows, xs, ys):
        lab = ("System 1 only" if r["threshold"] == 0 else "System 2 only" if r["threshold"] > 1 else f"escalate below {r['threshold']:.2f}")
        if r["threshold"] in (0.0, 0.7, 1.01):
            a1.annotate(f"{lab}\nmedian wait {r['median_seconds']:.1f} s", (x, y), textcoords="offset points",
                        xytext=(-60, 10) if r["threshold"] > 1 else (8, -24), fontsize=8)
    a1.set_xlabel("questions sent to System 2 (%)"); a1.set_ylabel("accuracy"); a1.set_xlim(-5, 110)
    a1.set_ylim(min(ys) - 0.04, max(ys) + 0.04)
    a1.set_title("Escalate only when System 1 is unsure")
    names = list(per)
    w = 0.26
    for k, (key, lab, col) in enumerate((("s1", "System 1 only", BASE), ("gated", "gated @ 0.70", JEV), ("s2", "System 2 only", ALT))):
        a2.bar([i + (k - 1) * w for i in range(len(names))], [per[s][key] for s in names], w, label=lab, color=col)
    for i, s in enumerate(names):
        a2.text(i, 1.03, f"{per[s]['esc']:.0%} sent", ha="center", fontsize=8, color=WARN)
    a2.set_xticks(range(len(names))); a2.set_xticklabels(names, fontsize=9); a2.set_ylim(0, 1.25)
    a2.legend(loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    a2.set_title("Per source")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "..", "assets", "system1_to_system2.png")); plt.close(fig)


def examples():
    cases = [("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball.", "How much does the ball cost?", ["$0.10", "$0.05", "$1.00", "$0.55"]),
             ("Refund policy: full refund within 30 days of delivery; store credit within 60 days; nothing after 60 days. Order delivered 45 days ago.",
              "What should the customer get?", ["full refund", "store credit", "nothing"]),
             ("Today is Tuesday, 29 September 2026.", "Which weekday is 100 days from today?",
              ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])]
    lines = []
    for st, q, opts in cases:
        t = time.time()
        p = decide("choice", st, q, opts)
        b = max(p, key=p.get)
        if p[b] >= 0.90:
            lines.append(f"- **System 1** ({p[b]:.2f} ≥ 0.90, {1000 * (time.time() - t):.0f} ms): {q} → **{b}**")
        else:
            s2 = system2(q, opts, st)
            ans = opts[s2["pick"]] if s2["pick"] is not None else "?"
            lines.append(f"- **System 1 unsure** (top “{b}” at {p[b]:.2f} < 0.90) → **System 2** ({s2['seconds']:.1f} s, {s2['tokens']} tokens): {q} → **{ans}**")
    print("\n".join(lines))
    open(os.path.join(HERE, "examples_output.md"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "examples":
        examples()
    else:
        if mode == "all":
            run()
        summary()
