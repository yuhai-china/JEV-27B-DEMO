"""JEV-27B demo app — one engine, two systems.

    bash common/serve_jev27b.sh      # terminal 1: vLLM server (System 1 + System 2)
    python app/app.py                # terminal 2: http://localhost:7860
"""
import datetime as dt, importlib.util, json, os, sys, time

import gradio as gr
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "common"))
from jev_client import decide, decide_many, expected_score  # noqa: E402


def load(folder, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, folder, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


search = load("01-search-ranking", "demo")
stock = load("02-stock-outlook", "demo")
agent = load("03-agent-decisions", "demo")
s12 = load("04-system1-to-system2", "demo")

INTRO = """# JEV-27B — one engine, two systems
**System 1** answers typed decisions (yes/no · pick one of 2-16 options · rate 0-5) in a single forward pass with calibrated
probabilities. **System 2** is the same 27B model thinking step by step. Both run in one vLLM engine."""


# ---------------------------------------------------------------- playground
def playground(kind, state, question, options):
    opts = [o.strip() for o in options.splitlines() if o.strip()] if kind == "choice" else None
    if kind == "choice" and not 2 <= len(opts or []) <= 16:
        raise gr.Error("choice needs 2-16 options, one per line")
    t = time.time()
    p = decide(kind, state, question, opts)
    ms = 1000 * (time.time() - t)
    extra = f" · expected rating **{expected_score(p):.2f} / 5**" if kind == "score" else ""
    return p, f"{ms:.0f} ms · one forward pass{extra}"


# ---------------------------------------------------------------- search
QUERIES = {q["query"]: q for q in search.DATA["queries"]}


def rerank(qtext):
    q = QUERIES[qtext]
    cands = [dict(c) for c in q["candidates"]]
    t = time.time()
    search.rerank(q["query"], cands)
    secs = time.time() - t
    bm = sorted(cands, key=lambda c: -c["bm25"])
    jv = sorted(cands, key=lambda c: -c["p_relevant"])
    nb = search.ndcg([c["label"] for c in bm], q["relevant_grades"])
    nj = search.ndcg([c["label"] for c in jv], q["relevant_grades"])
    df = pd.DataFrame([{"JEV #": i + 1, "BM25 #": bm.index(c) + 1, "P(rel.)": round(c["p_relevant"], 3),
                        "rating": round(c["rating"], 2), "label": c["label"], "document": c["text"][:220]} for i, c in enumerate(jv)])
    return df, f"**{2 * len(cands)} decisions in {secs:.1f} s** · nDCG@10 BM25 **{nb:.3f}** → JEV-27B **{nj:.3f}**"


# ---------------------------------------------------------------- stocks
def stocks(tickers):
    ts = [t.strip().upper() for t in tickers.replace(",", " ").split() if t.strip()][:12]
    snaps = [stock.snapshot(t) for t in ts]
    rows, n, secs = stock.analyse(snaps)
    df = pd.DataFrame([{"ticker": r["ticker"], "close": r["last_close"], "5d %": r["return_pct"]["5d"], "20d %": r["return_pct"]["20d"],
                        "P(up in 5 days)": round(r["p_up_5d"], 2), "stance": f"{r['stance']} ({r['stance_p']:.2f})",
                        "news tone 0-5": round(r["news_tone"], 1), "headlines on-topic": round(r["headlines_on_topic"], 2),
                        "latest headline": (r["latest_headlines"] or [{"title": ""}])[0]["title"]} for r in rows])
    return df, f"**{n} decisions in {secs:.1f} s** · prices as of {rows[0]['as_of']} · illustration only, not investment advice"


# ---------------------------------------------------------------- agent decisions
def triage(ticket):
    tm, pr, hu = decide_many([("choice", ticket, "Which team should handle this support ticket?", agent.TEAMS),
                              ("choice", ticket, "What priority should this support ticket get?", agent.PRIO),
                              ("noul", ticket, "Is this scenario one where: a human agent must respond personally (not an automated reply)?")])
    return tm, pr, {"needs a human": hu["true"], "automated reply is fine": hu["false"]}


def phishing(sender, subject, body, link):
    e = {"from": sender, "subject": subject, "body": body, "link": link or None}
    return decide("choice", e, "Is this email a phishing or fraud attempt, or a legitimate message?", ["phishing", "legitimate"])


def route(request, tools):
    ts = [t.strip() for t in tools.split(",") if t.strip()]
    return decide("choice", {"user_request": request, "available_tools": ts}, "Which tool should the agent call first?", ts)


# ---------------------------------------------------------------- System 1 -> System 2
def escalate(context, question, options, threshold):
    opts = [o.strip() for o in options.splitlines() if o.strip()]
    t = time.time()
    p = decide("choice", context, question, opts)
    s1_ms = 1000 * (time.time() - t)
    best = max(p, key=p.get)
    if p[best] >= threshold:
        return p, f"### System 1 answers: **{best}**\nconfidence {p[best]:.2f} ≥ {threshold:.2f} · {s1_ms:.0f} ms", ""
    s2 = s12.system2(question, opts, context)
    ans = opts[s2["pick"]] if s2["pick"] is not None else "(could not parse)"
    return p, (f"### System 1 unsure → System 2: **{ans}**\nSystem 1 top pick “{best}” at {p[best]:.2f} < {threshold:.2f} ({s1_ms:.0f} ms) · "
               f"System 2 thought for {s2['seconds']:.1f} s ({s2['tokens']} tokens)"), s2["final"]


with gr.Blocks(title="JEV-27B demo") as demo:
    gr.Markdown(INTRO)
    with gr.Tab("System 1 playground"):
        with gr.Row():
            with gr.Column():
                kind = gr.Radio(["choice", "noul", "score"], value="choice", label="kind (choice = pick one · noul = true/false · score = rate 0-5)")
                state = gr.Textbox(label="state (any text or JSON)", lines=4,
                                   value="Customer message: 'I love the product, but the invoice was charged twice and nobody answers my emails.'")
                question = gr.Textbox(label="question", value="What is the customer's overall sentiment?")
                options = gr.Textbox(label="options (choice only, one per line)", lines=4, value="positive\nmixed\nnegative")
                go = gr.Button("Decide", variant="primary")
            with gr.Column():
                probs = gr.Label(label="probabilities", num_top_classes=16)
                info = gr.Markdown()
        go.click(playground, [kind, state, question, options], [probs, info])
    with gr.Tab("Search re-ranking"):
        q = gr.Dropdown(list(QUERIES), value=list(QUERIES)[0], label="TREC-COVID query (BM25 top-20 candidates, human relevance labels)")
        b = gr.Button("Re-rank with JEV-27B", variant="primary")
        m1 = gr.Markdown()
        gr.Markdown("*JEV # / BM25 # = rank in each ordering · P(rel.) = System 1 probability of relevance · rating = expected 0-5 relevance · "
                    "label = human judgement (0 not relevant, 1 partially, 2 relevant)*")
        t1 = gr.Dataframe(wrap=True, column_widths=["6%", "7%", "8%", "7%", "6%", "66%"])
        b.click(rerank, q, [t1, m1])
    with gr.Tab("Stock signals"):
        tk = gr.Textbox(value="NVDA AAPL MSFT AMZN META GOOGL TSLA AMD", label="tickers (Yahoo Finance symbols)")
        b = gr.Button("Analyse", variant="primary")
        m2 = gr.Markdown()
        t2 = gr.Dataframe(wrap=True)
        b.click(stocks, tk, [t2, m2])
    with gr.Tab("Agent decisions"):
        with gr.Row():
            with gr.Column():
                tick = gr.Textbox(label="support ticket", lines=3, value=agent.TICKETS[3])
                b = gr.Button("Triage", variant="primary")
            with gr.Column():
                team, prio, hum = gr.Label(label="team"), gr.Label(label="priority"), gr.Label(label="needs a human")
        b.click(triage, tick, [team, prio, hum])
        with gr.Row():
            with gr.Column():
                e = agent.EMAILS[2]
                snd, sub = gr.Textbox(e["from"], label="from"), gr.Textbox(e["subject"], label="subject")
                bod, lnk = gr.Textbox(e["body"], label="body", lines=3), gr.Textbox("", label="link (optional)")
                b = gr.Button("Phishing check", variant="primary")
            ph = gr.Label(label="verdict")
        b.click(phishing, [snd, sub, bod, lnk], ph)
        with gr.Row():
            with gr.Column():
                req = gr.Textbox(agent.ASKS[0], label="user request")
                tls = gr.Textbox(", ".join(agent.TOOLS), label="available tools (comma separated)")
                b = gr.Button("Route", variant="primary")
            rt = gr.Label(label="call first")
        b.click(route, [req, tls], rt)
    with gr.Tab("System 1 → System 2"):
        with gr.Row():
            with gr.Column():
                ctx = gr.Textbox("Today is Tuesday, 29 September 2026.", label="context")
                qq = gr.Textbox("Which weekday is 100 days from today?", label="question")
                oo = gr.Textbox("Monday\nTuesday\nWednesday\nThursday\nFriday\nSaturday\nSunday", label="options (one per line)", lines=7)
                th = gr.Slider(0.5, 0.99, value=0.9, step=0.01, label="escalate to System 2 when System 1 confidence is below")
                b = gr.Button("Answer", variant="primary")
            with gr.Column():
                p1 = gr.Label(label="System 1 probabilities", num_top_classes=16)
                verdict = gr.Markdown()
                s2txt = gr.Textbox(label="System 2 final answer text", lines=4)
        b.click(escalate, [ctx, qq, oo, th], [p1, verdict, s2txt])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=16).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
