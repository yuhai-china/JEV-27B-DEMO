"""Demo 04 — everyday agent decisions with JEV-27B System 1.

Four things an AI agent or a back-office workflow decides all day, each returned as calibrated probabilities in one
forward pass (no text generation, no parsing):
  1. support-ticket triage (team, priority, needs a human)
  2. phishing / fraud check
  3. content moderation
  4. tool routing (which tool should the agent call first)

    python demo.py
"""
import json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from charts import JEV, BASE, WARN, style  # noqa: E402
from jev_client import decide_many  # noqa: E402

TICKETS = [
    "I was charged twice for my annual plan this morning. Please refund the duplicate charge ASAP.",
    "The iOS app crashes every time I open the camera scanner since yesterday's update. iPhone 15, iOS 19.2.",
    "Can you add dark mode to the web dashboard? Not urgent, just a suggestion.",
    "Our whole team is locked out — SSO returns error 500 for everyone since 9:05. We have a board demo at 11.",
]
TEAMS = ["billing", "mobile app", "platform / SSO", "product feedback"]
PRIO = ["low", "medium", "high", "urgent"]
EMAILS = [
    {"from": "security@paypa1-support.com", "subject": "Unusual sign-in — verify within 24h",
     "body": "We detected a login from a new device. Verify your account now or it will be suspended.", "link": "http://paypa1-support.com/verify"},
    {"from": "no-reply@github.com", "subject": "[GitHub] A new SSH key was added to your account",
     "body": "A new public key was added to your account. If you did not add it, remove it in settings.", "link": "https://github.com/settings/keys"},
    {"from": "ceo.office@company-mail.net", "subject": "Urgent wire before 3pm",
     "body": "I'm in a meeting, can't talk. Please wire $48,500 to the vendor below today and keep it confidential.", "link": None},
    {"from": "billing@notion.so", "subject": "Your receipt from Notion Labs",
     "body": "Thanks for your payment of $96.00 for Notion Plus (annual). View your invoice in Settings → Billing.", "link": "https://www.notion.so/settings/billing"},
]
POSTS = ["You're all idiots and I hope your office burns down.",
         "This update is terrible, the new layout makes everything slower.",
         "DM me for cheap followers, 10k for $5, guaranteed!!!",
         "Great tutorial, the part about caching finally made it click for me."]
ACTIONS = ["allow", "allow with warning", "remove: harassment/threat", "remove: spam"]
TOOLS = ["search_web", "query_orders_db", "send_email", "create_calendar_event", "ask_user_for_clarification"]
ASKS = ["Where is my order #88213? It was supposed to arrive Monday.",
        "Set up a 30-minute sync with Dana next Tuesday afternoon.",
        "What's the latest news on the EU AI Act enforcement?",
        "Can you handle the thing we discussed?"]


def build():
    reqs, tags = [], []
    for t in TICKETS:
        reqs += [("choice", t, "Which team should handle this support ticket?", TEAMS),
                 ("choice", t, "What priority should this support ticket get?", PRIO),
                 ("noul", t, "Is this scenario one where: a human agent must respond personally (not an automated reply)?")]
        tags += [("ticket", t, "team"), ("ticket", t, "priority"), ("ticket", t, "human")]
    for e in EMAILS:
        reqs.append(("choice", e, "Is this email a phishing or fraud attempt, or a legitimate message?", ["phishing", "legitimate"]))
        tags.append(("email", e, "phishing"))
    for p in POSTS:
        reqs.append(("choice", p, "What moderation action does this post need?", ACTIONS))
        tags.append(("post", p, "action"))
    for a in ASKS:
        reqs.append(("choice", {"user_request": a, "available_tools": TOOLS}, "Which tool should the agent call first?", TOOLS))
        tags.append(("ask", a, "tool"))
    return reqs, tags


def main():
    reqs, tags = build()
    t0 = time.time()
    out = decide_many(reqs)
    secs = time.time() - t0
    top = lambda p: max(p, key=p.get)
    md = [f"**{len(reqs)} decisions in {secs:.2f} s** (all sent at once; vLLM batches them)\n", "## 1. Support-ticket triage\n",
          "| ticket | team (p) | priority (p) | needs a human |", "|---|---|---|---:|"]
    for i, t in enumerate(TICKETS):
        tm, pr, hu = out[3 * i: 3 * i + 3]
        md.append(f"| {t} | {top(tm)} ({tm[top(tm)]:.2f}) | {top(pr)} ({pr[top(pr)]:.2f}) | {hu['true']:.2f} |")
    k = 3 * len(TICKETS)
    md += ["\n## 2. Phishing / fraud check\n", "| from | subject | P(phishing) |", "|---|---|---:|"]
    for i, e in enumerate(EMAILS):
        md.append(f"| `{e['from']}` | {e['subject']} | **{out[k + i]['phishing']:.2f}** |")
    k += len(EMAILS)
    md += ["\n## 3. Content moderation\n", "| post | action (p) |", "|---|---|"]
    for i, p in enumerate(POSTS):
        o = out[k + i]
        md.append(f"| {p} | {top(o)} ({o[top(o)]:.2f}) |")
    k += len(POSTS)
    md += ["\n## 4. Tool routing — which tool should the agent call first?\n", "| user request | tool (p) |", "|---|---|"]
    for i, a in enumerate(ASKS):
        o = out[k + i]
        md.append(f"| {a} | `{top(o)}` ({o[top(o)]:.2f}) |")
    open(os.path.join(HERE, "sample_output.md"), "w").write("\n".join(md) + "\n")
    json.dump({"seconds": secs, "decisions": [{"type": t[0], "input": t[1], "question": t[2], "probs": o} for t, o in zip(tags, out)]},
              open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False)
    print("\n".join(md))

    plt = style()
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ph = [out[3 * len(TICKETS) + i]["phishing"] for i in range(len(EMAILS))]
    labels = [f"{e['from']}\n“{e['subject'][:38]}”" for e in EMAILS]
    ax.barh(labels[::-1], ph[::-1], color=[WARN if p > 0.5 else JEV for p in ph[::-1]])
    ax.axvline(0.5, color=BASE, ls="--", lw=1); ax.set_xlim(0, 1); ax.set_xlabel("P(phishing or fraud)")
    for i, p in enumerate(ph[::-1]):
        ax.text(min(p + 0.02, 0.9), i, f"{p:.2f}", va="center", fontsize=9)
    ax.tick_params(axis="y", labelsize=8); ax.set_title("Phishing check — one forward pass per email")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "..", "assets", "agent_phishing.png")); plt.close(fig)


if __name__ == "__main__":
    main()
