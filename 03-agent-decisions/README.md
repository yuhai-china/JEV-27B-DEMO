# 03 · Everyday agent decisions

**Idea.** Agents and back-office workflows make the same small decisions all day: which team, how urgent, is this a scam,
may this post stay up, which tool to call. Asking a chat model means generating text, parsing it and hoping the format
holds. JEV-27B System 1 returns a **probability for every option in one forward pass**. There is nothing to parse, and the
probability tells you when to hand over to a human or to [System 2](../04-system1-to-system2).

**24 decisions in 0.43 s** (sent together; vLLM batches them on one GPU):

![phishing](../assets/agent_phishing.png)

### 1. Support-ticket triage

| ticket | team (p) | priority (p) | needs a human |
|---|---|---|---:|
| I was charged twice for my annual plan this morning. Please refund the duplicate charge ASAP. | billing (1.00) | urgent (0.68) | 0.70 |
| The iOS app crashes every time I open the camera scanner since yesterday's update. iPhone 15, iOS 19.2. | mobile app (1.00) | urgent (0.58) | 0.56 |
| Can you add dark mode to the web dashboard? Not urgent, just a suggestion. | product feedback (0.98) | low (0.99) | 0.39 |
| Our whole team is locked out — SSO returns error 500 for everyone since 9:05. We have a board demo at 11. | platform / SSO (1.00) | urgent (0.91) | 0.86 |

### 2. Phishing / fraud check

| from | subject | P(phishing) |
|---|---|---:|
| `security@paypa1-support.com` | Unusual sign-in — verify within 24h | **1.00** |
| `no-reply@github.com` | [GitHub] A new SSH key was added to your account | 0.17 |
| `ceo.office@company-mail.net` | Urgent wire before 3pm | **0.95** |
| `billing@notion.so` | Your receipt from Notion Labs | 0.10 |

The look-alike domain (`paypa1`) and the classic CEO-wire pattern are caught. Genuine security notices and receipts stay low.

### 3. Content moderation

| post | action (p) |
|---|---|
| You're all idiots and I hope your office burns down. | remove: harassment/threat (1.00) |
| This update is terrible, the new layout makes everything slower. | allow (0.81) |
| DM me for cheap followers, 10k for $5, guaranteed!!! | remove: spam (1.00) |
| Great tutorial, the part about caching finally made it click for me. | allow (0.99) |

Harsh-but-legitimate criticism is allowed; the threat and the spam are removed.

### 4. Tool routing: which tool should the agent call first?

| user request | tool (p) |
|---|---|
| Where is my order #88213? It was supposed to arrive Monday. | `query_orders_db` (0.99) |
| Set up a 30-minute sync with Dana next Tuesday afternoon. | `ask_user_for_clarification` (0.57) |
| What's the latest news on the EU AI Act enforcement? | `search_web` (0.99) |
| Can you handle the thing we discussed? | `ask_user_for_clarification` (1.00) |

The calendar request is the interesting one: "Tuesday afternoon" does not give a time, and the model splits between asking (0.57)
and `create_calendar_event` (0.41). A confidence of 0.57 is exactly the signal an agent can use to confirm with the user
instead of guessing.

```bash
python demo.py
```

```python
from jev_client import decide
decide("choice", ticket_text, "Which team should handle this support ticket?", ["billing", "mobile app", "platform / SSO", "product feedback"])
# -> {'billing': 1.000, 'mobile app': 0.000, 'platform / SSO': 0.000, 'product feedback': 0.000}
decide("noul", ticket_text, "Is this scenario one where: a human agent must respond personally (not an automated reply)?")
# -> {'false': 0.295, 'true': 0.705}
```

These are hand-written examples to show the interface. For measured accuracy on thousands of public decision tasks, see the
model card of [autotrust/JEV-27B](https://huggingface.co/autotrust/JEV-27B).
