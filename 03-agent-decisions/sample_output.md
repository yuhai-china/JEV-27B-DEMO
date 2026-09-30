**24 decisions in 0.43 s** (all sent at once; vLLM batches them)

## 1. Support-ticket triage

| ticket | team (p) | priority (p) | needs a human |
|---|---|---|---:|
| I was charged twice for my annual plan this morning. Please refund the duplicate charge ASAP. | billing (1.00) | urgent (0.68) | 0.70 |
| The iOS app crashes every time I open the camera scanner since yesterday's update. iPhone 15, iOS 19.2. | mobile app (1.00) | urgent (0.58) | 0.56 |
| Can you add dark mode to the web dashboard? Not urgent, just a suggestion. | product feedback (0.98) | low (0.99) | 0.39 |
| Our whole team is locked out — SSO returns error 500 for everyone since 9:05. We have a board demo at 11. | platform / SSO (1.00) | urgent (0.91) | 0.86 |

## 2. Phishing / fraud check

| from | subject | P(phishing) |
|---|---|---:|
| `security@paypa1-support.com` | Unusual sign-in — verify within 24h | **1.00** |
| `no-reply@github.com` | [GitHub] A new SSH key was added to your account | **0.17** |
| `ceo.office@company-mail.net` | Urgent wire before 3pm | **0.95** |
| `billing@notion.so` | Your receipt from Notion Labs | **0.10** |

## 3. Content moderation

| post | action (p) |
|---|---|
| You're all idiots and I hope your office burns down. | remove: harassment/threat (1.00) |
| This update is terrible, the new layout makes everything slower. | allow (0.81) |
| DM me for cheap followers, 10k for $5, guaranteed!!! | remove: spam (1.00) |
| Great tutorial, the part about caching finally made it click for me. | allow (0.99) |

## 4. Tool routing — which tool should the agent call first?

| user request | tool (p) |
|---|---|
| Where is my order #88213? It was supposed to arrive Monday. | `query_orders_db` (0.99) |
| Set up a 30-minute sync with Dana next Tuesday afternoon. | `ask_user_for_clarification` (0.57) |
| What's the latest news on the EU AI Act enforcement? | `search_web` (0.99) |
| Can you handle the thing we discussed? | `ask_user_for_clarification` (1.00) |
