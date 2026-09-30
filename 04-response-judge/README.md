# 04 · Response judge: grade answers in one forward pass

**Idea.** Evaluating AI answers (for model comparisons, RLHF reward, quality monitoring) usually means asking a large model
to write a critique and parsing its verdict. JEV-27B System 1 reads *(user request, response A, response B)* and returns
**P(A is better)** in one forward pass. It asks once per order (A/B and B/A) and averages, which removes position bias.

![judge](../assets/judge_rewardbench.png)

## RewardBench: 89.9, ahead of GPT-4o, Gemini 1.5 Pro and Claude 3.5 Sonnet as judges

[RewardBench](https://huggingface.co/datasets/allenai/reward-bench) (filtered): 2,985 human-verified pairs of a better and a worse
response, official section weighting. Comparison scores are from the public
[RewardBench leaderboard](https://huggingface.co/spaces/allenai/reward-bench).

| judge | Score | Chat | Chat Hard | Safety | Reasoning |
|---|---:|---:|---:|---:|---:|
| **JEV-27B System 1 (one forward pass per order)** | **89.9** | 94.4 | 81.2 | 93.1 | 90.9 |
| Gemini 1.5 Pro (0514) | 88.2 | 92.3 | 80.6 | 87.9 | 92.0 |
| GPT-4o (2024-08-06) | 86.7 | 96.1 | 76.1 | 88.1 | 86.6 |
| Claude 3.5 Sonnet (2024-06-20) | 84.2 | 96.4 | 74.0 | 81.6 | 84.7 |
| Llama 3.1 405B Instruct | 84.1 | 97.2 | 74.6 | 77.6 | 87.1 |
| GPT-4 Turbo (2024-04-09) | 84.0 | 95.3 | 75.4 | 87.6 | 82.7 |
| Claude 3 Opus | 80.1 | 94.7 | 60.3 | 86.6 | 78.7 |

All 5,970 decisions took **185 seconds on one GPU** (32 judgements per second).

## As accurate as thinking, about 6× faster

On a 400-pair sample (100 per section), the same engine judged each pair two ways: System 1 in one pass, and System 2
thinking it through before naming a winner.

| section | System 1 | System 2 (thinking) | **gated**: System 2 only when System 1 < 0.80 |
|---|---:|---:|---:|
| Chat | 0.920 | 0.870 | 0.900 (22% sent) |
| Chat Hard | 0.830 | 0.860 | 0.850 (32% sent) |
| Safety | 0.890 | 0.910 | 0.920 (15% sent) |
| Reasoning | 0.960 | 0.970 | 0.990 (8% sent) |
| **All** | **0.900** | **0.902** | **0.915** (19% sent) |

Median time per judgement: System 1 **1.3 s** for both orders, System 2 **7.5 s** (16 requests in parallel).

* System 1 alone matches the thinking judge.
* Sending only the 19% of pairs where System 1 is unsure to System 2 beats both: **0.915**.
* Swapping A and B changes System 1's verdict in only 4% of pairs.

```bash
python demo.py full      # all 2,985 pairs, official score (~3 min)
python demo.py gated     # 400-pair System 1 vs System 2 vs gated comparison (~6 min)
python demo.py summary   # tables and chart from the cached results
```

```python
from demo import judge
p_a_better, _, _ = judge(user_request, response_a, response_b)   # both orders averaged
```
