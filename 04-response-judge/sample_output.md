## RewardBench, all 2,985 pairs (official scoring)

5,970 System 1 decisions in 185 s (32 decisions/s on one GPU)

| judge | Score | Chat | Chat Hard | Safety | Reasoning |
|---|---:|---:|---:|---:|---:|
| **JEV-27B System 1 (one forward pass per order)** | **89.9** | 94.4 | 81.2 | 93.1 | 90.9 |
| Gemini 1.5 Pro (0514) | 88.2 | 92.3 | 80.6 | 87.9 | 92.0 |
| GPT-4o (2024-08-06) | 86.7 | 96.1 | 76.1 | 88.1 | 86.6 |
| Claude 3.5 Sonnet (2024-06-20) | 84.2 | 96.4 | 74.0 | 81.6 | 84.7 |
| Llama 3.1 405B Instruct | 84.1 | 97.2 | 74.6 | 77.6 | 87.1 |
| GPT-4 Turbo (2024-04-09) | 84.0 | 95.3 | 75.4 | 87.6 | 82.7 |
| Claude 3 Opus | 80.1 | 94.7 | 60.3 | 86.6 | 78.7 |

## System 1 vs System 2 thinking, 400-pair sample (100 per section)

| section | System 1 | System 2 (thinking) | gated: System 2 only when System 1 < 0.80 |
|---|---:|---:|---:|
| Chat | 0.920 | 0.870 | 0.900 (22% sent) |
| Chat Hard | 0.830 | 0.860 | 0.850 (32% sent) |
| Safety | 0.890 | 0.910 | 0.920 (15% sent) |
| Reasoning | 0.960 | 0.970 | 0.990 (8% sent) |
| All | 0.900 | 0.902 | 0.915 (19% sent) |

System 1 verdict unchanged when A and B are swapped: 96%

Median time per judgement: System 1 1.26 s (both orders) · System 2 7.5 s (456 thinking tokens), 16 requests in parallel
