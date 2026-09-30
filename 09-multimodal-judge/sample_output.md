## VL-RewardBench

1,247 pairs · 2,494 System 1 decisions in 163 s · overall accuracy 95% interval [75.9, 80.5] · verdict unchanged when A and B are swapped: 89%

| judge | general | hallucination | reasoning | **overall** | macro average |
|---|---:|---:|---:|---:|---:|
| **JEV-27B-VL System 1** | 58.0 | 83.2 | 78.2 | **78.3** | **73.1** |
| Skywork-VL-Reward-7B | 65.6 | 80.2 | 61.3 | 73.3 | 69.0 |
| Gemini 2.0 Flash | 50.8 | 72.6 | 70.1 | 68.8 | 64.5 |
| Gemini 1.5 Pro | 50.8 | 72.5 | 64.2 | 67.2 | 62.5 |
| GPT-4o | 49.1 | 67.6 | 70.5 | 65.8 | 62.4 |
| Gemini 1.5 Flash | 47.8 | 59.6 | 58.4 | 57.6 | 55.3 |
| Llama-3.2-90B-Vision | 42.6 | 57.3 | 61.7 | 56.2 | 53.9 |
| Claude 3.5 Sonnet | 43.4 | 55.0 | 62.3 | 55.3 | 53.6 |
| Qwen2-VL-72B | 38.1 | 32.8 | 58.0 | 39.5 | 43.0 |

## Multimodal RewardBench 2 (Meta, Dec 2025): 4 of 4 tasks, 1,000 pairs each

| judge | text-to-image | image editing | interleaved | reasoning | average |
|---|---:|---:|---:|---:|---:|
| Gemini 3 Pro | 74.4 | 74.9 | 76.4 | 79.5 | 76.3 |
| GPT-5 | 70.5 | 73.8 | 74.4 | 70.2 | 72.2 |
| Gemini 2.5 Pro | 70.5 | 71.3 | 75.1 | 66.6 | 70.9 |
| Qwen3-VL-32B | 64.1 | 67.3 | 70.5 | 56.6 | 64.6 |
| Gemini 2.5 Flash | 63.1 | 66.5 | 69.4 | 57.5 | 64.1 |
| **JEV-27B-VL System 1** | **69.2** | **57.8** | **67.8** | **60.4** | **63.8** |
| GPT-4.1 | 65.8 | 68.2 | 67.0 | 53.0 | 63.5 |
| Qwen3-VL-235B-A22B | 62.0 | 64.8 | 69.0 | 55.9 | 62.9 |
| GPT-4o | 60.3 | 65.0 | 61.5 | 51.9 | 59.7 |
| Gemma 3 27B | 58.3 | 60.2 | 61.1 | 49.4 | 57.2 |
