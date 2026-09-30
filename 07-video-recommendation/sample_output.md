200 users · 20 candidates each (1 hidden last video + 19 videos others watched within ±3 days)

### Zero-shot (no interaction data used)

| method | AUC | HR@1 | HR@5 | NDCG@10 | MRR |
|---|---:|---:|---:|---:|---:|
| random order | 0.489 | 0.040 | 0.205 | 0.219 | 0.168 |
| popularity so far | 0.505 | 0.040 | 0.230 | 0.204 | 0.165 |
| title similarity (TF-IDF) | 0.602 | 0.205 | 0.385 | 0.367 | 0.328 |
| JEV-27B, titles only | 0.649 | 0.205 | 0.455 | 0.399 | 0.344 |
| **JEV-27B, covers only** | **0.727** | **0.280** | **0.590** | **0.498** | **0.428** |
| JEV-27B, covers + titles | 0.706 | 0.245 | 0.565 | 0.469 | 0.405 |

### Reference: learned from other users' interaction logs

| method | AUC | HR@1 | HR@5 | NDCG@10 | MRR |
|---|---:|---:|---:|---:|---:|
| item-based collaborative filtering | 0.728 | 0.355 | 0.490 | 0.503 | 0.460 |

Paired bootstrap of the AUC difference (95% interval):
- JEV-27B, covers only − JEV-27B, titles only: +0.078 [+0.031, +0.126]
- JEV-27B, covers only − title similarity (TF-IDF): +0.126 [+0.075, +0.175]
- JEV-27B, covers only − item-based collaborative filtering: -0.000 [-0.041, +0.040]
