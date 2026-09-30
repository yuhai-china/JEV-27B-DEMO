500 impressions from 15 Nov 2019 · 9,346 candidate articles · 6.8% clicked

### 1. Zero-shot: no training on MIND

| method | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| random order | 0.515 | 0.278 | 0.284 | 0.367 |
| popularity so far | 0.546 | 0.315 | 0.340 | 0.404 |
| category match | 0.606 | 0.346 | 0.384 | 0.447 |
| title similarity (TF-IDF) | 0.546 | 0.315 | 0.331 | 0.407 |
| **JEV-27B System 1, zero-shot** | **0.642** | **0.388** | **0.424** | **0.486** |

AUC gain of JEV over the best zero-shot baseline: +0.036 (95% bootstrap interval +0.008 to +0.064)

### 2. Trained on MIND (1,000 training impressions, same data for both)

| method | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| LightGBM ranker | 0.590 | 0.339 | 0.374 | 0.434 |
| **LightGBM ranker + JEV zero-shot score** | **0.617** | **0.369** | **0.393** | **0.459** |
| LightGBM ranker, 4x more training data (4,000 impressions) | 0.616 | 0.361 | 0.394 | 0.462 |

AUC gain from adding the JEV feature: +0.026 (95% bootstrap interval +0.000 to +0.053)
