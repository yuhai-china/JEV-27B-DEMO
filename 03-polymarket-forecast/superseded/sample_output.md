## A. Without the market price

100 resolved markets · 61% resolved Yes · forecast 7 days before resolution

| forecaster | Brier ↓ | log loss ↓ | direction ✓ |
|---|---:|---:|---:|
| always 0.5 | 0.250 | 0.693 | 39% |
| base rate of this sample (0.61, hindsight) | 0.238 | 0.669 | 61% |
| JEV System 1 — question only | 0.357 | 0.979 | 46% |
| JEV System 1 + headlines | 0.243 | 0.775 | 67% |
| JEV System 2 (thinking) — stated probability | 0.456 | 3.924 | 54% |
| **JEV System 1 + 2** (headlines + System 2 analysis → System 1) | 0.242 | 0.751 | 66% |

On the 99 markets with a price history: market price 7 days before resolution Brier 0.124; JEV System 1 + 2 Brier 0.239

## B. With the market price in the input (JEV as a market corrector)

99 resolved markets · 61% resolved Yes · forecast 7 days before resolution · market price in the input

| forecaster | Brier ↓ | log loss ↓ | direction ✓ | moved toward the outcome (of moves > 0.02) |
|---|---:|---:|---:|---:|
| market price 7 days before resolution | 0.124 | 0.391 | 87% | — |
| JEV System 1 + market price | 0.101 | 0.349 | 87% | 45/75 |
| JEV System 1 + market price + headlines | 0.105 | 0.350 | 85% | 43/75 |
| JEV System 2 (thinking) + market price + headlines — stated probability | 0.337 | 2.716 | 66% | 17/45 |
| **JEV System 1 + 2** + market price + headlines | 0.129 | 0.450 | 85% | 49/79 |

Bootstrap (10,000 resamples of the 99 markets): Brier improvement of *System 1 + market price* over the market = +0.0239, 95% interval [-0.0025, +0.0483], share of resamples > 0: 0.96
