250 matches · home 108 / draw 62 / away 80

| forecaster | RPS ↓ | Brier ↓ | log loss ↓ | top pick correct |
|---|---:|---:|---:|---:|
| uniform 1/3 | 0.2364 | 0.6667 | 1.0986 | 43.2% |
| league base rates (last season) | 0.2319 | 0.6501 | 1.0737 | 43.2% |
| JEV System 1 — stats only | 0.2646 | 0.7403 | 1.4293 | 45.6% |
| JEV System 1 + headlines | 0.2538 | 0.7087 | 1.3670 | 49.6% |
| JEV System 2 (thinking) — stated probabilities | 0.2065 | 0.5965 | 1.0015 | 47.2% |
| JEV System 1 + 2 | 0.2963 | 0.7939 | 1.9734 | 48.8% |
| bookmaker (Pinnacle, margin removed) | 0.2008 | 0.5826 | 0.9793 | 51.6% |
| JEV System 1 + 2 + bookmaker probabilities | 0.2802 | 0.7541 | 1.8656 | 52.0% |

Confidence check and a one-number fix (temperature scaling, fitted on half of the matches and scored on the other half):

| forecaster | avg. probability of its top pick | top pick correct | RPS after calibration | log loss after calibration |
|---|---:|---:|---:|---:|
| System 1 — stats only | 0.75 | 0.46 | 0.2094 | 1.0157 |
| System 1 + headlines | 0.75 | 0.50 | 0.2042 | 0.9954 |
| System 1 + 2 | 0.83 | 0.49 | 0.2113 | 1.0293 |
| System 1 + 2 + bookmaker | 0.83 | 0.52 | 0.2059 | 1.0052 |
| System 2 | 0.46 | 0.47 | 0.2045 | 0.9931 |
| bookmaker | 0.52 | 0.52 | 0.2008 (as is) | 0.9793 |

Blend 0.7 × bookmaker + 0.3 × System 2: RPS 0.2003 (bookmaker alone 0.2008)

Illustrative flat-stake betting at average odds (1 unit per bet). 250 matches is far too few: these numbers are luck, not an edge.

| strategy | bets | profit (units) | ROI |
|---|---:|---:|---:|
| JEV System 1 + 2, bet when p × odds > 1.05 | 279 | +8.1 | +2.9% |
| JEV System 1 + 2 + bookmaker, bet when p × odds > 1.02 | 276 | +17.2 | +6.2% |
| always back the bookmaker favourite | 250 | -19.1 | -7.7% |
