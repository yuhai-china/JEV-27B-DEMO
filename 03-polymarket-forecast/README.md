# 03 · Polymarket forecasting

**Question.** Can JEV-27B forecast real-world events (politics, economy, geopolitics, crypto, tech) that resolved after its
training data, and can it improve on the prediction-market price?

`forecast.py` compares, on resolved Polymarket markets:

| forecaster | input |
|---|---|
| JEV System 1 | question + resolution rules |
| JEV System 1 + headlines | + Google News headlines published before the forecast date |
| JEV System 2 (thinking) | same inputs; writes an analysis and states a probability |
| JEV System 1 + 2 | System 1 reads the headlines *and* System 2's analysis |
| JEV System 1 + market price | + the market's price on the forecast date (JEV as a "market corrector") |
| market price | the Yes price on the forecast date |

## Status: clean backtest in progress

The measured results are being re-run with a clean design and will be added here. **Fixed calendar forecast dates**
(1 April, 1 May, 1 June, 1 July, 1 August 2026): on each date, the 30 highest-volume non-sports markets that were open and
scheduled to end within 45 days, with the price and headlines as of that date (150 markets).

```bash
python forecast.py fixed           # clean backtest (fixed forecast dates), ~15 min
python forecast.py fixed-summary   # re-print from results_fixed.json
python forecast.py live            # today's top markets
```

## Why the first backtest was withdrawn

Our first backtest (files in [`superseded/`](superseded)) forecast each market **7 days before it actually closed**. That
looked like System 1 + market price improving the market's Brier score from 0.124 to 0.101. On inspection this was an
artefact of the design:

* "Will X happen by date D" markets that resolve **Yes** usually close **early**, as soon as the event happens.
  59% of the Yes markets closed at least 3 days before their scheduled end, versus 33% of the No markets.
* Anchoring the forecast date on the actual close therefore catches Yes markets just before the event, while their price
  still lags. The market *appears* to under-price Yes, and anything that leans Yes looks good.
* A control that simply shifts every market price towards Yes by a constant (fitted leave-one-out) scored **0.088**, better
  than JEV's 0.101. JEV's "gain" was mostly that Yes lean.

So the result is withdrawn. The clean design above uses forecast dates that do not depend on when or how a market resolved.
