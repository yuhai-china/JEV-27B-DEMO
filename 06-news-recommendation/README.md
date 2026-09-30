# 06 · Zero-shot news recommendation

> **Zero-shot.** JEV-27B has never been trained on MIND, on news recommendation, or on any click data. It gets no user
> embeddings, no article click counts and no training step. It only reads text.

**Idea.** System 1 reads the headlines a user clicked recently and one candidate article, and returns **P(this user clicks
it)** in one forward pass. Sort the candidates by that number and you have a recommender that works on day one: for new users
after a few clicks, and for brand-new articles from the moment they are published, with no click history needed.

**Data.** [MIND](https://msnews.github.io/) (Microsoft News Dataset), MIND-small: real MSN News users. Evaluation on 500 impressions
from 15 November 2019 (9,346 candidate articles, 6.8% clicked). Each method ranks the articles shown in each impression. Metrics
are the standard MIND ones, averaged per impression.

![news](../assets/news_recommendation.png)

## 1. Zero-shot against zero-shot: JEV wins clearly

None of these methods is trained on MIND.

| method | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| random order | 0.515 | 0.278 | 0.284 | 0.367 |
| popularity so far (clicks before the impression) | 0.546 | 0.315 | 0.340 | 0.404 |
| category match with the user's history | 0.606 | 0.346 | 0.384 | 0.447 |
| title similarity with the user's history (TF-IDF) | 0.546 | 0.315 | 0.331 | 0.407 |
| **JEV-27B System 1, zero-shot** | **0.642** | **0.388** | **0.424** | **0.486** |

JEV's AUC is 0.036 above the best zero-shot baseline (95% bootstrap interval +0.008 to +0.064). It reads *what the user cares
about* from their headlines and *what the article is about* from its title and abstract, instead of matching words or
categories.

## 2. Zero-shot JEV beats rankers trained on MIND

For reference, LightGBM learning-to-rank models trained on MIND training impressions (9-14 Nov), with popularity, category
and title-similarity features:

| method | trained on MIND? | AUC | nDCG@10 |
|---|---|---:|---:|
| LightGBM ranker | yes, 1,000 impressions | 0.590 | 0.434 |
| LightGBM ranker | yes, 4,000 impressions | 0.616 | 0.462 |
| **JEV-27B System 1** | **no, zero-shot** | **0.642** | **0.486** |

## 3. Add JEV to a trained model: as good as 4× more training data

Same 1,000 training impressions for both. The only difference is JEV's zero-shot score as one extra feature:

| method | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| LightGBM ranker | 0.590 | 0.339 | 0.374 | 0.434 |
| **LightGBM ranker + JEV zero-shot score** | **0.617** | **0.369** | **0.393** | **0.459** |
| LightGBM ranker with 4× more training data, no JEV | 0.616 | 0.361 | 0.394 | 0.462 |

One JEV feature is worth as much as quadrupling the training data.

## How it asks

```python
decide("noul", {
    "news_recently_clicked_by_this_user (oldest to newest)": ["sports › football_nfl: ...", "news › newspolitics: ...", ...],
    "candidate_article": {"category": "sports › football_nfl", "title": "...", "abstract": "..."}},
    "Is this scenario one where: this user clicks on the candidate article?")
# e.g. {'false': 0.83, 'true': 0.17}
```

All candidates are scored concurrently. The engine handled about 29 decisions per second on one GPU for this run (28,920
decisions).

```bash
python demo.py            # download MIND-small, score with JEV (cached in jev_scores.json), train the rankers, tables + chart
python demo.py eval       # re-evaluate without calling JEV
python demo.py summary    # re-print from summary.json
```

The web app has a **News recommendation (zero-shot)** tab: pick a real user, see their recent headlines, and watch JEV rank
the articles they were shown, with the one they actually clicked marked.

**Data licence:** MIND is released by Microsoft under the [Microsoft Research License Terms](https://msnews.github.io/)
(non-commercial research). It is downloaded at run time and not redistributed here; this folder stores only article IDs and scores.
