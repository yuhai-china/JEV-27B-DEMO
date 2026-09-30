# 07 · Zero-shot short-video recommendation: JEV-27B looks at the covers

> **Zero-shot and multimodal.** JEV-27B has never been trained on MicroLens, on recommendation, or on any click data, and its
> decision head was trained on text only. It looks at the cover images of the videos a user watched, and at a candidate cover.

**Idea.** A TikTok-style short-video feed has to decide, for every user and every new clip, whether to show it. System 1 sees the covers of the last 5 videos a user watched plus one candidate cover, and returns **P(this user
clicks it)** in one forward pass. No interaction logs, no item embeddings, no training. A new video can be recommended from its
cover alone, before anybody has watched it.

**Data.** [MicroLens-100k](https://github.com/westlake-repl/MicroLens): real users of a short-video app with the raw cover images.
For 200 random users, the last video they watched is hidden among 19 videos that other users watched within ±3 days (what a
feed would be showing at the time). Every method ranks the 20 candidates.

![short-video recommendation](../assets/image_recommendation.png)

## JEV looking at covers matches collaborative filtering, with zero interaction data

| method | uses interaction logs? | AUC | HR@1 | HR@5 | NDCG@10 | MRR |
|---|---|---:|---:|---:|---:|---:|
| random order | no | 0.489 | 0.040 | 0.205 | 0.219 | 0.168 |
| popularity so far | counts only | 0.505 | 0.040 | 0.230 | 0.204 | 0.165 |
| title similarity (TF-IDF) | no | 0.602 | 0.205 | 0.385 | 0.367 | 0.328 |
| JEV-27B, titles only | **no, zero-shot** | 0.649 | 0.205 | 0.455 | 0.399 | 0.344 |
| **JEV-27B, covers only** | **no, zero-shot** | **0.727** | 0.280 | **0.590** | 0.498 | 0.428 |
| JEV-27B, covers + titles | **no, zero-shot** | 0.706 | 0.245 | 0.565 | 0.469 | 0.405 |
| item-based collaborative filtering | yes, 59,045 other users | 0.728 | 0.355 | 0.490 | 0.503 | 0.460 |

* **Looking beats reading.** Covers alone lift AUC from 0.649 (titles) to **0.727** (+0.078, 95% interval +0.031 to +0.126).
* **Zero-shot equals collaborative filtering.** Same AUC as item-based CF learned from 59,045 other users' watch histories
  (difference 0.000, interval −0.041 to +0.040), and a higher hit rate in the top 5 (0.59 vs 0.49).
* **Solves cold start.** Collaborative filtering needs co-watch history; JEV only needs the cover, so new videos and new creators can be
  recommended the moment they are uploaded.

## How it asks

```python
from jev_client import decide_mm
parts = ["Short-video app. Covers of the videos this user watched most recently (oldest to newest):\n",
         "1. ", {"image": "covers/101.jpg"}, "\n", "2. ", {"image": "covers/202.jpg"}, "\n", ...,
         "Candidate video: ", {"image": "covers/303.jpg"}]
decide_mm("noul", parts, "Is this scenario one where: this user clicks on the candidate video?")
# e.g. {'false': 0.21, 'true': 0.79}
```

`decide_mm` accepts any mix of text and images (local paths, URLs or data URLs) inside the state.

## Running it

Image input needs [**autotrust/JEV-27B-VL**](https://huggingface.co/autotrust/JEV-27B-VL), JEV-27B with vision: the multimodal
Qwen3.8-27B (identical language weights) with the JEV adapter and decision head. `common/serve_jev27b_mm.sh` downloads and
serves it. Text decisions on this server match the text-only JEV-27B.

```bash
bash common/serve_jev27b_mm.sh     # autotrust/JEV-27B-VL on :8000 (System 1 + System 2, text and images)
python demo.py                     # downloads MicroLens-100k (~700 MB) and runs everything
python demo.py summary             # tables and chart from results.json
```

The web app has a **Video rec (zero-shot)** tab: pick a user, see the covers they watched, and watch JEV rank 20
candidate covers, with the one they actually clicked marked.

**Data licence:** MicroLens is provided by Westlake University for research. It is downloaded from the official site at run
time and not redistributed; this folder stores only user/video IDs and scores.
