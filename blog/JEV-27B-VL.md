# JEV-27B-VL: a decision model that learned to see without a single image of training

*AutoTrust · 30 September 2026*

Today we are releasing **[autotrust/JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL)**, JEV-27B with vision. Show it
images and text, ask a question, and it answers in one forward pass with a calibrated probability for every option. When a
question needs more thought, the same model thinks it through step by step, and it can look at the images while it does.

Its decision head never saw an image during training. It still makes image decisions well enough to compete with a
recommender built from tens of thousands of users' behaviour.

**TL;DR**

* **Zero-shot image recommendation:** looking only at video covers, JEV-27B-VL ranks what a user will watch next as well as
  collaborative filtering learned from **59,045 users' watch histories** (AUC 0.727 vs 0.728), with a higher top-5 hit rate
  (59% vs 49%), and without any interaction data.
* **Pictures beat words:** covers alone score far better than titles alone (AUC 0.727 vs 0.649).
* **Everything JEV-27B does on text, it still does:** it gives the same text decisions. That includes a **RewardBench score of
  89.9** (ahead of GPT-4o, Gemini 1.5 Pro and Claude 3.5 Sonnet used as judges), catching its own hallucinations (71% → 96%
  accuracy on the questions it chooses to answer) and zero-shot news recommendation that beats rankers trained on the data.
* **Open weights**, Apache-2.0, one command to serve.

## One model, two ways to answer, now with eyes

| | what you get | typical time |
|---|---|---|
| **System 1** | a yes/no, pick-one or 0-5 decision over text **and images**, as a calibrated probability for every option | about 0.1 s |
| **System 2** | the full 27B model thinking step by step, with image input | seconds |

Nothing to parse and no prompt gymnastics: you ask *"Is this scenario one where the user clicks this video?"* and get
`{"true": 0.83, "false": 0.17}`.

## It sees: zero-shot image recommendation

We tested it on [MicroLens](https://github.com/westlake-repl/MicroLens), a public dataset of real users of a short-video app
with the original cover images. For each of 200 users we hid the video they actually watched next among 19 videos other people
were watching at the same time, which is what a feed would have shown. JEV-27B-VL looked at the covers of the last five videos
the user watched and at each candidate cover, and ranked the 20 candidates.

![Zero-shot image recommendation on MicroLens](https://huggingface.co/autotrust/JEV-27B-VL/resolve/main/blog/image_recommendation.png)

| method | uses interaction data? | AUC | hit in top 5 |
|---|---|---:|---:|
| random order | no | 0.489 | 21% |
| title similarity | no | 0.602 | 39% |
| JEV-27B-VL reading titles only | no | 0.649 | 46% |
| **JEV-27B-VL looking at covers only** | **no** | **0.727** | **59%** |
| collaborative filtering | yes, 59,045 users | 0.728 | 49% |

Three things stand out:

* **It matches collaborative filtering with zero behaviour data.** Collaborative filtering needs to know who watched what. JEV
  only needs the pictures.
* **New videos are covered from the first second.** A video nobody has watched yet has no co-watch history, but it has a cover.
* **Looking is better than reading.** The covers carry more of a user's taste than the titles do (+0.078 AUC, statistically
  significant).

A typical example: a user had been watching manga-commentary and dubbing clips, including part 1.6 of one manga-commentary
series. Among 20 candidates, JEV-27B-VL put part 1.9 of the same series first, with a click probability of 1.00. That is the
video the user watched next. It recognised the series from the cover art alone, and scored all 20 covers in 2.6 seconds.

The same one-pass decisions work for any question about an image. A cover of a big plate of food gets *"shows food or
cooking" = 1.00*; a game screenshot gets *"video game" = 1.00* out of four categories.

## It still does everything JEV-27B does

JEV-27B-VL gives the same text decisions as [JEV-27B](https://huggingface.co/autotrust/JEV-27B) (same top option on 24 of 24
reference decisions). These results were measured with JEV-27B:

**A judge that beats frontier models used as judges.** On RewardBench, 2,985 human-verified pairs of a better and a worse
answer, System 1 scores **89.9** in one forward pass per ordering, ahead of the published results for Gemini 1.5 Pro (88.2),
GPT-4o (86.7) and Claude 3.5 Sonnet (84.2). It is as accurate as the same model thinking before it judges (0.900 vs 0.902
on a 400-pair sample), and about six times faster.

![RewardBench](https://huggingface.co/autotrust/JEV-27B-VL/resolve/main/blog/judge_rewardbench.png)

**It knows when an answer is made up.** System 2 answers 1,000 trivia questions and gets 71% right. System 1 then checks each
answer. Answering only the half it trusts most gives **96% accuracy**; ranking by the model's own stated confidence gets only
87%. When System 1 says 0.97 or more, the answer was right 99% of the time.

![Hallucination guard](https://huggingface.co/autotrust/JEV-27B-VL/resolve/main/blog/hallucination_guard.png)

**Zero-shot news recommendation.** On real MSN News users (MIND), reading the headlines a user clicked, JEV scores an AUC of
**0.642** with no training, above every zero-shot baseline (best 0.606) and above LightGBM rankers trained on the dataset
(0.590 and 0.616). Added to a trained ranker as one feature, it is worth as much as four times more training data.

![News recommendation](https://huggingface.co/autotrust/JEV-27B-VL/resolve/main/blog/news_recommendation.png)

**Search re-ranking** above a dedicated re-ranker: nDCG@10 **0.858** on TREC-COVID, against 0.793 for bge-reranker-v2-m3 and
0.623 for BM25.

**Thinks only when it needs to.** System 1 answers when it is confident and hands the rest to System 2. With a 0.70
threshold, 70% of questions are answered in 0.11 s and accuracy rises from 0.792 to **0.892**, close to thinking on every
question (0.917).

![System 1 to System 2](https://huggingface.co/autotrust/JEV-27B-VL/resolve/main/blog/system1_to_system2.png)

## Try it

```bash
hf download autotrust/JEV-27B-VL --local-dir JEV-27B-VL
bash JEV-27B-VL/serve.sh        # vLLM, one GPU with 80 GB or more
```

The [model card](https://huggingface.co/autotrust/JEV-27B-VL) has copy-paste code for image decisions and image chat. All the
experiments above, with code and an interactive web app (image recommendation, news recommendation, judge, hallucination
guard and more), are in **[JEV-27B-DEMO](https://github.com/yuhai-china/JEV-27B-DEMO)**.

Image decisions are zero-shot, and the recommendation result above comes from one dataset of 200 users. We would love to hear
what JEV-27B-VL does on your images.
