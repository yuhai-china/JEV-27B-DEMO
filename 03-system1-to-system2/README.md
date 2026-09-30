# 03 · System 1 → System 2: think only when needed

**Idea.** One vLLM engine serves both systems from the same weights:

| | what it is | typical time |
|---|---|---|
| **System 1** | the decision head: one forward pass, a calibrated probability for every option | ~0.1 s |
| **System 2** | the unmodified Qwen3.8-27B thinking step by step | seconds to a minute |

Because System 1's confidence is calibrated, it can decide *when* to think: answer directly when it is sure, escalate to
System 2 when its top option is below a threshold. There is no second model, no router to train, and no extra GPU.

## Measured on 120 questions with known answers

30 questions each from GSM8K (grade-school maths), AQuA-RAT (algebra word problems), ARC-Challenge (science) and
CommonsenseQA ([`data/questions.json`](data/questions.json)). System 1 latency is measured one request at a time.
System 2 ran 12 requests in parallel on the same GPU.

![system1 to system2](../assets/system1_to_system2.png)

| policy | answered by System 2 | accuracy | median wait | 90th-percentile wait | generated tokens per question |
|---|---:|---:|---:|---:|---:|
| System 1 only | 0% | 0.792 | 0.11 s | 0.1 s | 0 |
| escalate when System 1 confidence < 0.50 | 14% | 0.825 | 0.11 s | 5.1 s | 425 |
| escalate when System 1 confidence < 0.60 | 21% | 0.858 | 0.11 s | 22.0 s | 586 |
| **escalate when System 1 confidence < 0.70** | **30%** | **0.892** | **0.11 s** | 38.2 s | 658 |
| escalate when System 1 confidence < 0.90 | 48% | 0.900 | 0.12 s | 39.5 s | 775 |
| System 2 only | 100% | 0.917 | 3.75 s | 39.5 s | 892 |

**With the 0.70 threshold, 70% of questions are answered in about 0.1 s, and accuracy recovers 80% of the gap between System 1
and System 2** (0.792 → 0.892, versus 0.917 for thinking on everything).

Where the gate helps and where it does not (threshold 0.70):

| source | System 1 | System 2 | gated | sent to System 2 |
|---|---:|---:|---:|---:|
| GSM8K | 0.80 | 1.00 | 0.90 | 27% |
| AQuA-RAT | 0.60 | 0.90 | 0.87 | 47% |
| ARC-Challenge | 0.93 | 1.00 | 0.97 | 13% |
| CommonsenseQA | 0.83 | 0.77 | 0.83 | 33% |

* Maths benefits most from thinking, and System 1 knows it: it sends 27-47% of the maths questions up.
* On CommonsenseQA, thinking is *worse* than the fast answer (0.77 vs 0.83), so escalating does not help there. Thinking longer
  is not always better.

Why gating works: System 1 is right 97% of the time when it is ≥ 0.90 confident and about 50% of the time when it is below 0.70.

| System 1 confidence | questions | System 1 accuracy |
|---|---:|---:|
| below 0.50 | 17 | 0.53 |
| 0.50 – 0.70 | 19 | 0.42 |
| 0.70 – 0.90 | 21 | 0.86 |
| 0.90 – 0.99 | 34 | 0.97 |
| 0.99 and above | 29 | 0.93 |

## Three hand-written examples

```text
- System 1 (0.99 ≥ 0.90, 149 ms): How much does the ball cost? → $0.05
- System 1 (1.00 ≥ 0.90, 119 ms): What should the customer get? → store credit
- System 1 unsure (top “Thursday” at 0.32 < 0.90) → System 2 (3.4 s, 311 tokens): Which weekday is 100 days from today? → Thursday
```

(The classic bat-and-ball trap is answered correctly by System 1. Calendar arithmetic is not something a single forward pass
can do reliably; System 1 knows that, and System 2 gets it right.)

```bash
python demo.py            # everything: System 1 on 120 questions, System 2 on 120 questions (~3-10 min), tables, chart
python demo.py summary    # re-print from results.json
python demo.py examples   # the three examples above
```

```python
p = decide("choice", context, question, options)          # System 1
best = max(p, key=p.get)
if p[best] < 0.70:                                          # unsure -> think
    reasoning, answer = split_thinking(chat(prompt, thinking=True, max_tokens=8000))
```

**Data licences:** GSM8K (MIT), AQuA-RAT (Apache-2.0), ARC (CC BY-SA 4.0), CommonsenseQA (MIT).
