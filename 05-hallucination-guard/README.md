# 05 · Hallucination guard: answer only when the answer can be trusted

**Idea.** System 2 answers a question. System 1 then reads *(question, answer)* and returns **P(the answer is correct)** in one
forward pass. If that probability is low, the assistant says "I'm not sure" (or escalates to a search, a human, or a longer
think) instead of stating a made-up fact.

Measured on **1,000 random TriviaQA questions**. System 2 answers directly in a few words, and its answers are 71.2% correct.

![guard](../assets/hallucination_guard.png)

## Answer the trusted half and be right 96% of the time

| questions answered | accuracy with the System 1 check | accuracy with System 2's own confidence |
|---:|---:|---:|
| 30% | **97.7%** | 85.3% |
| 50% | **96.4%** | 87.2% |
| 70% | **90.3%** | 86.0% |
| 90% | **77.2%** | 76.0% |
| 100% | 71.2% | 71.2% |

Asking the model "how confident are you?" is the usual approach, and it barely helps: the stated confidences use only 7
distinct values, and 86% of them are 0.9 or higher (69% say exactly 1.0). The System 1 check separates right from wrong answers far better.

| | System 1 check | System 2's own confidence |
|---|---:|---:|
| separates right from wrong answers (AUROC, 1.0 = perfect) | **0.900** | 0.773 |
| distinct confidence values | 385 | 7 |

## When System 2 is sure and wrong

System 2 said it was at least 90% sure on 184 answers that were wrong. System 1 scored 54 of those below 0.6. Examples:

| question | System 2 answered | System 2's confidence | System 1 check | correct answer |
|---|---|---:|---:|---|
| The line 'The mirror crack'd from side to side' comes from which poem? | The Jabberwocky | 0.95 | **0.37** | The Lady of Shalott |
| Which musical instrument can have 21, 22, or 23 strings? | Harp | 0.95 | **0.41** | Sitar |
| The Florentine Girdle was a type of what? | Diamond | 1.00 | **0.41** | Chastity belt |
| Which professional golfer has three nicknames, one of which is 'The Wild Thing'? | Ernie Els | 0.90 | **0.42** | John Daly |

## The number means what it says

| System 1 says the answer is correct with probability | answers | actually correct |
|---|---:|---:|
| 0.97 – 1.00 | 223 | **99.1%** |
| 0.90 – 0.97 | 256 | 94.1% |
| 0.60 – 0.90 | 339 | 62.2% |
| 0.30 – 0.60 | 136 | 22.1% |
| below 0.30 | 46 | 19.6% |

The check adds about **0.24 s** per answer.

```bash
python demo.py            # 1,000 questions (~1 min on one GPU), tables and chart
python demo.py summary    # re-print from results.json
```

```python
from demo import answer, trust
q = "Which musical instrument can have 21, 22, or 23 strings?"
a = answer(q)                     # System 2
if trust(q, a) < 0.9:             # System 1
    a = "I'm not sure."
```

Grading uses the TriviaQA answer aliases with lenient matching (exact, contained either way, or high word overlap).
Every question, answer and score is in [`results.json`](results.json). Data: TriviaQA (Apache-2.0).
