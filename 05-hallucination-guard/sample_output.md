1000 TriviaQA questions · System 2 answers directly · accuracy of its answers: **71.2%**

| | System 1 check (one forward pass) | System 2 asked for its confidence |
|---|---:|---:|
| separates right from wrong answers (AUROC, 1.0 = perfect) | **0.900** | 0.773 |
| average confidence (actual accuracy 71.2%) | 79.7% | 85.0% |
| distinct confidence values used | 385 | 7 |

Answer only the questions the check trusts most, and say "I'm not sure" otherwise:

| questions answered | accuracy with System 1 check | accuracy with System 2's own confidence |
|---:|---:|---:|
| 30% | **97.7%** | 85.3% |
| 50% | **96.4%** | 87.2% |
| 70% | **90.3%** | 86.0% |
| 90% | **77.2%** | 76.0% |
| 100% | 71.2% | 71.2% |

How far System 1's number can be trusted:

| System 1 says | answers | actually correct |
|---|---:|---:|
| 0.00 – 0.30 | 46 | 19.6% |
| 0.30 – 0.60 | 136 | 22.1% |
| 0.60 – 0.90 | 339 | 62.2% |
| 0.90 – 0.97 | 256 | 94.1% |
| 0.97 – 1.00 | 223 | 99.1% |

Median time of the System 1 check: 237 ms (24 questions in flight)
