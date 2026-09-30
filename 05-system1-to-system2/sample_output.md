120 questions · System 1 median latency 110 ms · System 2 median 3.6 s (12 requests in parallel), median 261 thinking tokens

| policy | answered by System 2 | accuracy | median wait | 90th-percentile wait | mean wait | generated tokens per question |
|---|---:|---:|---:|---:|---:|---:|
| System 1 only | 0% | 0.792 | 0.11 s | 0.1 s | 0.1 s | 0 |
| escalate when System 1 confidence < 0.50 | 14% | 0.825 | 0.11 s | 5.1 s | 5.6 s | 425 |
| escalate when System 1 confidence < 0.60 | 21% | 0.858 | 0.11 s | 22.0 s | 7.7 s | 586 |
| **escalate when System 1 confidence < 0.70** | 30% | **0.892** | 0.11 s | 38.2 s | 8.7 s | 658 |
| escalate when System 1 confidence < 0.80 | 41% | 0.908 | 0.11 s | 39.5 s | 10.0 s | 751 |
| escalate when System 1 confidence < 0.90 | 48% | 0.900 | 0.12 s | 39.5 s | 10.3 s | 775 |
| escalate when System 1 confidence < 0.95 | 55% | 0.900 | 2.44 s | 39.5 s | 10.7 s | 802 |
| escalate when System 1 confidence < 0.99 | 76% | 0.908 | 3.63 s | 39.5 s | 11.4 s | 849 |
| System 2 only | 100% | 0.917 | 3.75 s | 39.5 s | 12.0 s | 892 |

Per source (threshold 0.70):

| source | System 1 | System 2 | gated | sent to System 2 |
|---|---:|---:|---:|---:|
| GSM8K-4 | 0.80 | 1.00 | 0.90 | 27% |
| AQuA-RAT | 0.60 | 0.90 | 0.87 | 47% |
| ARC-Challenge | 0.93 | 1.00 | 0.97 | 13% |
| CommonsenseQA | 0.83 | 0.77 | 0.83 | 33% |

Is System 1's confidence trustworthy? (why gating works)

| System 1 confidence | questions | System 1 accuracy |
|---|---:|---:|
| 0.00 – 0.50 | 17 | 0.53 |
| 0.50 – 0.70 | 19 | 0.42 |
| 0.70 – 0.90 | 21 | 0.86 |
| 0.90 – 0.99 | 34 | 0.97 |
| 0.99 – 1.00 | 29 | 0.93 |

Parse failures in System 2 answers: 2
