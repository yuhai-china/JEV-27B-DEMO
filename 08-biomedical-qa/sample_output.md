500 expert-labelled questions (official test set) · 500 decisions in 27 s

**Accuracy 77.8% · macro-F1 63.1%**

| model | accuracy (%) |
|---|---:|
| GPT-4 (Medprompt) | 82.0 |
| Med-PaLM 2 | 81.8 |
| MEDITRON 70B | 81.6 |
| Claude 3 | 79.7 |
| Flan-PaLM 540B (3-shot) | 79.0 |
| Human performance | 78.0 |
| **JEV-27B System 1 (zero-shot, one pass)** | **77.8** |
| Galactica 120B | 77.6 |
| GPT-4 (Nori et al. 2023) | 75.2 |
| PubMedGPT 2.7B | 74.4 |
| PMC-LLaMA 7B | 73.4 |
| BioLinkBERT large | 72.2 |
| BioBERT | 68.1 |

Confusion (rows = expert answer, columns = JEV):

| expert \ JEV | yes | no | maybe |
|---|---:|---:|---:|
| yes | 238 | 15 | 23 |
| no | 15 | 140 | 14 |
| maybe | 29 | 15 | 11 |
