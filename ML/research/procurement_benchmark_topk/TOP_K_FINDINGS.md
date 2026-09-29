# Top-K Candidate Analysis Findings (Step-37 Frozen T3 Model)

## 1. Top-K Accuracy@10% ($N = 10,560$)

| Metric | Accuracy (@ $\pm 10\%$) | Correct Queries |
| :--- | :---: | :---: |
| **Rank-1 Accuracy** | **37.57%** | 3,967 |
| **Top-3 Accuracy** | **48.49%** | 5,121 |
| **Top-5 Accuracy** | **54.14%** | 5,717 |
| **Top-10 Accuracy** | **62.64%** | 6,615 |

---

## 2. Recovery of Rank-1 Failures by Top-5

* **Recovered Queries**: **1,750**
* **Population Percentage**: **16.57%** of all 10,560 validation queries
* **Failure Set Recovery**: **26.54%** of all 6,593 Rank-1 failure cases contain a valid ($\pm 10\%$) candidate within Top-5.

---

## 3. Recovery of Rank-1 Failures by Top-10

* **Recovered Queries**: **2,648**
* **Population Percentage**: **25.08%** of all 10,560 validation queries
* **Failure Set Recovery**: **40.16%** of all 6,593 Rank-1 failure cases contain a valid ($\pm 10\%$) candidate within Top-10.

---

## 4. Interpretation for LLM Semantic Reranking

The frozen tabular LambdaMART ranker reliably identifies high-quality candidates, but its Rank-1 accuracy (37.57%) is constrained by subtle scope and specification differences that shallow structured features cannot fully untangle. Crucially, over 40% of these Rank-1 failures (2,648 queries) already possess an accurate price ($\pm 10\%$) residing within the Top-10 candidates, pushing the reachable accuracy ceiling to 62.64%. Passing this compact Top-5 or Top-10 candidate pool into a downstream LLM semantic reranker transforms pricing into a closed-context selection task where language comprehension can resolve technical incompatibilities and recover misranked items without searching the entire catalog.
