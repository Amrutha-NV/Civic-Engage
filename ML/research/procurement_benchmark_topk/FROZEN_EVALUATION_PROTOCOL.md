# Frozen Evaluation Protocol: LLM Semantic Reranking Benchmark

**Document Version:** 1.0 (Frozen Research Baseline)  
**Location:** `D:\CIVIC-ENGAGE\ML\research\procurement_benchmark_topk\FROZEN_EVALUATION_PROTOCOL.md`  
**Evaluation Population:** CivicEngage Frozen Validation Set ($N = 10,560$ queries)  
**Artifacts Referenced:**  
- [`frozen_evaluation_dataset.parquet`](./frozen_evaluation_dataset.parquet)  
- [`top_10_candidates.csv`](./top_10_candidates.csv)  
- [`HISTORICAL_EVIDENCE_GATE_CONTRACT.md`](./HISTORICAL_EVIDENCE_GATE_CONTRACT.md)  
- [`LLM_RERANKING_INPUT_SPEC.md`](./LLM_RERANKING_INPUT_SPEC.md)  

---

# 1. Purpose

This protocol establishes the formal, immutable evaluation procedure for comparing:

* **System A (Baseline):** The existing frozen Step-37 tabular LambdaMART ranker.
* **System B (Experimental):** The existing LambdaMART retriever + Historical Evidence Gate + Person 2 LLM Semantic Reranking pipeline.

The scientific objective is to determine whether downstream natural language comprehension can resolve subtle specification mismatches and improve historical procurement candidate selection over the tabular prior, **without altering the underlying retrieval pool, feature representations, or validation population**.

---

# 2. Frozen Evaluation Population

All evaluations must be conducted strictly on the pre-computed, self-contained evaluation artifact:

* **Artifact File:** `frozen_evaluation_dataset.parquet`
* **Total Rows:** $105,600$
* **Validation Queries ($N$):** $10,560$ (representing the 2023–2024 temporal validation split)
* **Candidates per Query:** Exactly $10$ ranked candidates per query
* **Candidate Ranks:** $1$ through $10$

> [!CRITICAL]
> The evaluation artifact `frozen_evaluation_dataset.parquet` is **frozen**. It must not be modified, sampled, re-indexed, or filtered prior to evaluation. Every query in the validation population must be accounted for in the evaluation ledger.

---

# 3. Baseline System

The baseline system represents the pure tabular machine learning pipeline:

```
[ Requisition Query ] ──► [ Step-37 Retrieval ] ──► [ 176-Feature LambdaMART ] ──► [ Rank-1 Candidate ] ──► candidate_unit_price[rank=1]
```

* **Candidate Selection:** The candidate at `candidate_rank == 1` is selected automatically for every query.
* **Price Output:** The predicted benchmark unit price is strictly `candidate_unit_price` of the Rank-1 candidate.
* **Immutability:** The baseline model parameters, candidate rankings, and candidate scores are frozen in `frozen_evaluation_dataset.parquet` and `models/procurement_ranker.joblib`. The baseline model must not be retrained, fine-tuned, or modified.

---

# 4. LLM Reranking Experiment

The experimental pipeline evaluates candidate quality through sequential statistical and semantic gates:

```
[ Requisition Query ]
         │
         ▼
[ Top-10 Candidates ]
         │
         ▼
[ Historical Evidence Gate ] ──(composite < 0.0)──► [ HISTORICAL_INSUFFICIENT ] ──► (Person 3 Fallback)
         │
    (composite >= 0.0)
         │
         ▼
[ Person 2: LLM Reranker ] ──(semantic reject)──► [ HISTORICAL_INSUFFICIENT ] ──► (Person 3 Fallback)
         │
  (HISTORICAL_ACCEPTED)
         │
         ▼
selected_candidate_rank (1–10)
         │
         ▼
predicted_unit_price = candidate_unit_price[selected_candidate_rank]
```

### Operational Constraints for Person 2:
1. **Candidate Rank Bound:** The LLM may select **only** an integer rank in $\{1, 2, \dots, 10\}$ from the supplied candidate array.
2. **Zero Price Hallucination:** The LLM must **never** generate, interpolate, estimate, inflation-adjust, or average candidate prices. It must not output a price field.
3. **Price Resolution:** The predicted historical unit price is deterministically resolved by the harness as:
   $$P_{\text{predicted}} = \text{candidates}[\text{selected\_candidate\_rank} - 1].\text{candidate\_unit\_price}$$
4. **Rejection Semantics:** If Person 2 concludes that all 10 candidates have critical specification flaws, it returns `"decision": "HISTORICAL_INSUFFICIENT"`, marking the query as rejected by semantic evidence and triggering the Person 3 current-market fallback.

---

# 5. Primary Metric

The primary benchmark metric is **Accuracy@10%**.

A price prediction is considered an accurate **Hit** if its absolute percentage error relative to ground truth does not exceed $10\%$:

$$\text{APE} = \frac{|P_{\text{predicted}} - P_{\text{actual}}|}{P_{\text{actual}}} \times 100 \le 10.0\%$$

When reporting results across any query set $S$, researchers must report:
* **Numerator:** Number of hits ($\sum_{i \in S} \mathbb{I}[\text{APE}_i \le 10.0\%]$)
* **Denominator:** Total query count ($|S|$)
* **Percentage:** $\frac{\text{Numerator}}{\text{Denominator}} \times 100\%$

> **Direct Comparison Rule:** The primary metric must be reported over the identical $10,560$ validation queries for both baseline and experimental systems wherever a direct comparison is possible.

---

# 6. Secondary Metrics

To provide a comprehensive evaluation of pricing accuracy and ranking quality, the following secondary metrics must be reported:

### A. Price Selection Metrics (Evaluated on Final Selected Price)
1. **Accuracy@5%:** $\frac{1}{|S|} \sum_{i \in S} \mathbb{I}[\text{APE}_i \le 5.0\%]$
2. **Accuracy@20%:** $\frac{1}{|S|} \sum_{i \in S} \mathbb{I}[\text{APE}_i \le 20.0\%]$
3. **Median Absolute Percentage Error (MdAPE):** $\text{median}(\{\text{APE}_i : i \in S\})$
4. **Mean Absolute Error (MAE):** $\frac{1}{|S|} \sum_{i \in S} |P_{\text{predicted}, i} - P_{\text{actual}, i}|$

### B. Candidate Pool & Ranking Coverage Metrics (Evaluated on Candidate Set)
1. **Top-3 Accuracy@10%:** Fraction of queries containing at least one candidate within $\pm 10\%$ in ranks 1–3.
2. **Top-5 Accuracy@10%:** Fraction of queries containing at least one candidate within $\pm 10\%$ in ranks 1–5.
3. **Top-10 Accuracy@10%:** Fraction of queries containing at least one candidate within $\pm 10\%$ in ranks 1–10.
4. **Mean Reciprocal Rank of Valid Candidate (MRR@10):**
   $$\text{MRR@10} = \frac{1}{|S|} \sum_{i \in S} \frac{1}{\min(\{r \in [1, 10] : \text{APE}_{i, r} \le 10.0\%\} \cup \{\infty\})}$$

> **Distinction Requirement:** Researchers must clearly distinguish between **candidate retrieval/ranking capacity** (e.g. Top-10 Coverage = $62.65\%$) and **final price-selection accuracy** (e.g. Baseline Rank-1 = $37.58\%$).

---

# 7. Evaluation-Only Fields

The following fields in `frozen_evaluation_dataset.parquet` are strictly reserved for post-prediction evaluation:

* `actual_unit_price`
* `is_within_10pct`
* `absolute_percentage_error`

### Strict Non-Leakage Rules:
1. These fields **must never** be included in the prompt, context window, or metadata supplied to the LLM.
2. They **must never** be accessed during candidate filtering, evidence gating, or reranking decisions.
3. They are used **solely by the evaluation harness** after all selections are frozen to compute benchmark metrics.

---

# 8. Fair Comparison Rules

To ensure scientific validity and parity between Baseline and Experimental runs:

1. **Identical Query Population:** Both systems must be evaluated on the exact same $10,560$ validation queries.
2. **Identical Candidate Inputs:** The candidate text, candidate scores, and candidate unit prices must be identical to those in `frozen_evaluation_dataset.parquet`.
3. **No Retraining:** The baseline model and candidate retriever must not be retrained or altered.
4. **Zero Outcome Leakage:** No actual procurement prices or hit indicators may influence LLM selection.
5. **Standardized Error Formula:** Absolute percentage error must use the exact formula defined in Section 5.

---

# 9. Handling `HISTORICAL_INSUFFICIENT`

When a query fails historical evidence criteria, it can occur at one of two distinct stages:

* **Case A: Pre-LLM ML Evidence Gate Rejection**  
  The gate calculation yields $\text{composite\_evidence} < 0.0$ (or $K = 0$). Person 2 is bypassed entirely.
* **Case B: Post-LLM Semantic Reranker Rejection**  
  The query passes the ML gate ($\text{composite\_evidence} \ge 0.0$), but Person 2 inspects the candidate descriptions and returns `"decision": "HISTORICAL_INSUFFICIENT"` due to specification mismatches.

### Evaluation Accounting Rules:
1. **Ledger Distinction:** The evaluation ledger must record `gate_status` (`SUFFICIENT` vs. `INSUFFICIENT`) and `llm_decision` (`ACCEPTED`, `INSUFFICIENT`, or `BYPASSED`) separately.
2. **No Free Hits:** A rejected query **must never** be counted as a successful prediction in historical benchmark metrics.
3. **Dual Metric Reporting:**
   - **Gated Precision (Accepted Pool):** Accuracy evaluated exclusively on queries where historical evidence was accepted ($\text{decision} == \text{HISTORICAL\_ACCEPTED}$).
   - **End-to-End Historical Coverage:** Accuracy evaluated across the full $10,560$ query population (treating rejected queries as non-hits for pure historical pricing).
4. **Decoupled Architecture:** The historical reranking experiment and Person 3's current-market fallback are independent research stages. They must be evaluated separately before any combined end-to-end routing experiment is defined.

---

# 10. Required Experiment Outputs

Any research report or evaluation submission for Person 2 must provide:

### 10.1 Aggregate Summary Metrics
* Total queries evaluated ($10,560$)
* Pre-LLM Gate pass count & pass rate (%)
* Pre-LLM Gate reject count & reject rate (%)
* Post-LLM acceptance count & acceptance rate (%)
* Post-LLM rejection count & rejection rate (%)
* Baseline Accuracy@10% ($37.58\%$)
* Experimental Accuracy@10% (on accepted pool and on total population)
* Accuracy@5% and Accuracy@20%
* MdAPE and MAE
* Rank distribution of LLM selections ($r = 1, 2, \dots, 10$)

### 10.2 Per-Query Evaluation Record
A serialized dataset (CSV or Parquet) containing exactly one record per validation query ($10,560$ rows) with the following mandatory schema:

| Column Name | Type | Description |
| :--- | :---: | :--- |
| `query_index` | `int64` | Validation query index ($0$ to $10,559$). |
| `baseline_rank` | `int64` | Always `1`. |
| `baseline_unit_price` | `float64` | `candidate_unit_price` of Rank 1 candidate. |
| `gate_status` | `string` | `HISTORICAL_SUFFICIENT` or `HISTORICAL_INSUFFICIENT`. |
| `llm_decision` | `string` | `HISTORICAL_ACCEPTED`, `HISTORICAL_INSUFFICIENT`, or `BYPASSED`. |
| `selected_candidate_rank` | `int64 / null` | Rank selected by LLM ($1$–$10$), or `null` if rejected/bypassed. |
| `selected_unit_price` | `float64 / null` | Resolved unit price, or `null` if rejected/bypassed. |
| `actual_unit_price` | `float64` | Ground-truth unit price (evaluation only). |
| `baseline_absolute_percentage_error` | `float64` | Rank-1 baseline APE relative to actual. |
| `llm_absolute_percentage_error` | `float64 / null` | Selected candidate APE relative to actual (null if rejected). |
| `is_llm_within_10pct` | `bool / null` | True if LLM selected price is within $\pm 10\%$. |

---

# 11. Reproducibility

An independent researcher must be able to reproduce the baseline and experimental comparison using solely the artifacts in this repository:

1. [`frozen_evaluation_dataset.parquet`](./frozen_evaluation_dataset.parquet): The frozen candidate and ground-truth source.
2. [`top_10_candidates.csv`](./top_10_candidates.csv): Candidate pool reference.
3. [`HISTORICAL_EVIDENCE_GATE_CONTRACT.md`](./HISTORICAL_EVIDENCE_GATE_CONTRACT.md): Protocol for gating and routing.
4. [`LLM_RERANKING_INPUT_SPEC.md`](./LLM_RERANKING_INPUT_SPEC.md): Input schema for Person 2 prompts.
5. `FROZEN_EVALUATION_PROTOCOL.md`: This evaluation protocol.

---

# 12. Limitations & Boundary Conditions

1. **Historical Evidence Scope:** This experiment evaluates reranking of *past public procurement purchase orders*. It does not evaluate real-time market inflation or current supplier stock.
2. **Decoupling from Person 3:** Live vendor catalog lookups and current-market indices are the responsibility of Person 3. Failure of historical reranking triggers Person 3, but does not invalidate the historical reranking evaluation.
3. **Retrieval Upper Bound:** An LLM reranker cannot recover queries where no valid candidate exists in the Top-10 ($37.35\%$ of total queries). The maximum achievable accuracy for pure historical candidate selection is bounded by Top-10 coverage ($62.65\%$).
4. **Domain Generalization:** The frozen dataset represents Texas municipal and state purchase order records from 2023–2024. Results should not be generalized across unverified international procurement domains without domain calibration.

---

# 13. Research Comparison Table

| Dimension / Component | System A: Baseline | System B: LLM Semantic Reranking |
| :--- | :--- | :--- |
| **Retrieval Engine** | Frozen Step-37 Multi-View Retrieval | Same (Frozen Step-37 Multi-View Retrieval) |
| **Candidate Pool** | Top-10 Ranked Candidates | Same (Top-10 Ranked Candidates) |
| **Tabular Scoring** | 176-Feature LambdaMART | Same (176-Feature LambdaMART) |
| **Evidence Gating** | None (All queries evaluated at Rank 1) | Historical Evidence Gate ($\text{composite} \ge 0.0$) |
| **Candidate Selection** | Hard-coded Rank 1 | Closed-Context LLM Semantic Matching |
| **Price Invention** | Not Applicable | **Strictly Prohibited** |
| **Evaluation Population** | Identical $10,560$ Validation Queries | Identical $10,560$ Validation Queries |
| **Primary Metric** | Accuracy@10% ($37.58\%$) | Accuracy@10% |
| **Evaluation Ground Truth** | Same `actual_unit_price` | Same `actual_unit_price` |

---

# 14. No Production Changes

> [!NOTE]
> This protocol is strictly a **research and benchmark specification**.
> - It does **not** alter, retrain, or replace any model in `ML/procurement_benchmark_engine/`.
> - It does **not** modify `frozen_evaluation_dataset.parquet`.
> - It does **not** modify production inference scripts, dependencies, or the FastAPI service.
