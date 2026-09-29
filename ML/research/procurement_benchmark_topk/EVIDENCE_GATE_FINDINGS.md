# Historical Evidence Gate: Empirical Findings & Architectural Specification

**Research Module:** Step 1 — Historical Evidence Gate  
**Location:** `ML/research/procurement_benchmark_topk/`  
**Dataset Artifact:** `frozen_evaluation_dataset.parquet` ($N = 10,560$ validation queries, $105,600$ candidate rows)  
**Execution Script:** [`evidence_gate_analysis.py`](./evidence_gate_analysis.py)  
**Quantitative Metrics:** [`evidence_gate_metrics.json`](./evidence_gate_metrics.json)  

---

## 1. Executive Summary & Gate Role

The **Historical Evidence Gate** serves as the initial decision filter in the CivicEngage Procurement Benchmark pipeline. Its purpose is to evaluate the quality and statistical coherence of the Top-10 historical candidates retrieved by the frozen tabular LambdaMART ranker **before invoking downstream components**.

```
                        [ Target Line-Item Requisition ]
                                      │
                                      ▼
                        [ Frozen Tabular Ranker (Step-37) ]
                                      │
                                      ▼
                           [ Top-10 Candidate Pool ]
                                      │
                                      ▼
                     ┌──────────────────────────────────┐
                     │    HISTORICAL EVIDENCE GATE      │
                     │  (Inference-Safe Signal Analysis)│
                     └─────────────────┬────────────────┘
                                       │
                      ┌────────────────┴────────────────┐
                      ▼                                 ▼
           [ HISTORICAL_SUFFICIENT ]         [ HISTORICAL_INSUFFICIENT ]
                      │                                 │
                      ▼                                 ▼
           [ Person 2: LLM Reranker ]       [ Person 3: Market Evidence ]
            (Closed-Context Semantic           (Live Market Search &
             Selection from Top-10)             Commercial Retrieval)
```

### Strict Gate Invariants
* **The gate does NOT generate a price.**
* **The gate does NOT modify `candidate_unit_price`.**
* **The gate does NOT invoke an LLM.**
* **The gate does NOT perform web search or current-market retrieval.**
* **The gate does NOT modify the frozen production model or dataset.**

---

## 2. Signal Availability & Safety Taxonomy

To guarantee zero data leakage during real-world inference, all signals are strictly bifurcated:

| Category | Field Name(s) | Availability | Inference-Safe? | Role / Notes |
| :--- | :--- | :---: | :---: | :--- |
| **Candidate ML Scores** | `candidate_score` (Ranks 1–10) | Parquet | **YES** | Raw LambdaMART ensemble score indicating similarity & prior ranker confidence. |
| **Candidate Unit Prices** | `candidate_unit_price` (Ranks 1–10) | Parquet | **YES** | Historical unit prices of candidates. Used to compute intra-pool price dispersion. |
| **Candidate Extended Text** | `candidate_description` | Parquet | **YES** | Historical item description. |
| **Target Line Item Metadata** | `target_description`, `target_quantity`, `target_unit_of_measure`, `target_procurement_date`, `target_city`, `target_state`, `target_brand`, `target_model`, `target_commodity_code`, `target_commodity_family` | Parquet | **YES** | Public requisition context available at runtime. |
| **Evaluation Ground Truth** | `actual_unit_price` | Parquet | **NO (EVAL ONLY)** | Ground truth price. **Strictly forbidden during inference.** Used solely for research validation. |
| **Evaluation Error Metrics** | `absolute_percentage_error`, `is_within_10pct` | Parquet | **NO (EVAL ONLY)** | Accuracy indicators. **Strictly forbidden during inference.** |
| **Candidate Structured Specs** | *Candidate vendor, award date, quantity, UOM, contract type, specs* | Absent | **N/A** | **NOT AVAILABLE** in frozen artifact. Not used or fabricated. |

---

## 3. Key Empirical Findings

From the empirical analysis of all $10,560$ validation queries in [`evidence_gate_metrics.json`](./evidence_gate_metrics.json):

### 3.1 Baseline Benchmark Accuracies
* **Validation Queries ($N$)**: $10,560$
* **Candidates per Query**: Exactly $10$ for every query.
* **Rank-1 Accuracy ($\pm 10\%$)**: **$37.58\%$** ($3,968$ queries)
* **Top-5 Accuracy ($\pm 10\%$)**: **$54.14\%$** ($5,717$ queries)
* **Top-10 Accuracy ($\pm 10\%$)**: **$62.65\%$** ($6,616$ queries)
* **Completely Unrecoverable Queries**: **$37.35\%$** ($3,944$ queries)
  * *For these $3,944$ queries, not a single candidate among the Top-10 has a price within $\pm 10\%$ of actual.*

### 3.2 Signal 1: Top-1 Candidate Score (`score_r1`)
The raw ensemble score of the top-ranked candidate is a strong predictor of candidate pool validity.
* **Distribution**: Mean $= 1.2179$, Median $= 1.1588$, IQR $= 1.4302$ (P10 $= 0.0873$, P90 $= 2.4525$).
* **Predictive Power**:
  * Pearson correlation with Top-10 Hit: **$+0.3952$** (Rank-1 Hit: $+0.4800$).
  * **ROC-AUC (Top-10 Hit)**: **$0.7381$** (Top-5 Hit: $0.7515$, Rank-1 Hit: $0.7834$).
* **Quintile Performance**:
  * Lowest Quintile (`score_r1 <= 0.341`): Top-10 Hit Rate is only **$32.91\%$** (**$67.09\%$ unrecoverable**).
  * Highest Quintile (`score_r1 > 2.080`): Top-10 Hit Rate reaches **$88.73\%$** (only $11.27\%$ unrecoverable).

### 3.3 Signal 2: Candidate Price Dispersion (`log_price_std_top10`)
The standard deviation of natural log candidate unit prices across the Top-10 candidates captures intra-pool disagreement.
$$\sigma_{\ln(P)} = \sqrt{\frac{1}{10} \sum_{i=1}^{10} \left( \ln(P_i) - \overline{\ln(P)} \right)^2}$$
* **Distribution**: Mean $= 0.6895$, Median $= 0.5530$, IQR $= 0.9358$ (P10 $= 0.0364$, P90 $= 1.5751$).
* **Predictive Power**:
  * Pearson correlation with Top-10 Hit: **$-0.3555$** (Rank-1 Hit: $-0.4008$).
  * **ROC-AUC (Top-10 Hit)**: **$0.7172$** (Top-5 Hit: $0.7383$, Rank-1 Hit: $0.7707$).
* **Quintile Performance**:
  * Lowest Dispersion Quintile (`std <= 0.120`): Top-10 Hit Rate is **$83.95\%$** (Rank-1 Hit Rate: $74.72\%$).
  * Highest Dispersion Quintile (`std > 1.234`): Top-10 Hit Rate plummets to **$37.74\%$** (**$62.26\%$ unrecoverable**).

### 3.4 Signal 3: Composite Evidence Score
Combining candidate score strength and price consistency yields a highly effective composite index:
$$\text{composite\_evidence} = \text{candidate\_score}_{r=1} - \sigma_{\ln(P)}$$
* Logistic regression coefficients on standardized features confirm equal and opposing weights ($+0.8412$ vs. $-0.8103$).
* **ROC-AUC (Top-10 Hit)**: **$0.7630$**
* **ROC-AUC (Top-5 Hit)**: **$0.7794$**
* **ROC-AUC (Rank-1 Hit)**: **$0.8099$**
* **Performance Across Composite Quintiles**:
  1. Quintile 1 ($\le -0.703$): Top-10 Hit = **$30.73\%$** ($69.27\%$ unrecoverable)
  2. Quintile 2 ($-0.703$ to $0.109$): Top-10 Hit = **$48.11\%$** ($51.89\%$ unrecoverable)
  3. Quintile 3 ($0.109$ to $0.919$): Top-10 Hit = **$66.24\%$** ($33.76\%$ unrecoverable)
  4. Quintile 4 ($0.919$ to $1.798$): Top-10 Hit = **$79.36\%$** ($20.64\%$ unrecoverable)
  5. Quintile 5 ($> 1.798$): Top-10 Hit = **$88.83\%$** ($11.17\%$ unrecoverable)

---

## 4. Proposed Gate Policies & Threshold Recommendations

Depending on system-level latency, API cost, and accuracy priorities, three operating policies are evaluated:

| Policy | Threshold Rule | Pass Rate | Passed Top-10 Acc | Passed Top-5 Acc | Passed Rank-1 Acc | Rejection Rate | Rejected Top-10 Acc | Rejected Unrecoverable % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (No Gate)** | *All queries pass* | $100.0\%$ | $62.65\%$ | $54.14\%$ | $37.58\%$ | $0.0\%$ | N/A | $37.35\%$ |
| **Policy 1: Permissive (High Coverage)** | $\text{composite} \ge -1.0$ | **$86.54\%$** | $67.92\%$ | $59.66\%$ | $42.38\%$ | $13.46\%$ | $28.78\%$ | **$71.22\%$** |
| **Policy 2: Balanced (Recommended)** | $\text{composite} \ge 0.0$ | **$62.91\%$** | **$77.18\%$** | **$70.47\%$** | **$53.17\%$** | $37.09\%$ | $38.01\%$ | **$61.99\%$** |
| **Policy 3: Strict (High Precision)** | $\text{composite} \ge 0.7$ | **$45.67\%$** | **$82.52\%$** | **$76.65\%$** | **$61.70\%$** | $54.33\%$ | $45.95\%$ | **$54.05\%$** |
| **Policy 4: Dual-Threshold Explicit** | $\text{score}_{r=1} \ge 0.5 \land \sigma_{\ln(P)} \le 1.0$ | **$57.89\%$** | **$78.23\%$** | **$71.45\%$** | **$54.31\%$** | $42.11\%$ | $41.24\%$ | **$58.76\%$** |

### Recommended Default Gate: Policy 2 (Balanced Gate, $\text{composite} \ge 0.0$)
* **Action for Person 2**: Routes $6,643$ queries ($62.91\%$) to the LLM reranker.
  * Within these routed pools, Top-10 accuracy is **$77.18\%$** (up $+14.53$ percentage points from baseline).
  * Top-5 accuracy is **$70.47\%$** (up $+16.33$ percentage points).
  * Rank-1 accuracy is **$53.17\%$** (up $+15.59$ percentage points).
* **Action for Person 3**: Diverts $3,917$ queries ($37.09\%$) to current-market retrieval.
  * Among diverted queries, **$61.99\%$** ($2,428$ queries) have **zero** valid candidates in the entire Top-10.

---

## 5. Architectural Rationale: Routing Logic

### 5.1 Why Send Strong Historical Evidence (`HISTORICAL_SUFFICIENT`) to Person 2?
1. **Preservation of Empirical Commercial Grounding**: Historical public sector purchase orders represent legally awarded, audit-verified pricing reflecting negotiated agency discounts, volume tiers, and freight realities that generic web searches cannot replicate.
2. **High Reachability Ceiling**: In the passed candidate pools, over $77\%$ of queries have a price within $\pm 10\%$ present in the Top-10.
3. **Closed-Context Invariant Enforcement**: Person 2's LLM operates under the non-negotiable rule that it *cannot invent prices* and *must select the exact price of one supplied candidate*. Providing the LLM with a pool where an accurate price already exists enables language comprehension to resolve subtle model numbers, kit configurations, and scope distinctions effectively.

### 5.2 Why Send Insufficient Evidence (`HISTORICAL_INSUFFICIENT`) to Person 3?
1. **Preventing Guaranteed Failure**: When candidate scores are weak and price dispersion is extreme ($\sigma_{\ln(P)} > 1.0$), the tabular retriever has failed to find comparable goods. In $62\%$ to $71\%$ of these cases, **no candidate in the Top-10 is within $\pm 10\%$**.
2. **Eliminating Forced LLM Errors**: Because Person 2 is strictly forbidden from hallucinating or interpolating prices, presenting Person 2 with an unrecoverable candidate pool **guarantees an erroneous benchmark prediction**.
3. **Current-Market Retrieval as the Correct Recovery Path**: For items that are rare, obsolete, or poorly matched in historical PO records, real-time market discovery (Person 3) via current vendor catalog lookups or live price indices is the only valid mechanism to establish an accurate procurement benchmark.

---

## 6. Implementation Specifications for Downstream Integration

When the Historical Evidence Gate is eventually integrated into the inference pipeline, it should be implemented as a pure Python decision rule requiring zero additional model dependencies:

```python
import numpy as np

def evaluate_historical_evidence_gate(
    candidate_scores: list[float],
    candidate_unit_prices: list[float],
    threshold: float = 0.0
) -> dict:
    """
    Inference-safe Historical Evidence Gate.
    
    Args:
        candidate_scores: Ranked candidate scores from Step-37 model (len >= 1).
        candidate_unit_prices: Corresponding candidate unit prices (len == len(scores)).
        threshold: Decision threshold for composite evidence score (default: 0.0).
        
    Returns:
        dict containing decision ('HISTORICAL_SUFFICIENT' or 'HISTORICAL_INSUFFICIENT'),
        composite score, score_r1, and log_price_std.
    """
    score_r1 = float(candidate_scores[0])
    
    # Calculate price dispersion across Top-K (clamped for numerical safety)
    prices = np.maximum(np.array(candidate_unit_prices, dtype=np.float64), 1e-4)
    log_price_std = float(np.std(np.log(prices)))
    
    composite_score = score_r1 - log_price_std
    is_sufficient = bool(composite_score >= threshold)
    
    return {
        "status": "HISTORICAL_SUFFICIENT" if is_sufficient else "HISTORICAL_INSUFFICIENT",
        "composite_evidence_score": round(composite_score, 4),
        "candidate_score_rank_1": round(score_r1, 4),
        "log_price_dispersion_std": round(log_price_std, 4),
        "routed_to": "Person_2_LLM_Reranker" if is_sufficient else "Person_3_Market_Evidence"
    }
```

---

## 7. Limitations & Scope Constraints

1. **Uniform Candidate Pool Size**: The frozen evaluation artifact contains exactly 10 candidates per validation query. While real-time retrieval may return fewer than 10 candidates if the catalog is sparse, the mathematical formulation ($\sigma_{\ln(P)}$ and $\text{score}_{r=1}$) remains valid for any $K \ge 2$.
2. **Commodity-Specific Variations**: While commodity family variance exists (e.g., Transformers exhibit $81.7\%$ Top-10 accuracy vs. Pumps at $36.0\%$), candidate score and price dispersion naturally capture this difficulty without needing hard-coded commodity-level rules.
3. **Decoupled Architecture**: This analysis is strictly research-only and has **not** modified the frozen production package `ML/procurement_benchmark_engine/` or its FastAPI service.

---

## 8. Threshold Validation Review (Policy 2: Composite Evidence $\ge 0.0$)

A targeted validation review was performed on the proposed balanced routing rule:
$$\text{composite\_evidence} = \text{candidate\_score}_{r=1} - \sigma_{\ln(P)}$$
$$\text{Routing: } \begin{cases} \text{HISTORICAL\_SUFFICIENT} \to \text{Person 2 (LLM Semantic Reranker)} & \text{if } \text{composite\_evidence} \ge 0.0 \\ \text{HISTORICAL\_INSUFFICIENT} \to \text{Person 3 (Current-Market Evidence)} & \text{if } \text{composite\_evidence} < 0.0 \end{cases}$$

All statistics below are recalculated directly from `frozen_evaluation_dataset.parquet` and serialized in [`evidence_gate_threshold_validation.json`](./evidence_gate_threshold_validation.json).

### 8.1 Recalculated Policy 2 Verification
* **Total Validation Queries ($N$)**: $10,560$
* **Accepted Queries (`HISTORICAL_SUFFICIENT`)**: **$6,643$** ($62.91\%$ pass rate)
* **Rejected Queries (`HISTORICAL_INSUFFICIENT`)**: **$3,917$** ($37.09\%$ rejection rate)
* **Accepted Pool Accuracies**:
  * **Top-10 Accuracy**: **$77.18\%$** ($5,127$ valid queries vs. $62.65\%$ baseline, **$+14.53\%$ boost**)
  * **Top-5 Accuracy**: **$70.47\%$** ($4,681$ valid queries vs. $54.14\%$ baseline, **$+16.33\%$ boost**)
  * **Rank-1 Accuracy**: **$53.17\%$** ($3,532$ valid queries vs. $37.58\%$ baseline, **$+15.59\%$ boost**)
* **Rejected Pool Profile**:
  * **Top-10 Accuracy**: **$38.01\%$** ($1,489$ queries)
  * **Completely Unrecoverable**: **$61.99\%$** ($2,428$ queries have **zero** candidates within $\pm 10\%$)

### 8.2 Sensitivity Analysis across Neighborhood $[-0.2, +0.2]$

To verify that threshold $0.0$ does not sit on an unstable cliff edge or numerical boundary, a sensitivity sweep was evaluated across $\Delta = \pm 0.1$ and $\pm 0.2$:

| Threshold | Pass Count | Pass Rate | Accepted Top-10 Acc | Accepted Top-5 Acc | Accepted Rank-1 Acc | Rejected Top-10 Acc | Rejected Unrecoverable % |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$-0.2$** | $7,163$ | $67.83\%$ | $75.62\%$ | $68.60\%$ | $50.87\%$ | $35.30\%$ | $64.70\%$ |
| **$-0.1$** | $6,889$ | $65.24\%$ | $76.50\%$ | $69.68\%$ | $52.13\%$ | $36.67\%$ | $63.33\%$ |
| **$0.0$ (Proposed)** | **$6,643$** | **$62.91\%$** | **$77.18\%$** | **$70.47\%$** | **$53.17\%$** | **$38.01\%$** | **$61.99\%$** |
| **$+0.1$** | $6,355$ | $60.18\%$ | $78.03\%$ | $71.46\%$ | $54.32\%$ | $39.41\%$ | $60.59\%$ |
| **$+0.2$** | $6,089$ | $57.66\%$ | $78.72\%$ | $72.31\%$ | $55.44\%$ | $40.77\%$ | $59.23\%$ |

**Observations on Sensitivity**:
1. **Linear Monotonicity**: Pass rates shift smoothly by approximately $2.5\%$ per $0.1$ increment without non-linear jumps.
2. **Stable Precision Gain**: Accepted Top-10 accuracy scales monotonically from $75.62\%$ to $78.72\%$ ($\approx 0.8\%$ per $0.1$ threshold step).
3. **Robust Rejection Purity**: The rejected pool consistently isolates $59.2\%$ to $64.7\%$ unrecoverable queries.
4. **Natural Zero-Centered Anchor**: At threshold $0.0$, the decision rule translates directly to:
   $$\text{candidate\_score}_{r=1} \ge \sigma_{\ln(P)}$$
   The model selects historical routing when the ranker's confidence on the primary match exceeds or equals the natural log price dispersion across the candidate pool.

### 8.3 Inference Safety Confirmation
* **Features Used by Gate**:
  1. `candidate_score` of Rank 1 ($s_1 \in \mathbb{R}$)
  2. `candidate_unit_price` of Top-10 candidates ($P_1, \dots, P_{10} \in \mathbb{R}^+$)
* **Prohibited Ground Truth Fields**:
  - `actual_unit_price`: **NOT USED**
  - `is_within_10pct`: **NOT USED**
  - `absolute_percentage_error`: **NOT USED**
* **Verification Status**: Fully inference-safe and compliant with zero data-leakage requirements.

### 8.4 Final Recommendation & Determination

> [!IMPORTANT]
> **Threshold Determination: ACCEPTED AS RESEARCH BASELINE**  
> Threshold **$0.0$** is formally accepted as the research baseline for the Step 1 Historical Evidence Gate handoff.  
> - It delivers a substantial **$+14.53\%$ quality boost** to Person 2 ($77.18\%$ Top-10 accuracy in accepted pools).  
> - It retains **$77.49\%$ of all recoverable queries** ($5,127$ of $6,616$).  
> - It correctly filters out **$2,428$ completely unrecoverable queries** to Person 3.  
> - Its smooth behavior across the $[-0.2, +0.2]$ neighborhood proves it is free of numerical cliff edges.
