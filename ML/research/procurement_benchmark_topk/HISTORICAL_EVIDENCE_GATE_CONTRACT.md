# Historical Evidence Gate Contract & Interface Specification (Step 2)

**Document Version:** 1.0 (Frozen Research Baseline)  
**Upstream System:** CivicEngage Frozen Step-37 / T3 Candidate Retrieval & Ranking Pipeline  
**Downstream Systems:**  
- **Person 2:** LLM Semantic Reranking Pipeline  
- **Person 3:** Current-Market Evidence & Vendor Catalog Retrieval Pipeline  
**Location:** `D:\CIVIC-ENGAGE\ML\research\procurement_benchmark_topk\HISTORICAL_EVIDENCE_GATE_CONTRACT.md`  
**Underlying Evaluated Artifact:** [`frozen_evaluation_dataset.parquet`](./frozen_evaluation_dataset.parquet) ($N = 10,560$ validation queries)  
**Validation Evidence:** [`evidence_gate_metrics.json`](./evidence_gate_metrics.json) & [`evidence_gate_threshold_validation.json`](./evidence_gate_threshold_validation.json)  

---

## 1. System Role & Routing Architecture

The **Historical Evidence Gate** is a deterministic, inference-safe statistical filter positioned between the upstream tabular LambdaMART ranker (Step-37) and downstream evidence processing.

```
                           [ Target Requisition Query ]
                                        │
                                        ▼
                         [ Step-37 Candidate Retrieval ]
                                        │
                                        ▼
                         [ Top-10 Ranked Candidates ]
                                        │
                                        ▼
                     ┌──────────────────────────────────────┐
                     │       HISTORICAL EVIDENCE GATE       │
                     │ composite = score_r1 - std(ln(P_top10))│
                     └──────────────────┬───────────────────┘
                                        │
                      ┌─────────────────┴─────────────────┐
                      │ composite >= 0.0                  │ composite < 0.0 OR K == 0
                      ▼                                   ▼
          ┌─────────────────────────┐         ┌─────────────────────────┐
          │  HISTORICAL_SUFFICIENT  │         │ HISTORICAL_INSUFFICIENT │
          └───────────┬─────────────┘         └───────────┬─────────────┘
                      │                                   │ (Bypass Person 2)
                      ▼                                   │
          ┌─────────────────────────┐                     │
          │   Person 2: LLM         │                     │
          │   Semantic Reranker     │                     │
          └───────────┬─────────────┘                     │
                      │                                   │
            ┌─────────┴─────────┐                         │
            │                   │                         │
            ▼                   ▼                         │
   HISTORICAL_ACCEPTED   HISTORICAL_INSUFFICIENT          │
            │                   │                         │
            ▼                   └───────────┬─────────────┘
   [ Historical Benchmark                   ▼
     Price Resolved ]             ┌─────────────────────────┐
                                  │   Person 3: Current-    │
                                  │   Market Evidence       │
                                  └─────────────────────────┘
```

---

## 2. Gate Input Specification

For each incoming requisition query, the upstream retrieval engine provides the Historical Evidence Gate with:
1. **Target Query Attributes** (verified query fields from `LLM_RERANKING_INPUT_SPEC.md`).
2. **Top-10 Historical Candidates** retrieved from the historical catalog.

### 2.1 Target Query Schema (`target_query`)
```json
{
  "query_index": 8,
  "target_description": "SULFUR DIOXIDE Liquid Sulfur Dioxide in one ton container for SAR JAN - MAR 2023",
  "target_quantity": 4.0,
  "target_unit_of_measure": "TON",
  "target_procurement_date": "2023-01-03",
  "target_city": "AUSTIN",
  "target_state": "TX",
  "target_brand": null,
  "target_model": null,
  "target_commodity_code": "88578",
  "target_commodity_family": "CHEMICALS_GASES"
}
```

### 2.2 Candidate Pool Schema (`candidates`)
An ordered array of $K$ candidate objects (where $K = 10$ in standard operation, or $0 \le K < 10$ in edge cases):
```json
[
  {
    "candidate_rank": 1,
    "candidate_score": 3.199756,
    "candidate_unit_price": 961.51,
    "candidate_description": "SULFUR DIOXIDE Liquid Sulfur Dioxide in one ton container for SAR JAN - MAR 2023"
  },
  {
    "candidate_rank": 2,
    "candidate_score": 2.580145,
    "candidate_unit_price": 980.00,
    "candidate_description": "SULFUR DIOXIDE Liquid Sulfur Dioxide in one ton cylinder container"
  },
  "... up to candidate_rank 10 ..."
]
```

### 2.3 Strict Ground-Truth Prohibition
> [!CRITICAL]
> The fields `actual_unit_price`, `is_within_10pct`, and `absolute_percentage_error` are **strictly evaluation ground truth**. They must **never** be supplied to the Historical Evidence Gate or to Person 2 during real-time inference.

---

## 3. Gate Calculation & Mathematical Formulation

The gate computes two inference-safe signals directly from the supplied candidates:

1. **Rank-1 Match Confidence ($s_1$)**:
   The ensemble relevance score of the primary ranked candidate:
   $$s_1 = \text{candidates}[0].\text{candidate\_score}$$

2. **Candidate Price Dispersion ($\sigma_{\ln(P)}$)**:
   The standard deviation of natural log unit prices across the Top-10 candidates (clamped at $10^{-4}$ for numerical stability):
   $$\sigma_{\ln(P)} = \sqrt{\frac{1}{K} \sum_{i=1}^{K} \left( \ln(P_i) - \overline{\ln(P)} \right)^2}, \quad P_i = \max(\text{candidate\_unit\_price}_i, 10^{-4})$$

3. **Composite Evidence Score**:
   $$\text{composite\_evidence} = s_1 - \sigma_{\ln(P)}$$

---

## 4. Decision Logic & Routing Boundary

```python
if len(candidates) == 0:
    gate_decision = "HISTORICAL_INSUFFICIENT"
    route_to = "Person_3_Current_Market_Evidence"
elif composite_evidence >= 0.0:
    gate_decision = "HISTORICAL_SUFFICIENT"
    route_to = "Person_2_LLM_Semantic_Reranker"
else:
    gate_decision = "HISTORICAL_INSUFFICIENT"
    route_to = "Person_3_Current_Market_Evidence"
```

### Routing Rules:
* **`HISTORICAL_SUFFICIENT` ($\text{composite\_evidence} \ge 0.0$)**:
  * Forward the target query and Top-10 candidates to **Person 2**.
  * Empirically, this pool has a **$77.18\%$ Top-10 accuracy** and **$70.47\%$ Top-5 accuracy**.
* **`HISTORICAL_INSUFFICIENT` ($\text{composite\_evidence} < 0.0$ or $K = 0$)**:
  * **Bypass Person 2 entirely** (zero LLM token invocation).
  * Route the requisition directly to **Person 3** (Current-Market Evidence).
  * Empirically, **$61.99\%$** of queries in this rejected pool contain zero valid historical candidates.

---

## 5. Person 2 Interface Contract

When Person 2 receives a `HISTORICAL_SUFFICIENT` payload, Person 2 must evaluate the candidates through closed-context semantic specification matching and return one of two strict JSON structures.

### 5.1 Success Contract: `HISTORICAL_ACCEPTED`
Used when Person 2 identifies a candidate that matches the target specification, scope, unit of measure, and commercial context:

```json
{
  "decision": "HISTORICAL_ACCEPTED",
  "selected_candidate_rank": 1,
  "reason": "Candidate Rank 1 is an exact specification match for Liquid Sulfur Dioxide in 1-ton containers with identical commercial delivery terms."
}
```

#### Field Specifications:
* `decision` (`str`): Must be exactly `"HISTORICAL_ACCEPTED"`.
* `selected_candidate_rank` (`int`): Must be an integer between `1` and `10` corresponding to one of the provided `candidate_rank` values.
* `reason` (`str`): Short evidence-based textual justification citing matching or distinguishing features.

### 5.2 Rejection Contract: `HISTORICAL_INSUFFICIENT`
Used when Person 2 determines that **all 10 candidates** exhibit critical specification mismatches (e.g., mismatched hardware generation, incompatible power ratings, bare tool vs. full kit, license vs. physical equipment):

```json
{
  "decision": "HISTORICAL_INSUFFICIENT",
  "selected_candidate_rank": null,
  "reason": "All 10 candidates exhibit critical specification mismatches: candidates 1-6 describe handheld sampling wands without housing, while candidates 7-10 describe recurring lab service fees rather than hardware."
}
```

#### Field Specifications:
* `decision` (`str`): Must be exactly `"HISTORICAL_INSUFFICIENT"`.
* `selected_candidate_rank` (`null`): Must be literal `null`.
* `reason` (`str`): Short explanation detailing why none of the historical candidates are acceptable.

---

## 6. Strict No-Price-Invention Rule

> [!WARNING]
> ### Core System Invariant: Zero Price Hallucination
> 1. Person 2 is **strictly an evidence-based selector**, never a price generator.
> 2. Person 2 **must never** invent, estimate, interpolate, average, inflation-adjust, or alter a price.
> 3. Person 2 **must never** output a price field in its response.
> 4. The final historical unit price is **strictly resolved by the pipeline**:
>    $$P_{\text{final}} = \text{candidates}[\text{selected\_candidate\_rank} - 1].\text{candidate\_unit\_price}$$
> 5. Any price not identical to one of the supplied candidate prices is invalid and rejected.

---

## 7. Invalid LLM Output Handling & Fallback Protocols

To safeguard production pipelines from non-deterministic LLM failures, the system validates Person 2's response against strict defensive criteria:

| Failure Mode | Definition | Pipeline Action | Final Routing |
| :--- | :--- | :--- | :--- |
| **Invalid Rank** | `selected_candidate_rank` not in $\{1, \dots, 10\}$ (e.g. `0`, `15`, `"first"`, `null` with `ACCEPTED`). | Invalidate LLM output; log schema error. | Fallback to **Person 3** |
| **Missing Decision** | `decision` field absent or not in `["HISTORICAL_ACCEPTED", "HISTORICAL_INSUFFICIENT"]`. | Invalidate LLM output; log parse error. | Fallback to **Person 3** |
| **Fabricated Price** | Response contains an invented price key (e.g., `"unit_price": 500.0`, `"adjusted_price": 950.0`). | Reject fabricated price immediately. | Fallback to **Person 3** |
| **Malformed JSON** | Output is non-parseable JSON (unclosed braces, trailing text, markdown codefence corruption). | Invalidate response; log syntax error. | Fallback to **Person 3** |
| **Contradictory State** | `decision == "HISTORICAL_INSUFFICIENT"` but `selected_candidate_rank` is an integer. | Coerce to rejection or fail schema validation. | Route to **Person 3** |

> **Fail-Safe Principle:** Any contract breach by Person 2 is treated as a failure of historical semantic reranking. The query immediately falls back to Person 3 current-market evidence. The pipeline **never** guesses an unverified price.

---

## 8. Post-Person-2 Routing Summary

| Gate Decision | Person 2 Output | Downstream Action | Final Benchmark Price Source |
| :--- | :--- | :--- | :--- |
| `HISTORICAL_SUFFICIENT` | `HISTORICAL_ACCEPTED` ($r \in [1, 10]$) | Accept historical candidate $r$. | `candidate_unit_price` of candidate $r$. |
| `HISTORICAL_SUFFICIENT` | `HISTORICAL_INSUFFICIENT` ($r = \text{null}$) | Route to Person 3. | Person 3 Current-Market Evidence. |
| `HISTORICAL_SUFFICIENT` | *Malformed / Invalid Output* | Invalidate & route to Person 3. | Person 3 Current-Market Evidence. |
| `HISTORICAL_INSUFFICIENT` | *(Bypassed — Person 2 not called)* | Route to Person 3 directly. | Person 3 Current-Market Evidence. |

---

## 9. Distinguishing Pre-LLM ML Gate vs. Post-LLM Semantic Gate

It is critical to distinguish between the two sequential filtering stages:

| Dimension | Stage 1: Pre-LLM ML Evidence Gate | Stage 2: Post-LLM Semantic Filter (Person 2) |
| :--- | :--- | :--- |
| **Execution Point** | Immediately after candidate retrieval. | After candidate pool passes the ML gate. |
| **Mechanism** | Tabular statistical computation ($s_1 - \sigma_{\ln(P)}$). | Deep LLM natural language comprehension. |
| **Evaluation Scope** | Candidate scores and intra-pool price dispersion. | Specification nuance, accessories, scope, packaging. |
| **Latency & Cost** | Sub-millisecond ($< 0.1\text{ ms}$), zero API cost. | $500\text{–}1500\text{ ms}$, standard LLM token cost. |
| **Primary Failure Prevented** | Filters out unrecoverable pools ($62\%$ junk pools). | Catches subtle text mismatches inside high-scoring pools. |

---

## 10. Complete End-to-End Walkthrough Examples

> **Note on Examples:** The following scenarios are provided as illustrative walkthrough examples to demonstrate deterministic contract interactions, payload schemas, and routing boundaries across the pipeline.

### Example 1: Accepted Historical Evidence
* **Target Query:** Liquid Sulfur Dioxide requisition (`query_index = 8`).
* **Candidate Pool:**
  * Rank 1: `$961.51` (Score: `3.200`) — Liquid Sulfur Dioxide 1-ton container
  * Rank 2: `$980.00` (Score: `2.580`) — Liquid Sulfur Dioxide 1-ton cylinder
  * Ranks 3–10: Prices ranging from `$950.00` to `$1,020.00`.
* **Gate Computation:**
  $$s_1 = 3.200, \quad \sigma_{\ln(P)} = 0.096 \implies \text{composite} = 3.200 - 0.096 = +3.104$$
  Since $+3.104 \ge 0.0 \implies \mathbf{HISTORICAL\_SUFFICIENT}$.
* **Person 2 Invocation:** Pool sent to Person 2.
* **Person 2 Response:**
  ```json
  {
    "decision": "HISTORICAL_ACCEPTED",
    "selected_candidate_rank": 1,
    "reason": "Candidate Rank 1 is an exact chemical specification and packaging match (1-ton container liquid sulfur dioxide)."
  }
  ```
* **Pipeline Resolution:** Benchmark price finalized at **`$961.51`**.

---

### Example 2: Rejected Historical Evidence (Pre-LLM Gate Bypass)
* **Target Query:** Portable Sampler Maintenance Evaluation Fee (`query_index = 1`).
* **Candidate Pool:**
  * Rank 1: `$22.00` (Score: `0.884`) — Sample Collecting wand tip
  * Rank 2: `$2,724.00` (Score: `0.720`) — Automated wastewater sampler unit
  * Rank 3: `$312.00` (Score: `0.510`) — Calibration assembly
  * Ranks 4–10: Prices spanning `$65.00` to `$3,970.00`.
* **Gate Computation:**
  $$s_1 = 0.884, \quad \sigma_{\ln(P)} = 2.255 \implies \text{composite} = 0.884 - 2.255 = -1.371$$
  Since $-1.371 < 0.0 \implies \mathbf{HISTORICAL\_INSUFFICIENT}$.
* **Person 2 Invocation:** **BYPASSED (Person 2 is NOT called).**
* **Pipeline Resolution:** Requisition routed directly to **Person 3** for live market catalog search on sampler maintenance fees.

---

### Example 3: Zero Historical Candidates
* **Target Query:** Specialized custom telemetry buoy array (`query_index = 99999`).
* **Candidate Pool:** `[]` ($K = 0$).
* **Gate Computation:**
  $$\text{Candidate count } K = 0 \implies \mathbf{HISTORICAL\_INSUFFICIENT}$$
* **Person 2 Invocation:** **BYPASSED.**
* **Pipeline Resolution:** Requisition routed directly to **Person 3** for current-market evidence retrieval.
