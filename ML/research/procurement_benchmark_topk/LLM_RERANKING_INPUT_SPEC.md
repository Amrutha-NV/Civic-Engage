# LLM Semantic Reranker Specification: Candidate Input & Output Schema

**Target Audience:** Person 2 (LLM Semantic Reranking Pipeline & Downstream Research)  
**Upstream System:** CivicEngage Frozen Step-37 / T3 Co-Occurrence Model  
**Upstream Data Source:** `ML/research/procurement_benchmark_topk/frozen_evaluation_dataset.parquet`  
**Auxiliary Analysis Artifacts:** `top_10_candidates.csv`, `top_k_metrics.json`, `recovery_rank_distribution.json`  

---

## 1. System Role & Core Invariant

The upstream tabular LambdaMART ranker retrieves and scores procurement candidates across 176 structured and PO-context features. While its Rank-1 accuracy is **37.57%**, **62.64%** of queries contain a valid historical price ($\pm 10\%$) within the **Top-10 candidates** (and **54.14%** within **Top-5**). Furthermore, **40.17%** of Rank-1 failures (2,648 queries) contain an accurate price in Ranks 2–10.

### The Non-Negotiable Invariant
> **The LLM is strictly an evidence-based semantic reranker, NOT a price generator.**  
> - The LLM **must NEVER** generate, interpolate, estimate, average, or invent unit prices.  
> - The LLM **must NEVER** reference external knowledge or training memory for market pricing.  
> - The final price prediction **MUST ALWAYS** be the exact `candidate_unit_price` of the candidate selected from the supplied Top-$K$ list.

---

## 2. Frozen Dataset Schema & Field Availability

The downstream evaluation artifact is **`frozen_evaluation_dataset.parquet`** ($105,600$ rows, $10,560$ validation queries $\times$ $10$ candidates).

Downstream researchers and Person 2 **must only use the fields actually present in this frozen artifact**.

### 2.1 Available Fields in `frozen_evaluation_dataset.parquet`

#### Target Query Fields (Supplied to LLM):
* `query_index` (`int64`): Unique index of the validation query ($0$ to $10,559$).
* `target_description` (`str`): Target line-item purchase requirement text.
* `target_quantity` (`float64`): Requisition volume/quantity.
* `target_unit_of_measure` (`str`): Requisition unit of measure (e.g., `EA`, `PKG`, `LOT`).
* `target_procurement_date` (`str`): Date of the target requisition (ISO format `YYYY-MM-DD`).
* `target_city` (`str`): Requisition municipality / agency city.
* `target_state` (`str`): Requisition jurisdiction / state.
* `target_brand` (`str`): Extracted target brand/manufacturer (if detected, else empty/null).
* `target_model` (`str`): Extracted target model or product line (if detected, else empty/null).
* `target_commodity_code` (`str`): Detailed commodity classification code.
* `target_commodity_family` (`str`): High-level commodity group classification.

#### Candidate Fields (Supplied to LLM):
* `candidate_rank` (`int64`): Upstream ML ranker rank position ($1$ through $10$).
* `candidate_description` (`str`): Historical catalog line item extended description.
* `candidate_unit_price` (`float64`): Historical unit price awarded for this candidate item.
* `candidate_score` (`float64`): Raw LambdaMART ensemble relevance score from Step-37 model.

#### Ground Truth & Benchmark Evaluation Fields (For Scoring, NOT Leaked to LLM Prompt):
* `actual_unit_price` (`float64`): The true ground truth unit price of the target item.
* `absolute_percentage_error` (`float64`): Percentage error of the candidate price relative to `actual_unit_price` ($|P_{cand} - P_{actual}| / P_{actual} \times 100$).
* `is_within_10pct` (`bool`): Ground truth indicator whether candidate price is within $\pm 10\%$ of `actual_unit_price`.

### 2.2 NOT AVAILABLE in the Current Frozen Artifact

The following fields are **NOT** present in `frozen_evaluation_dataset.parquet`:
* **Candidate quantity**
* **Candidate UOM**
* **Candidate award date**
* **Candidate vendor**
* **Candidate vendor location**
* **Candidate contract type**
* **Candidate extracted structured specifications**

> [!CRITICAL]
> **Do NOT invent, fabricate, or hallucinate these fields.**  
> Person 2 must use **only** the candidate fields actually supplied by the frozen artifact (`candidate_rank`, `candidate_description`, `candidate_unit_price`, `candidate_score`) unless additional candidate metadata is later provided as a separate verified artifact.

---

## 3. LLM Prompt Input Schema

For each procurement request, Person 2's pipeline formats a target query object and an array of 10 candidate objects ($K = 10$) constructed exclusively from the verified fields.

### 3.1 Target Query Input (`query`)

```json
{
  "query_index": 412,
  "target_description": "Dell Latitude 5530 15.6 Inch Laptop Intel Core i7-1265U 16GB RAM 512GB SSD Windows 11 Pro",
  "target_quantity": 25.0,
  "target_unit_of_measure": "EA",
  "target_procurement_date": "2024-03-15",
  "target_city": "CHICAGO",
  "target_state": "IL",
  "target_brand": "Dell",
  "target_model": "Latitude 5530",
  "target_commodity_code": "20453",
  "target_commodity_family": "COMPUTER_HARDWARE"
}
```

### 3.2 Candidate Input Object (`candidates[i]`)

```json
{
  "candidate_rank": 3,
  "candidate_score": 1.482019,
  "candidate_unit_price": 1189.50,
  "candidate_description": "Dell Latitude 5530 vPro Laptop 15.6 FHD i7-1265U 16GB DDR4 512GB PCIe NVMe SSD Win11 Pro"
}
```

---

## 4. Field Usage & Semantic Rationale

| Field | Availability | Role in Downstream Reranking |
| :--- | :--- | :--- |
| **`target_description` & `candidate_description`** | **Available** | **Primary semantic anchor**: Allows the LLM to identify subtle specification mismatches (e.g., bare tool vs. kit with battery, base trim vs. enterprise grade, accessories included) that tabular tokenizers miss. |
| **`candidate_unit_price`** | **Available** | **Candidate grounding**: The LLM needs visibility into candidate prices to evaluate whether candidate differences logically justify price dispersion across the pool. |
| **`candidate_rank` & `candidate_score`** | **Available** | **Statistical prior**: Gives the LLM visibility into the strong tabular prior; the LLM should require clear semantic justification before demoting Rank-1 in favor of a lower-ranked candidate. |
| **`target_quantity` & `target_unit_of_measure`** | **Available (Target Only)** | **Target context**: Provides the LLM with requisition scale and unit expectations. |
| **`target_brand` & `target_model`** | **Available (Target Only)** | **Direct identity matching**: Anchors target manufacturer and product lines to compare against candidate textual descriptions. |
| **`target_city` & `target_state`** | **Available (Target Only)** | **Geographic jurisdiction**: Indicates target municipality and state purchasing context. |
| **`target_commodity_code` & `family`** | **Available (Target Only)** | **Categorical context**: Anchors procurement taxonomy. |
| *Candidate metadata (vendor, date, quantity, UOM, contract)* | **NOT Available** | *Excluded from current frozen evaluation artifact. Must not be fabricated.* |

---

## 5. Canonical LLM Output Schema (Authoritative Contract Alignment)

> [!IMPORTANT]
> **Authoritative Contract Reference:**  
> The finalized [`HISTORICAL_EVIDENCE_GATE_CONTRACT.md`](./HISTORICAL_EVIDENCE_GATE_CONTRACT.md) defines the authoritative runtime output contract for Person 2. All downstream implementations, prompt templates, and evaluation harnesses must strictly adhere to the canonical schema defined below.

Person 2 evaluates the supplied candidates through closed-context semantic specification matching and must return a strict JSON response in one of two canonical forms:

### 5.1 Success Contract: `HISTORICAL_ACCEPTED`
Returned when Person 2 identifies a candidate that matches the target line-item specification, scope, unit of measure, and commercial context:

```json
{
  "decision": "HISTORICAL_ACCEPTED",
  "selected_candidate_rank": 1,
  "reason": "<short explanation>"
}
```

* `decision`: Must be exactly `"HISTORICAL_ACCEPTED"`.
* `selected_candidate_rank`: Must be an integer between `1` and `10` corresponding to one of the supplied candidate ranks.
* `reason`: Concise textual explanation citing textual specification match.

### 5.2 Rejection Contract: `HISTORICAL_INSUFFICIENT`
Returned when Person 2 determines that **all 10 candidates** exhibit critical specification mismatches (e.g., mismatched hardware generation, missing required accessories/kit items, incompatible voltage/ratings):

```json
{
  "decision": "HISTORICAL_INSUFFICIENT",
  "selected_candidate_rank": null,
  "reason": "<short explanation>"
}
```

* `decision`: Must be exactly `"HISTORICAL_INSUFFICIENT"`.
* `selected_candidate_rank`: Must be literal `null`.
* `reason`: Concise explanation detailing why all candidates are semantically unacceptable.

---

### 5.3 Non-Negotiable Output Rules & Price Resolution

1. **Rank Selection Only**: Person 2 selects **ONLY** a candidate rank from the supplied Top-10 ($1 \le \text{rank} \le 10$).
2. **No Price Output**: Person 2 **MUST NOT** output any price field (e.g., no `selected_historical_unit_price`, `unit_price`, or `price`).
3. **Strict No-Price-Invention Invariant**: Person 2 **MUST NOT** invent, modify, average, interpolate, or generate a price.
4. **Deterministic Pipeline Price Resolution**: The evaluation harness and production runtime deterministically resolve the historical price using:
   $$P_{\text{final}} = \text{candidate\_unit\_price}[\text{selected\_candidate\_rank}]$$
5. **Invalid Output / Price Fabrication Fallback**: Any output containing a price field or invalid rank is malformed and must follow the fallback protocol defined in [`HISTORICAL_EVIDENCE_GATE_CONTRACT.md`](./HISTORICAL_EVIDENCE_GATE_CONTRACT.md) (invalidation of historical candidate and failover to Person 3 current-market evidence).
6. **Authoritative Specification**: [`HISTORICAL_EVIDENCE_GATE_CONTRACT.md`](./HISTORICAL_EVIDENCE_GATE_CONTRACT.md) is the authoritative interface contract governing all Person 2 interactions.

---

### 5.4 Legacy Prototype Output Schema (DEPRECATED — NOT VALID FOR FINAL INTERFACE)

> [!CAUTION]
> **LEGACY ARCHIVE — NOT VALID FOR THE FINAL INTERFACE:**  
> Earlier research prototypes explored richer candidate reranking payloads containing `"selected_historical_unit_price"`, `"reranked_candidate_ranks"`, `"confidence"`, and `"evidence_reasoning"`. These prototype fields are **strictly deprecated and invalid** in the production contract because allowing the LLM to output a price field introduces price hallucination risk.
>
> ```json
> /* LEGACY PROTOTYPE — DO NOT USE IN PRODUCTION PROMPTS */
> {
>   "selected_candidate_rank": 3,
>   "selected_historical_unit_price": 1189.50,  /* DEPRECATED: LLM must not output price */
>   "reranked_candidate_ranks": [3, 1, 2, 5, 4],  /* DEPRECATED: Not in final contract */
>   "confidence": "HIGH",                         /* DEPRECATED: Not in final contract */
>   "evidence_reasoning": "..."                   /* DEPRECATED: Replaced by 'reason' */
> }
> ```

---

## 6. Pool Sizing Guidance ($K = 5$ vs. $K = 10$)

* **Canonical Research Setting: Top-10 Candidates ($K = 10$)**:
  * **Authoritative Benchmark Pipeline**: The canonical research evaluation setting is strictly Top-10 candidates ($K = 10$). The frozen evaluation dataset (`frozen_evaluation_dataset.parquet`), Evidence Gate, LLM reranking contract (`HISTORICAL_EVIDENCE_GATE_CONTRACT.md`), and evaluation protocol (`FROZEN_EVALUATION_PROTOCOL.md`) use the Top-10 candidate pool.
  * **Recovery Ceiling**: Captures an additional 899 recoverable queries (+13.64% of failure pool, +33.95% relative increase in recovered queries over Top-5), expanding the theoretical reachable ceiling to 62.64%.
* **Top-5 Window ($K = 5$) as Future Ablation Only**:
  * **Ablation Scope**: $K = 5$ is **NOT** the canonical research evaluation setting. $K = 5$ may only be treated as a future latency/ablation experiment.
  * **Coverage Context**: Based on `recovery_rank_distribution.json`, $K = 5$ covers 66.05% of recoverable failures (1,749 queries) and minimizes prompt token counts, but caps maximum reachable accuracy at 54.14%.
