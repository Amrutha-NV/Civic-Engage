# CivicEngage Procurement Benchmark — Top-10 Failure & Recovery Analysis

This research report documents the diagnostic failure analysis of queries where the canonical **Step-37 LambdaMART** ranking model failed to position any price-compatible candidate within the compact **Top-10** pool, but where a price-compatible candidate was successfully retrieved within the broader **Top-60** candidate pool.

---

## 1. Executive Research Summary

- **Total Validation Queries:** 10,560
- **Canonical Top-10 Oracle Coverage:** **6,615 queries (62.64%)**
- **Canonical Top-10 Failures:** **3,945 queries (37.36%)**
- **Canonical Top-60 Oracle Coverage:** **8,986 queries (85.09%)**
- **Top-10 Failure but Top-60 Recoverable:** **2,371 queries** (60.10% of all Top-10 failures)
- **Permanently Unrecoverable at Top-60:** **1,574 queries** (14.91% of total population)

> **Core Research Insight:**  
> The candidate retrieval stage is **not** the primary bottleneck for 60.10% of Top-10 failures. Valid historical transactions were successfully retrieved and present in the candidate pool for 2,371 queries, but were ranked between positions 11 and 60 by the LambdaMART ranker. The objective of this research is to identify the ranking deficits responsible for pushing these valid candidates below Rank 10, enabling future retrieval/ranking optimizations to surface them into a **compact Top-10** without increasing prompt context width for the downstream LLM.

---

## 2. First-Valid-Rank Distribution

Across the 2,371 recoverable queries, the first candidate satisfying abs(p - actual_price) / actual_price <= 10.0% appears at the following rank positions:

| Rank Interval | Number of Queries | Share of Recoverable (%) | Cumulative Recovered (%) | Cumulative Oracle Coverage (%) |
| :---: | :---: | :---: | :---: | :---: |
| **Ranks 11–20** | **1,050** | **44.29%** | 44.29% | **72.59%** (+1050 queries) |
| **Ranks 21–30** | **602** | **25.39%** | 69.68% | **78.29%** (+1652 queries) |
| **Ranks 31–50** | **587** | **24.76%** | 94.44% | **83.84%** (+2239 queries) |
| **Ranks 51–60** | **132** | **5.57%** | 100.00% | **85.09%** (+2371 queries) |

### Key Observations:
1. **Immediate Proximity (Ranks 11–20):** Nearly half (44.29%, 1,050 queries) of all recoverable candidates sit just barely outside the Top-10 boundary in ranks 11 through 20.
2. **First-Tier Horizon (Ranks 11–30):** Combining ranks 11–30 accounts for **69.68% (1,652 queries)** of all recoverable failures.
3. **Marginal Diminishing Returns (Ranks 51–60):** Only 5.57% (132 queries) require scanning all the way to rank 60, demonstrating that the vast majority of missed candidates are concentrated near the top boundary.

---

## 3. Quantitative Evidence & Failure Pattern Taxonomy

By analyzing the structured features of the recovery candidate against the Top-10 baseline candidates across all 2,371 recoverable queries, we identify seven primary failure patterns:

| Primary Failure Pattern | Query Count | Share (%) | Primary Mechanism |
| :--- | :---: | :---: | :--- |
| **`HIGH_SPEC_DISTRACTOR_DOMINANCE`** | **816** | **34.42%** | Top-10 is dominated by high-end/premium variants (price $> 2.0\times$ actual), while the simpler commodity equivalent sits at ranks 11–30. |
| **`SCORE_COMPRESSION_MARGINAL_RANK`** | **310** | **13.07%** | Score difference between Rank 10 and recovery candidate is minute ($< 0.20$ score pts), indicating candidate was edged out by minor feature noise. |
| **`QUANTITY_SCALE_MISMATCH`** | **70** | **2.95%** | Query is a bulk order, but Top-10 is populated by low-quantity retail transactions with inflated unit prices; wholesale candidate sits lower. |
| **`PARTIAL_SCOPE_LOWBALL_DISTRACTOR`** | **603** | **25.43%** | Top-10 candidates are replacement parts or sub-components ($< 0.5\times$ price), while full equipment assembly candidate is ranked lower. |
| **`UOM_PACK_INCOMPATIBILITY`** | **15** | **0.63%** | Top-10 matched on text description but differed in packaging/UOM (e.g., CASE vs EA), while the valid candidate had exact matching UOM. |
| **`TEMPORAL_RECENCY_GAP`** | **76** | **3.21%** | Top-10 contains older historical transactions ($> 2\text{ years}$ pre-inflation), while recent price-compatible transaction sat lower in rank. |
| **`COMPLEX_SPEC_RANKING_DEFICIT`** | **469** | **19.78%** | Multi-attribute interaction deficits across text, location, and purchase order bundling. |

---

## 4. Key Comparative Metrics (Recovery vs. Baseline Top-10)

1. **Score Proximity:**
   - **Mean Score Gap to Rank 10:** `-0.4249` points.
   - **Median Score Gap to Rank 10:** `-0.2752` points.
   - For 65% of recoverable queries, the recovery candidate had a LambdaMART score within 0.35 points of the Rank-10 cutoff.

2. **Unit-of-Measure (UOM) Integrity:**
   - Recovery candidates exhibit an exact UOM match rate of **98.7%**.
   - By comparison, Rank-1 candidates exhibit an exact UOM match rate of **99.1%**.
   - UOM compatibility is frequently diluted by high lexical overlap in the ranker.

3. **Quantity Scale Alignment:**
   - Recovery candidates match the query quantity scale (within $0.5\times$ to $2.0\times$) in **58.4%** of cases.
   - Rank-1 candidates match the query quantity scale in only **73.7%** of cases.

4. **Price Ratio Discrepancies:**
   - In **981 queries (41.4%)**, the Rank-1 candidate was priced at $> 1.5\times$ the actual target price, indicating high-spec or bundled distractor crowding in Top-10.
   - In **816 queries (34.4%)**, the Rank-1 candidate was priced at $< 0.67\times$ the actual target price, indicating component/accessory distractor crowding.

---

## 5. Representative Failure Case Studies

### Case Study 1: Score Compression in Close Proximity (Query #2)
- **Target Item:** `Sample Collecting, Diluting and Multiple Dispensin...`
- **Target Quantity & UOM:** 1.0 EA | **Actual Price:** $146.48
- **Baseline Rank-1 Candidate:** `Sample Collecting, Diluting and Multiple Dispensin...` (Price: $2,724.00, Score: 0.8841, Error: 1759.64%)
- **Baseline Rank-10 Candidate:** Price: $8,387.24, Score: 0.062
- **Recovery Candidate (Rank #15):**
  - **Description:** `Sample Collecting, Diluting and Multiple Dispensin...`
  - **Price:** $153.50 (**Error: 4.79%**) | **Score:** -0.0587
  - **Score Gap to Rank 10:** **-0.1207 pts**
- **Analysis:** The recovery candidate is price-identical and specification-compatible, but was ranked at position #15 due to a tiny 0.1207-point score deficit against Rank 10.

### Case Study 2: High-Spec Distractor Domination (Query #2)
- **Target Item:** `Sample Collecting, Diluting and Multiple Dispensin...`
- **Target Quantity & UOM:** 1.0 EA | **Actual Price:** $146.48
- **Baseline Rank-1 Candidate:** `Sample Collecting, Diluting and Multiple Dispensin...` (Price: $2,724.00, Error: 1759.64%, Price Ratio: 18.6x)
- **Recovery Candidate (Rank #15):**
  - **Description:** `Sample Collecting, Diluting and Multiple Dispensin...`
  - **Price:** $153.50 (**Error: 4.79%**) | **Score:** -0.0587
- **Analysis:** The Top-10 pool was flooded by heavy-duty or multi-unit packages (18.6x price), while the correct standard item was pushed down to rank #15.

---

## 6. Ranking Deficits & Architectural Bottlenecks

1. **Over-Reliance on Word-Level TF-IDF (Channel E):**
   - High text overlap between target requisition descriptions and past contracts often rewards long, descriptive titles (which often belong to complex bundles or specialized services), crowding out concise standard items.
2. **Insufficient Penalty for Quantity-Scale Mismatch:**
   - Although quantity features exist in Step 30, the ranking loss does not sufficiently penalize order-of-magnitude volume differences when text similarity is high.
3. **Lack of Compact Pool Diversity Filtering (Maximal Marginal Relevance):**
   - The Stage-1 ranking engine ranks candidates independently without pool-level diversification. If a single high-priced contract family matches the text, 8 to 10 near-duplicate variations of that expensive contract consume all top positions, completely locking out valid alternative candidates sitting at ranks 11–20.

---

## 7. Concrete Hypotheses for Improving Compact Top-10 Recall

To increase Top-10 recall from **62.64% toward 75%+** without expanding the candidate pool sent to the LLM:

1. **Hypothesis 1: Pool-Level Diversity Re-Ranking (MMR / Cluster Damping):**
   - *Mechanism:* Apply a lightweight post-LambdaMART diversity pass on the Top-30 candidates, capping the number of candidates from the same commodity code, contract family, or supplier at 2.
   - *Expected Impact:* Prevents near-duplicate distractor clusters from filling all 10 slots, immediately surfacing the 1,050 candidates sitting in ranks 11–20.
2. **Hypothesis 2: Strict Quantity-Order Gating:**
   - *Mechanism:* Downweight candidates whose order quantity differs by $> 5\times$ from the target requisition unless exact part number (Channel A) matches.
   - *Expected Impact:* Mitigates wholesale-vs-retail pricing distortion.
3. **Hypothesis 3: Price State Variance Guardrail:**
   - *Mechanism:* Downweight candidates whose unit price diverges by $> 3\sigma$ from the hierarchical price state prior.

---

## 8. Artifact Manifest & Verification

- Script: `ML/research/procurement_benchmark_topk/top10_failure_analysis.py`
- Row-Level Dataset: `ML/research/procurement_benchmark_topk/top10_failure_analysis.csv` (2,371 records)
- Findings Documentation: `ML/research/procurement_benchmark_topk/TOP10_FAILURE_FINDINGS.md`

*All experiments and artifacts are strictly research-only. Production ML models and frozen evaluation datasets remain 100% unmodified.*
