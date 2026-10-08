# CivicEngage Procurement Benchmark Engine — Official Freeze Manifest

**Freeze Date:** October 7, 2026  
**Status:** **FROZEN & VERIFIED**  
**Component:** Person 1 (Tabular LightGBM Ranker, Conformal Uncertainty, & Selective MMR Candidate Selection)  
**Package Root:** `D:\CIVIC-ENGAGE\ML\procurement_benchmark_engine\`  
**Research Archive:** `D:\CIVIC-ENGAGE\ML\research\procurement_benchmark_topk\`  

---

## 1. Executive Summary

This manifest formally freezes the machine learning models, inference pipeline, candidate selection policy, and REST API for the **CivicEngage Procurement Benchmark ML Engine**.

The core predictive system uses an ensemble LightGBM LambdaMART ranker evaluated across a 10-channel multi-view retrieval index and 176 structured/co-occurrence features. It is augmented with **Split Conformal Prediction** for calibrated price intervals and a **Deterministic Reliability Classifier**.

To provide the downstream Large Language Model (Person 2) with the highest-quality candidate evidence within a strict budget of **exactly 10 candidates**, the engine incorporates the accepted research improvement: **Selective MMR Compact Top-10 Candidate Selection**.

### Final Benchmark Performance Summary ($N = 10,560$ Validation Queries)

| Metric | Canonical Baseline | Global MMR ($\lambda=0.6$) | Accepted System: Selective MMR | Net Gain vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Top-10 Oracle Coverage ($\le 10\%$)** | **62.64%** (6,615) | **63.12%** (6,666) | **63.46%** (6,701) | **+0.82 pp (+86 queries)** |
| **Top-5 Oracle Coverage ($\le 10\%$)** | 54.14% | 53.99% | **54.38%** | **+0.24 pp** |
| **Tight Accuracy ($\le 5\%$)** | 53.29% | 53.22% | **53.71%** | **+0.42 pp** |
| **Broad Accuracy ($\le 20\%$)** | 74.29% | 74.59% | **74.73%** | **+0.44 pp** |
| **Rank-1 Benchmark Accuracy ($\le 10\%$)** | **37.57%** | **37.57%** | **37.57%** | **+0.00 pp (Identically Preserved)** |
| **Rank-1 Median APE** | 26.95% | 26.95% | 26.95% | Identically Preserved |
| **Rank-1 Mean APE** | 1206.51% | 1206.51% | 1206.51% | Identically Preserved |
| **Queries Triggering MMR** | 0 (0.0%) | 10,560 (100.0%) | 7,182 (68.0%) | — |
| **Queries Retaining Baseline** | 10,560 (100.0%) | 0 (0.0%) | 3,378 (32.0%) | — |

---

## 2. Frozen Ranker Configuration & Hyperparameters

The tabular ranker is completely frozen and serialized. No further retraining, hyperparameter sweeps, or weight updates are permitted.

* **Model Serialized File:** `models/procurement_ranker.joblib`
* **Algorithm:** LightGBM LambdaMART (`LGBMRanker`)
* **Objective:** `lambdarank` (NDCG optimization)
* **Metric:** `ndcg`
* **Number of Trees / Estimators:** 300
* **Learning Rate:** 0.04
* **Num Leaves:** 31
* **Random State:** 42
* **Total Features:** 176 features
  * **170 Base Features:** Multi-view text embeddings, TF-IDF char/word SVD, Jaccard token similarities, commercial unit alignments, hierarchical price-state estimates, and causal quantity shrinkages.
  * **6 T3 Co-Occurrence Features:** Purchase Order Herfindahl index (`q_hhi`), query primary item indicator (`q_is_primary_item`), log-bundle volume (`log1p_q_other_qty`), candidate commodity membership in PO set (`cand_commodity_in_query_po_set`), absolute HHI difference (`abs_hhi_diff`), and joint primary item status (`both_primary_item`).
* **Inference Invariant:** The benchmark unit price is **ALWAYS** the exact historical awarded unit price of the Rank-1 candidate (`predicted_unit_price = rank1_cand['target_unit_price']`). No synthetic price interpolation, blending, or averaging is ever performed.

---

## 3. Conformal Prediction & Reliability Calibration

### 3.1 Split Conformal Calibration
* **Artifact:** `models/conformal_calibration.joblib`
* **Method:** Multiplicative log-ratio split conformal prediction:
  $$\hat{q} = \text{Quantile}_{1 - \alpha}\left( \left| \log\left( \frac{\hat{P}}{P_{\text{true}}} \right) \right| \right)$$
* **Calibrated Conformal Quantile:** $\hat{q} = 1.5404$
* **Nominal Coverage:** 80.0%
* **Prediction Interval:**
  $$\left[ \max\left(0.01, \hat{P} \cdot e^{-\hat{q}}\right), \hat{P} \cdot e^{\hat{q}} \right]$$

### 3.2 Deterministic Reliability Classifier
Transparently classifies evidence strength into `HIGH`, `MEDIUM`, or `LOW`:
* **`HIGH`:** $\ge 5$ candidates retrieved, structured identity match (Channel A/B/C) or multi-channel consensus ($\ge 2$), narrow price dispersion ($\sigma_{\log} \le 0.35$), positive score margin ($\Delta \ge 0.0$), and recency $\le 3$ years.
* **`LOW`:** $< 3$ candidates retrieved, severe price dispersion ($\sigma_{\log} > 0.85$), coarse fallback channels only (subset of $\{I, J\}$), or zero candidate matches.
* **`MEDIUM`:** All intermediate cases with moderate support and valid specifications.

---

## 4. Accepted Candidate Selection Policy: Selective MMR

The final accepted compact candidate selection policy is **Selective MMR**, validated using deterministic 2-fold cross-validation across all 10,560 queries.

### 4.1 Stratified Product Identity Representation
Because 86.51% of candidate items use generic fallbacks, candidates are represented using stratified identities:
$$\text{StratifiedKey} = \begin{cases} 
\text{product\_identity\_key} & \text{if structured and non-generic} \\
\text{COMMODITY}|\text{GENERIC}|\text{TOKENS}_{1:4}|\text{UOM} & \text{otherwise}
\end{cases}$$

### 4.2 Candidate Pairwise Similarity Kernel
$$\text{Sim}(c_i, c_j) = 0.50 \cdot \mathbb{I}(\text{Key}_i = \text{Key}_j) + 0.30 \cdot \mathbb{I}(\text{PO}_i = \text{PO}_j) + 0.20 \cdot \text{Jaccard}(\text{Tokens}_i, \text{Tokens}_j)$$

### 4.3 Pre-Answer Redundancy Gating
Before reranking, the maximum pairwise similarity among the canonical Top-10 candidates is computed:
$$\text{max\_sim} = \max_{1 \le a < b \le 10} \text{Sim}(c_a, c_b)$$
* **Gate Trigger:** If $\text{max\_sim} \le 0.98$, MMR ($\lambda = 0.6$) is executed over the Top-30 candidate pool to select exactly 10 candidates.
* **Gate Bypass:** If $\text{max\_sim} > 0.98$, near-identical copies choke the top ranks; MMR is bypassed to avoid regressions, and the canonical Top-10 candidates are retained.

### 4.4 Transition Matrix Findings (Why Selective MMR Outperforms Global MMR)
* **Baseline Success $\to$ Preserved Success:** 6,492 queries
* **Baseline Success $\to$ Degradation (Regressions):** **123 queries** (reduced from 281 in Global MMR; 158 regressions avoided!)
* **Baseline Failure $\to$ Recovery (Hits Gained):** **209 queries**
* **Net Out-of-Sample Query Gain:** **+86 queries net** (vs +51 net for Global MMR)

---

## 5. Formal Log of Rejected Experiments

| Experiment ID | Strategy | Result | Decision | Technical Rationale for Rejection |
| :--- | :--- | :---: | :---: | :--- |
| **Exp 1** | Contract Diversity (Max 2 per PO) | 62.44% (-0.20 pp) | **REJECTED** | Artificially truncates valid candidates from single large blanket purchase orders. |
| **Exp 2** | Stratified Identity Diversity (Max 2 per Identity) | 62.65% (+0.01 pp) | **REJECTED** | Statistically neutral; hard truncation removes legitimate price points within identical descriptions. |
| **Exp 3** | Global MMR ($\lambda = 0.6$ applied globally) | 63.12% (+0.48 pp) | **REJECTED AS FINAL** | Caused 281 regressions by penalizing valid candidates when pools were already diverse. Retained only as an ablation. |
| **Exp 5** | Candidate Rescue (Ranks 31–60 Rescue Gate) | 63.55% (+0.09 pp vs Sel MMR) | **REJECTED** | Net gain was only +10 queries across 10,560 queries (+0.09 pp) while causing 116 regressions. Fold 2 held-out net was negative (-7 queries vs Sel MMR). |

---

## 6. Production Smoke Test & API Verification Results

All tests have been verified end-to-end via `python test_freeze_verification.py`:

```
================================================================================
CIVICENGAGE PROCUREMENT BENCHMARK ML -- FINAL FREEZE VERIFICATION
================================================================================
[TEST 1] GET /health ...
Response: {"status": "ok", "service": "procurement-benchmark-ml"}
  -> PASS: Health check endpoint working.

================================================================================
[TEST 2] POST /predict -- Case A: Normal procurement query (Laptop)
Summary Case A:
  Currency:             INR (Exchange Rate: 83.5)
  Benchmark Unit Price: Rs. 60,784.66 ($727.96 USD)
  Expected Range:       Rs. 13,026.00 -- Rs. 283,648.66 (80% coverage)
  Reliability:          MEDIUM
  Candidates Retrieved: 60
  Selective MMR Fired:  True (Max Pair Sim: 0.6455)
  Compact Top-10 Count: 10
  Total Inference Time: 0.135s
  -> PASS: Case A executed successfully.

================================================================================
[TEST 3] POST /predict -- Case B: Multiple plausible historical candidates (Flashlight Batteries)
Summary Case B:
  Currency:             INR (Exchange Rate: 83.5)
  Benchmark Unit Price: Rs. 4,858.03 ($58.18 USD)
  Expected Range:       Rs. 1,041.25 -- Rs. 22,669.42 (80% coverage)
  Reliability:          MEDIUM
  Candidates Retrieved: 60
  Selective MMR Fired:  True (Max Pair Sim: 0.7000)
  Compact Top-10 Count: 10
  Total Inference Time: 0.143s
  -> PASS: Case B executed successfully.

================================================================================
[TEST 4] POST /predict -- Case C: Weak/insufficient evidence (Exotic item)
Summary Case C:
  Benchmark Unit Price: None
  Reliability:          LOW
  Candidates Retrieved: 0
  Compact Top-10 Count: 0
  Selective MMR Fired:  False
  Message:              No matching historical procurement candidates found in catalog.
  -> PASS: Case C handled gracefully with transparent reliability classification.

================================================================================
[TEST 5] POST /predict -- Case D: Locked frozen test period guard (>= 2025-01-01)
  Status code: 400
  Error message: Query date 2025-02-15 is in the frozen test period (>= 2025-01-01). The frozen test is LOCKED. Use a development-period query.
  -> PASS: Frozen test anti-leakage guard strictly rejected post-2025 query.

================================================================================
ALL FREEZE VERIFICATION TESTS PASSED END-TO-END in 406.4s
================================================================================
```

---

## 7. Model Artifacts Registry & Invariants

| Artifact File | Size | Path | Purpose |
| :--- | :---: | :--- | :--- |
| `procurement_ranker.joblib` | ~1.8 MB | `models/procurement_ranker.joblib` | Frozen LightGBM LambdaMART ranker weights |
| `procurement_ranker_metadata.joblib` | ~4 KB | `models/procurement_ranker_metadata.joblib` | Training metadata and feature specifications |
| `conformal_calibration.joblib` | ~2 KB | `models/conformal_calibration.joblib` | Calibrated conformal quantiles ($\hat{q} = 1.5404$) |
| `entity_matcher.joblib` | ~25 MB | `models/entity_matcher.joblib` | TF-IDF token/char vectors and SVD transformers |
| `procurement_catalog.parquet` | ~61 MB | `data/catalog/procurement_catalog.parquet` | Master procurement records |
| `normalized_identity_cache.parquet`| ~12 MB | `data/catalog/normalized_identity_cache.parquet`| Precomputed normalized product identities |
| `parsed_records_cache.joblib` | ~48 MB | `data/catalog/parsed_records_cache.joblib` | Pre-parsed specs and scopes for fast warm-up |
| `frozen_evaluation_dataset.parquet` | ~4.7 MB | `research/procurement_benchmark_topk/` | Frozen test benchmark dataset ($N=105,600$ rows) |

### Strict Operational Invariants
1. **No Data Leakage:** Target prices (`actual_unit_price` / `target_unit_price`) are strictly isolated from candidate retrieval, feature computation, and reranking.
2. **Locked Test Period:** Any query with `award_date_parsed >= 2025-01-01` is strictly rejected at the API and predictor boundary.
3. **No Retraining:** Model artifacts are immutable.

---

## 8. Handoff Contracts & Ownership Boundaries

### Person 1 (Upstream Tabular ML & Candidate Selection) — COMPLETE & FROZEN
* **Responsibility:** Multi-view retrieval, LightGBM LambdaMART scoring, Conformal uncertainty bounds, Reliability classification, and Selective MMR compact Top-10 candidate selection.
* **Status:** Finished and frozen.

### Person 2 (LLM Semantic Reranking Pipeline) — DOWNSTREAM CONSUMER
* **Input Contract:** Receives strictly **10 candidates** via `compact_top10_candidates` adhering to `LLM_RERANKING_INPUT_SPEC.md`.
* **Field Availability:** `candidate_rank`, `candidate_score`, `candidate_unit_price`, `candidate_description`, `purchase_order`, `commodity_code`, `unit_of_measure`, `award_date`.
* **The Non-Negotiable Invariant:** The LLM is strictly an evidence-based semantic reranker, **NEVER a price generator**. The LLM must never interpolate, average, synthesize, or invent prices. The selected price must always be the exact historical unit price of one of the 10 supplied candidates.

### Person 3 (Market Evidence & Pricing Discrepancy) — PARALLEL SERVICE
* **Input Contract:** Consumes benchmark unit price and conformal intervals to compute historical-vs-market price spreads.

### Person 4 (UI / Integration Layer) — SERVICE CONSUMER
* **API Endpoints:**
  * `GET /health` — Service liveness check.
  * `POST /predict` — Main prediction endpoint returning INR prices (converted at RBI reference rate), raw USD figures, conformal expected ranges, reliability badges (`HIGH`/`MEDIUM`/`LOW`), and compact Top-10 candidates.
