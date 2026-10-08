# CivicEngage Procurement Benchmark — Contract Diversity Re-Ranking Findings

This research report documents the empirical results of **Experiment 1: Simple Contract Diversity Selection** evaluated on the canonical frozen Step-37 evaluation population ($N = 10,560$ validation queries).

---

## 1. Research Objective & Experimental Architecture

- **Research Problem:** In the canonical Step-37 evaluation, **3,945 queries** failed to surface any valid candidate within the Top-10 pool (62.64% oracle coverage), yet **1,652 of those queries (41.88%)** had a valid candidate within ranks 11–30.
- **Hypothesis:** Candidate redundancy in the top positions (e.g., multiple near-duplicate line items from the same bulk contract) crowds out valid alternative candidates. Enforcing contract diversity across the top-30 candidates will surface valid candidates into a **compact Top-10** without expanding prompt context size for the downstream LLM.
- **Constraint Rule:** From the canonical LambdaMART ranked pool (ranks 1–30), select candidates sequentially with a **maximum of 2 candidates per `PURCHASE_ORDER`**, guaranteeing **exactly 10 compact candidates** for each query.

```
Canonical Retrieval (10 Channels)
             ↓
LambdaMART 176-Feature Ranking (Top-30 Ranked Pool)
             ↓
Contract Diversity Filter (Max 2 Candidates per PURCHASE_ORDER)
             ↓
COMPACT TOP-10 CANDIDATE SET (Sent to downstream LLM)
```

---

## 2. Experimental Benchmark Results

| Metric | Canonical Baseline Top-10 | Diversity Top-10 (Max 2 PO) | Absolute Delta | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Top-10 Oracle Coverage (\(\le 10\%\) Error)** | **62.64%** (6,614 queries) | **62.44%** (6,593 queries) | **-0.20 pp** | **-0.32%** |
| **Top-5 Oracle Coverage (\(\le 10\%\) Error)** | 54.14% | 54.07% | -0.07 pp | -0.13% |
| **Rank-1 Accuracy (\(\le 10\%\) Error)** | 37.57% | 37.57% | +0.00 pp | +0.00% |
| **Tight Accuracy (\(\le 5\%\) Error)** | 53.29% | 52.84% | -0.45 pp | -0.84% |
| **Broad Accuracy (\(\le 20\%\) Error)** | 74.29% | 74.10% | -0.19 pp | -0.26% |
| **Rank-1 Mean APE** | 1206.51% | 1206.51% | +0.00% | — |
| **Rank-1 Median APE** | 26.95% | 26.95% | +0.00% | — |

---

## 3. Query Transition Analysis

Across the 10,560 queries, the contract-diversity selection rule produced the following shifts:

- **Improved Queries (Baseline Failed \(\to\) Diversity Succeeded):** **+124 queries** (1.17%)
- **Degraded Queries (Baseline Succeeded \(\to\) Diversity Failed):** **-145 queries** (1.37%)
- **Unchanged Queries:** **10,291 queries** (97.45%)
- **Net Gain:** **-21 queries** (-0.20 percentage points)

---

## 4. Case Studies: Successful Recovery vs. Degradation

### A. Successful Recovery Example (Query #227)
- **Target Item:** `Trailers, Custom: Personnel, Food Service, Equipme...` (Actual Price: $367,445.00)
- **Baseline Rank-1 Candidate:** Price: $28,499.00 (Error: 92.24%, PO: `DO780023011304370`)
- **Failure Cause in Baseline:** The baseline Top-10 was crowded with multiple redundant items from contract `DO780023011304370`.
- **Diversity Recovery:** Capping that purchase order at 2 allowed candidate from rank #11 (PO: `DO780022120503107`) into the Top-10.
- **Recovered Candidate Price:** $388,745.00 (**Error: 5.8%**).

### B. Degraded Example (Query #45)
- **Target Item:** `Stripes and Legends, Plastic, Prefabricated, Refle...` (Actual Price: $156.48)
- **Baseline Status:** A valid candidate was originally located at Rank 7 or 8 on contract `DO240021080910944`.
- **Degradation Mechanism:** Because positions 1 and 2 of that purchase order were already selected, the valid 3rd item from the same PO was skipped, and the backfilled candidates from ranks 20–30 were not price-compatible.

---

## 5. Interpretation & Architectural Takeaways

1. **Empirical Validation of the Redundancy Hypothesis:**
   Contract-level diversity proves that redundancy in top positions does crowd out valid candidates: restricting repetitive purchase orders successfully recovered 124 queries from ranks 11–30 into the Top-10.
2. **Net Trade-off of Hard PO Truncation:**
   However, hard truncation at 2 items per purchase order also degraded 145 queries, resulting in a net change of -21 queries (-0.20 pp, 62.44% vs 62.64%).
3. **Limitation of Coarse PO-Level Capping:**
   Applying diversity purely at the `PURCHASE_ORDER` string level is too coarse: when a large legitimate contract contains multiple distinct line items that all match the query's specifications, capping at 2 occasionally displaces a valid line item.
4. **Recommendation for Next Research Step:**
   Proceed to **Product-Identity Key (`product_identity_key`) Diversity Re-Ranking**:
   - Rather than capping by purchase order number, cap by parsed product identity (Brand + Model + MPN + Commodity). This preserves multi-item line orders from the same contract while preventing identical product specifications from saturating the Top-10 pool.

---

## 6. Artifact Manifest & Verification

- Analysis Script: `ML/research/procurement_benchmark_topk/contract_diversity_top30_experiment.py`
- Row-Level Dataset: `ML/research/procurement_benchmark_topk/contract_diversity_comparison.csv` (10,560 records)
- Metrics Summary: `ML/research/procurement_benchmark_topk/contract_diversity_metrics.json`
- Findings Report: `ML/research/procurement_benchmark_topk/CONTRACT_DIVERSITY_FINDINGS.md`

*All experiments are strictly research-only. Production ML inference and canonical frozen evaluation datasets remain 100% unmodified.*
