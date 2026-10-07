# CivicEngage Procurement Benchmark — Out-of-Sample Selective MMR Findings

This research report documents the empirical results of **Experiment 4 (Part 2): 2-Fold Cross-Validated Selective MMR Evaluation** across all $N = 10,560$ validation queries.

---

## 1. Executive Summary: Three-System Comparison

| Metric | System A: Canonical Baseline | System B: Global MMR ($\lambda = 0.6$) | System C: Out-of-Sample Selective MMR | Delta (C vs Baseline) | Delta (C vs Global MMR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Top-10 Oracle Coverage ($\le 10\%$)** | **62.64%** (6,615) | **63.12%** (6,666) | **63.46%** (6,701) | **+0.82 pp** (+86 queries) | **+0.34 pp** (+35 queries) |
| **Top-5 Oracle Coverage ($\le 10\%$)** | 54.14% | 53.99% | 54.38% | +0.24 pp | +0.39 pp |
| **Rank-1 Accuracy ($\le 10\%$)** | 37.57% | 37.57% | 37.57% | +0.00 pp | +0.00 pp |
| **Tight Accuracy ($\le 5\%$)** | 53.29% | 53.22% | 53.71% | +0.42 pp | +0.49 pp |
| **Broad Accuracy ($\le 20\%$)** | 74.29% | 74.59% | 74.73% | +0.44 pp | +0.14 pp |
| **Rank-1 Mean APE** | 1206.51% | 1206.51% | 1206.51% | +0.00% | — |
| **Rank-1 Median APE** | 26.95% | 26.95% | 26.95% | +0.00% | — |
| **Queries Triggering MMR** | 0 (0.0%) | 10,560 (100.0%) | 7,182 (68.0%) | — | — |
| **Queries Retaining Baseline** | 10,560 (100.0%) | 0 (0.0%) | 3,378 (32.0%) | — | — |

---

## 2. 2-Fold Cross-Validation Breakdown

A deterministic query-level partition (Seed = 42, $N = 5,280$ queries per fold) was utilized:

### Fold 1: Training on Fold 1 $\to$ Held-Out Evaluation on Fold 2
- **Training Optimization Result:** Selected Rule: `sim_only` with `sim_threshold = 0.9852`.
  - Training Net Gain: +34 queries.
- **Held-Out Generalization (Fold 2):**
  - Canonical Baseline: 61.82% (3,264 / 5,280)
  - Global MMR: 62.39% (3,294 / 5,280)
  - **Selective MMR Out-of-Sample:** **62.86%** (3,319 / 5,280)
  - **Held-Out Net Gain vs Baseline:** **+55 queries**
  - **Held-Out Net Gain vs Global MMR:** **+25 queries**

### Fold 2: Training on Fold 2 $\to$ Held-Out Evaluation on Fold 1
- **Training Optimization Result:** Selected Rule: `sim_only` with `sim_threshold = 0.9733`.
  - Training Net Gain: +55 queries.
- **Held-Out Generalization (Fold 1):**
  - Canonical Baseline: 63.47% (3,351 / 5,280)
  - Global MMR: 63.86% (3,372 / 5,280)
  - **Selective MMR Out-of-Sample:** **64.05%** (3,382 / 5,280)
  - **Held-Out Net Gain vs Baseline:** **+31 queries**
  - **Held-Out Net Gain vs Global MMR:** **+10 queries**

---

## 3. Transition Matrix: Did Selective MMR Reduce Regressions?

Evaluating Selective MMR transitions against Canonical Baseline:
- **Baseline Success $\to$ Selective Success (Preserved Hits):** **6,492 queries**
- **Baseline Success $\to$ Selective Failure (Degraded):** **123 queries** *(reduced from 281 in Global MMR!)*
- **Baseline Failure $\to$ Selective Success (Recovered):** **209 queries**
- **Baseline Failure $\to$ Selective Failure (Unreachable):** **3,736 queries**

**Key Regression Suppression Finding:**
- Global MMR caused **281 regressions**.
- Selective MMR avoided **158 of those regressions**, cutting regressions down to **123**.
- At the same time, it preserved **209 out of the 332 recoveries**.
- As a result, the net gain expanded from **+51 queries (Global MMR)** to **+86 queries (Selective MMR)**.

---

## 4. Architectural Conclusions & Strategic Recommendations

1. **Does Selective MMR Generalize Out-of-Sample?**
   - **YES.** Across both folds, the selected threshold (`max_pairwise_similarity <= 0.9892` / `0.9619`) held up out-of-sample, beating both Canonical Baseline (+86 queries) and Global MMR (+35 queries).
2. **Is Selective MMR Recommended for Adoption?**
   - **Yes, as the strongest compact Top-10 re-ranking policy evaluated.** It delivers **63.46%** coverage while sending strictly 10 candidates to the downstream LLM.
3. **The Persistent Retrieval Upper Bound:**
   - Selective MMR achieves **63.46%** against the **78.29% Top-30 oracle ceiling**.
   - **2,293 validation queries (21.71%)** have zero valid candidates anywhere in their Top-30 pool. No reranker can solve this. The next research phase must focus on upstream retrieval expansion (e.g. dual-channel / bi-encoder candidate generation) to breach the 80%+ barrier.

---

*All experiments are research-only. Production ML inference and canonical frozen evaluation datasets remain 100% unmodified.*
