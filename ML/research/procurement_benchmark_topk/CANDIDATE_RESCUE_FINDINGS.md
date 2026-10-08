# CivicEngage Procurement Benchmark — Candidate Rescue (Ranks 31–60) Findings

This research report documents the empirical evaluation of **Candidate Rescue from Ranks 31–60** integrated with Selective MMR on the canonical validation population ($N = 10,560$ queries) using a strict **2-Fold Cross-Validation protocol**.

---

## 1. Executive Summary: Three-System Benchmark Comparison

| Metric | System A: Canonical Baseline | System B: Selective MMR | System C: Candidate Rescue + Selective MMR | Delta (C vs Baseline) | Delta (C vs Selective MMR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Top-10 Oracle Coverage ($\le 10\%$)** | **62.64%** (6,615) | **63.46%** (6,701) | **63.55%** (6,711) | **+0.91 pp** (+96 queries) | **+0.09 pp** (+10 queries) |
| **Top-5 Oracle Coverage ($\le 10\%$)** | 54.14% | 54.38% | 54.38% | +0.24 pp | +0.00 pp |
| **Rank-1 Accuracy ($\le 10\%$)** | 37.57% | 37.57% | 37.57% | +0.00 pp | +0.00 pp |
| **Tight Accuracy ($\le 5\%$)** | 53.29% | 53.71% | 53.88% | +0.59 pp | +0.17 pp |
| **Broad Accuracy ($\le 20\%$)** | 74.29% | 74.73% | 75.19% | +0.90 pp | +0.46 pp |
| **Rank-1 Mean APE** | 1206.51% | 1206.51% | 1206.51% | +0.00% | — |
| **Rank-1 Median APE** | 26.95% | 26.95% | 26.95% | +0.00% | — |

---

## 2. Rescue Activation & Transition Dynamics

Across the 10,560 validation queries:
- **Queries where Rescue Activated:** **6,494 queries** (61.50%)
- **Successful Rescues (Selective MMR Failed $\to$ Rescue Succeeded):** **+126 queries**
- **Harmful Rescues (Selective MMR Succeeded $\to$ Rescue Failed):** **-116 queries**
- **Neutral Rescues (Hit status invariant):** **6,252 queries**
- **Net Gain vs. Selective MMR:** **+10 queries** (+0.09 pp)
- **Net Gain vs. Canonical Baseline:** **+96 queries** (+0.91 pp)

---

## 3. 2-Fold Cross-Validation Performance

### Fold 1: Trained on Fold 1 $\to$ Held-Out Evaluation on Fold 2 ($N = 5,280$)
- **Selected Rescue Threshold:** `threshold = 0.7`
- **Held-Out Results:**
  - Baseline Hits: 3,264
  - Selective MMR Hits: 3,319
  - Candidate Rescue Hits: **3,312**
  - **Held-Out Net vs Selective MMR:** **-7 queries** (45 successful vs 52 harmful)

### Fold 2: Trained on Fold 2 $\to$ Held-Out Evaluation on Fold 1 ($N = 5,280$)
- **Selected Rescue Threshold:** `threshold = 0.4`
- **Held-Out Results:**
  - Baseline Hits: 3,351
  - Selective MMR Hits: 3,382
  - Candidate Rescue Hits: **3,399**
  - **Held-Out Net vs Selective MMR:** **+17 queries** (81 successful vs 64 harmful)

---

## 4. Architectural Decision & Final Proposal

### Decision:
1. **Does Candidate Rescue provide a meaningful out-of-sample improvement over Selective MMR?**
   - **Result:** Net gain vs Selective MMR is **+10 queries (+0.09 pp)**.
   - While Candidate Rescue successfully recovered **+126 previously unreachable queries** from ranks 31–60, swapping slot 10 also displaced a valid candidate in **116 queries**.
   - The marginal gain of 10 queries on top of Selective MMR confirms that **Selective MMR is already capturing the primary viable compact re-ranking headroom**.
2. **Proposed Final Compact Selection Architecture:**
   - **Selective MMR (System B)** remains the **cleanest, most stable, and mathematically grounded compact Top-10 policy** for Person 1.
   - It delivers **63.46% oracle coverage (+86 queries over baseline)** without requiring a secondary heuristic rescue layer.
   - For queries with genuine coverage gaps, the **Historical Evidence Gate** correctly routes them to **Market Evidence / Person 3**.

---

*All experiments are research-only. Production ML inference and canonical frozen evaluation datasets remain 100% unmodified.*
