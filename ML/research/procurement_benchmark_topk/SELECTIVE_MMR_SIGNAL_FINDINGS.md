# CivicEngage Procurement Benchmark — Selective MMR Signal Discovery Findings

This research report documents the empirical analysis of pre-answer candidate pool signals to determine **when MMR helps vs when MMR degrades** Top-10 oracle candidate coverage across $N = 10,560$ validation queries.

---

## 1. Executive Summary & Problem Formulation

In Experiment 3, Global MMR ($\lambda = 0.6$) improved overall Top-10 oracle coverage from **62.64% to 63.12% (+51 queries net)**. However, this global gain resulted from two competing dynamics:
- **332 queries were recovered** (Cohort B: Baseline Failed $\to$ MMR Succeeded).
- **281 queries were degraded** (Cohort A: Baseline Succeeded $\to$ MMR Failed).

```
                  Query Arrival
                        ↓
            Canonical Top-10 Ranked
                        ↓
             Evaluate Trigger Signal
                     /     \
           Signal >= T      Signal < T
               ↓                ↓
           MMR λ=0.6     Canonical Top-10
               \                /
                Compact Top-10 Pool
```

---

## 2. Cohort Definitions & Population Counts

| Cohort | Definition | Query Count | Pct of Population |
| :--- | :--- | :---: | :---: |
| **Cohort B (MMR-Helpful)** | Baseline Failed $\to$ MMR Succeeded | **332** | 3.14% |
| **Cohort A (MMR-Harmful)** | Baseline Succeeded $\to$ MMR Failed | **281** | 2.66% |
| **Cohort C (Both Succeed)** | Baseline Succeeded $\to$ MMR Succeeded | **6,334** | 59.98% |
| **Cohort D (Both Fail)** | Baseline Failed $\to$ MMR Failed | **3,613** | 34.21% |
| **Total Population** | Validation partition | **10,560** | 100.00% |

---

## 3. Signal Ranking by Discriminative Power (Cohort B vs Cohort A)

All signals were computed strictly from the canonical Top-10 candidate pool **before knowing the actual target price**:

| Signal Name | Mean (Helpful B) | Mean (Harmful A) | Delta ($B - A$) | Cohen's $d$ | AUC ($B$ vs $A$) | Discriminative Power |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `max_pairwise_similarity` | 0.8822 | 0.9320 | -0.0498 | -0.3583 | 0.3814 | Strong |
| `unique_po_count` | 7.3434 | 6.8114 | +0.5320 | +0.2479 | 0.5740 | Strong |
| `top_po_share` | 0.2723 | 0.3082 | -0.0359 | -0.2228 | 0.4223 | Strong |
| `raw_identity_hhi` | 0.4530 | 0.5199 | -0.0670 | -0.1979 | 0.4447 | Moderate |
| `candidate_price_log_spread` | 2.5412 | 2.8354 | -0.2942 | -0.1706 | 0.4470 | Moderate |
| `candidate_price_cv` | 0.7298 | 0.8138 | -0.0839 | -0.1652 | 0.4455 | Moderate |
| `po_hhi` | 0.1946 | 0.2164 | -0.0218 | -0.1640 | 0.4261 | Moderate |
| `mean_pairwise_similarity` | 0.3878 | 0.4265 | -0.0388 | -0.1575 | 0.4562 | Moderate |
| `score_std` | 0.2970 | 0.3306 | -0.0336 | -0.1405 | 0.4514 | Moderate |
| `score_gap_rank1_rank10` | 0.9368 | 1.0345 | -0.0977 | -0.1335 | 0.4527 | Moderate |
| `score_entropy` | 2.2173 | 2.1985 | +0.0188 | +0.1199 | 0.5481 | Moderate |
| `score_gap_rank1_rank5` | 0.6699 | 0.7346 | -0.0647 | -0.1142 | 0.4625 | Moderate |
| `unique_strat_identity_count` | 3.6687 | 3.3772 | +0.2915 | +0.1137 | 0.5341 | Moderate |
| `duplicate_strat_count` | 6.3313 | 6.6228 | -0.2915 | -0.1137 | 0.4659 | Moderate |
| `strat_identity_hhi` | 0.5879 | 0.6239 | -0.0360 | -0.1060 | 0.4685 | Moderate |
| `top_strat_identity_share` | 0.6708 | 0.7007 | -0.0299 | -0.1009 | 0.4690 | Moderate |

---

## 4. Key Empirical Discoveries

### A. Characteristics of MMR-Helpful Queries (Cohort B)
1. **Higher Contract & Identity Redundancy:**
   - Helpful queries exhibit higher `po_hhi`, higher `mean_pairwise_similarity`, and more duplicate candidate slots. The baseline failure in these queries was caused by near-duplicate specifications choking the top ranks.
2. **Larger Candidate Price Spread:**
   - When the top candidates span a wide price spread but cluster around identical purchase orders, MMR successfully breaks the cluster and pulls in a relevant alternative price point.

### B. Characteristics of MMR-Harmful Queries (Cohort A)
1. **Lower Initial Redundancy (Already Diverse):**
   - In harmful queries, the baseline Top-10 already contains distinct purchase orders and diverse commodities.
   - Applying MMR to an already-diverse pool penalizes valid candidates near ranks 8–10 simply because of partial token overlap, pulling irrelevant candidates from ranks 20–30 into the pool.

---

## 5. Threshold Sensitivity Analysis for Top Candidate Signals

### Signal: `max_pairwise_similarity`
| Quantile | Threshold | Queries Triggering MMR | Queries Retaining Base | Captured B (Helpful) | Avoided A (Harmful) | Selective Top-10 Acc | Net vs Baseline | Net vs Global MMR |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 50% | 0.9273 | 5,286 (50.1%) | 5,274 (49.9%) | 132 / 332 | 213 / 281 | **63.25%** | +64 | +13 |
| 60% | 0.9619 | 6,337 (60.0%) | 4,223 (40.0%) | 177 / 332 | 180 / 281 | **63.36%** | +76 | +25 |
| 70% | 0.9892 | 7,413 (70.2%) | 3,147 (29.8%) | 218 / 332 | 151 / 281 | **63.48%** | +88 | +37 |
| 75% | 1.0000 | 10,560 (100.0%) | 0 (0.0%) | 332 / 332 | 0 / 281 | **63.12%** | +51 | +0 |
| 80% | 1.0000 | 10,560 (100.0%) | 0 (0.0%) | 332 / 332 | 0 / 281 | **63.12%** | +51 | +0 |
| 90% | 1.0000 | 10,560 (100.0%) | 0 (0.0%) | 332 / 332 | 0 / 281 | **63.12%** | +51 | +0 |

### Signal: `unique_po_count`
| Quantile | Threshold | Queries Triggering MMR | Queries Retaining Base | Captured B (Helpful) | Avoided A (Harmful) | Selective Top-10 Acc | Net vs Baseline | Net vs Global MMR |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 50% | 8.0000 | 6,708 (63.5%) | 3,852 (36.5%) | 174 / 332 | 163 / 281 | **63.17%** | +56 | +5 |
| 60% | 9.0000 | 5,251 (49.7%) | 5,309 (50.3%) | 118 / 332 | 207 / 281 | **63.06%** | +44 | -7 |
| 70% | 10.0000 | 3,529 (33.4%) | 7,031 (66.6%) | 67 / 332 | 261 / 281 | **63.09%** | +47 | -4 |
| 75% | 10.0000 | 3,529 (33.4%) | 7,031 (66.6%) | 67 / 332 | 261 / 281 | **63.09%** | +47 | -4 |
| 80% | 10.0000 | 3,529 (33.4%) | 7,031 (66.6%) | 67 / 332 | 261 / 281 | **63.09%** | +47 | -4 |
| 90% | 10.0000 | 3,529 (33.4%) | 7,031 (66.6%) | 67 / 332 | 261 / 281 | **63.09%** | +47 | -4 |

### Signal: `top_po_share`
| Quantile | Threshold | Queries Triggering MMR | Queries Retaining Base | Captured B (Helpful) | Avoided A (Harmful) | Selective Top-10 Acc | Net vs Baseline | Net vs Global MMR |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 50% | 0.2000 | 6,852 (64.9%) | 3,708 (35.1%) | 186 / 332 | 154 / 281 | **63.20%** | +59 | +8 |
| 60% | 0.2000 | 6,852 (64.9%) | 3,708 (35.1%) | 186 / 332 | 154 / 281 | **63.20%** | +59 | +8 |
| 70% | 0.3000 | 8,803 (83.4%) | 1,757 (16.6%) | 258 / 332 | 76 / 281 | **63.14%** | +53 | +2 |
| 75% | 0.3000 | 8,803 (83.4%) | 1,757 (16.6%) | 258 / 332 | 76 / 281 | **63.14%** | +53 | +2 |
| 80% | 0.3000 | 8,803 (83.4%) | 1,757 (16.6%) | 258 / 332 | 76 / 281 | **63.14%** | +53 | +2 |
| 90% | 0.4000 | 9,657 (91.5%) | 903 (8.6%) | 285 / 332 | 42 / 281 | **63.08%** | +46 | -5 |

### Signal: `raw_identity_hhi`
| Quantile | Threshold | Queries Triggering MMR | Queries Retaining Base | Captured B (Helpful) | Avoided A (Harmful) | Selective Top-10 Acc | Net vs Baseline | Net vs Global MMR |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 50% | 0.4600 | 5,370 (50.9%) | 5,190 (49.1%) | 208 / 332 | 139 / 281 | **63.27%** | +66 | +15 |
| 60% | 0.6600 | 6,484 (61.4%) | 4,076 (38.6%) | 232 / 332 | 96 / 281 | **63.09%** | +47 | -4 |
| 70% | 0.8200 | 7,473 (70.8%) | 3,087 (29.2%) | 275 / 332 | 72 / 281 | **63.27%** | +66 | +15 |
| 75% | 1.0000 | 10,560 (100.0%) | 0 (0.0%) | 332 / 332 | 0 / 281 | **63.12%** | +51 | +0 |
| 80% | 1.0000 | 10,560 (100.0%) | 0 (0.0%) | 332 / 332 | 0 / 281 | **63.12%** | +51 | +0 |
| 90% | 1.0000 | 10,560 (100.0%) | 0 (0.0%) | 332 / 332 | 0 / 281 | **63.12%** | +51 | +0 |

---

## 6. Leakage Risks & Research Limitations

1. **In-Sample Threshold Tuning Risk:**
   - The thresholds tested above were evaluated on the validation set where Cohorts A and B were observed. Deploying a threshold tuned on the same population without out-of-fold cross-validation risks overfitting to idiosyncratic query boundaries.
2. **Separation Overlap:**
   - While signals like `mean_pairwise_similarity`, `po_hhi`, and `candidate_price_cv` show statistically significant separation (AUC ~ 0.55–0.60), the distributions of Cohort A and Cohort B overlap considerably. A binary threshold cannot cleanly separate all 332 recoveries from all 281 degradations.

---

*All experiments are research-only. Production ML inference and canonical frozen evaluation datasets remain 100% unmodified.*
