# FINAL FROZEN TEST EVALUATION REPORT

- **Execution Timestamp:** 2026-09-20T17:32:12.688711+00:00
- **Model Artifact:** `models/step37_t3_ranker.joblib`
- **Features:** 176
- **Frozen Test Set Size (N):** 6,755
- **Date Range:** 2025-01-02 to 2026-07-31
- **Frozen-Test Access Count:** 1

## Performance Metrics

| Metric | Result |
| :--- | :--- |
| **Primary Rank-1 (+-10%)** | **36.58%** |
| Rank-1 (+-5%) | 29.95% |
| Rank-1 (+-20%) | 44.47% |
| Top-3 (+-10%) | 48.73% |
| Top-5 (+-10%) | 54.39% |
| Top-10 (+-10%) | 63.97% |
| MAE | $5,732.05 |
| MdAPE | 30.00% |
| log-RMSE | 1.4422 |
| High-Dispersion (+-10%) | 29.68% (N=5287) |
| Candidate-Pool Oracle (+-10%) | 85.73% |
| Mean Candidate Count | 56.16 |
| Inference Failures | 0 |

## Reliability Tier Breakdown

- **HIGH**: 2,951 queries (43.7%)
- **MEDIUM**: 1,674 queries (24.8%)
- **LOW**: 2,130 queries (31.5%)
