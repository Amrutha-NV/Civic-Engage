# ML FINAL FREEZE MANIFEST

**Project:** CivicEngage Procurement Benchmark ML  
**Status:** FROZEN & COMPLETED  
**Evaluation Type:** Single-Pass Final Held-Out Evaluation  

---

## 1. Core Model & Pipeline Artifacts

- **Final Model Artifact:** `models/step37_t3_ranker.joblib`
- **Metadata Artifact:** `models/step37_t3_metadata.joblib`
- **Conformal Calibration Artifact:** `models/step37_t3_conformal.joblib`
- **Core Inference Engine:** `src/inference/predict.py`
- **REST Service API:** `src/api/server.py`

---

## 2. Final Model Architecture & Hyperparameters

- **Model Type:** LightGBM `LGBMRanker`
- **Objective:** `lambdarank`
- **Metric:** `ndcg` (eval_at = `[1, 3, 5, 10]`)
- **Number of Estimators (`n_estimators`):** `300`
- **Learning Rate (`learning_rate`):** `0.04`
- **Number of Leaves (`num_leaves`):** `31`
- **Random State (`random_state`):** `42`
- **Feature Count:** `176` features total
  - 170 Step 30 / Step 36 baseline features (`T5_INDICES`)
  - 6 Step 37 T3 Co-Occurrence / Bundle context features (`T3_CO_OCC_FEATURE_NAMES`):
    1. `q_hhi` (PO commodity concentration Herfindahl-Hirschman Index)
    2. `q_is_primary_item` (primary high-quantity item indicator)
    3. `log1p_q_other_qty` (log other quantities in the same PO)
    4. `cand_commodity_in_query_po_set` (commodity co-occurrence flag)
    5. `abs_hhi_diff` (absolute concentration difference)
    6. `both_primary_item` (joint primary bundle item indicator)

---

## 3. Development Validation Results

- **Validation Window:** 2023-01-03 to 2024-12-31 ($N = 10,560$)
- **Rank-1 $\pm 10\%$ Accuracy:** **`37.57%`**
- **Rank-1 $\pm 5\%$ Accuracy:** `31.52%`
- **Rank-1 $\pm 20\%$ Accuracy:** `45.69%`
- **Top-3 $\pm 10\%$:** `48.49%`
- **Top-5 $\pm 10\%$:** `54.14%`
- **Top-10 $\pm 10\%$:** `62.64%`
- **MAE:** `$6,729.19`

---

## 4. Final Frozen Test Results

The frozen test set was held strictly untouched and locked throughout the entire research and development lifecycle. It was accessed exactly once for this final evaluation.

- **Frozen Test Population ($N$):** `6,755`
- **Date Range:** `2025-01-02` to `2026-07-31`
- **Rank-1 $\pm 10\%$ Accuracy (Primary):** **`36.58%`**
- **Rank-1 $\pm 5\%$ Accuracy:** `29.95%`
- **Rank-1 $\pm 20\%$ Accuracy:** `44.47%`
- **Top-3 $\pm 10\%$ Accuracy:** `48.73%`
- **Top-5 $\pm 10\%$ Accuracy:** `54.39%`
- **Top-10 $\pm 10\%$ Accuracy:** `63.97%`
- **Mean Absolute Error (MAE):** `$5,732.05`
- **Median Absolute Percentage Error (MdAPE):** `30.00%`
- **Log Root Mean Squared Error (logRMSE):** `1.4422`
- **High-Dispersion ($\sigma_{\ln} \ge 0.80$) $\pm 10\%$ Accuracy:** `29.68%` ($N = 5,287$)
- **Candidate-Pool Oracle $\pm 10\%$ Ceiling:** `85.73%`
- **Mean Candidate Count per Query:** `56.16`
- **Inference Failures:** `0` (100.0% completion rate)
- **Frozen-Test Access Count:** `1`

---

## 5. Target Assessment

> **The target of >65% within $\pm 10\%$ of actual `UNIT_PRICE` was NOT achieved.**
> The empirical rank-1 accuracy on the frozen test partition is **36.58%**. This result is recorded objectively without alteration, cherry-picking, or post-hoc reinterpretation.

---

## 6. Uncertainty & Reliability Calibration

### Conformal Prediction Layer
- **Method:** Multiplicative Log-Ratio Split Conformal Prediction
- **Calibration Quantile ($\hat{q}$):** `1.5404`
- **Coverage Level Target:** `80%`
- **Multipliers:**
  - Lower bound multiplier: $e^{-\hat{q}} = 0.2143$
  - Upper bound multiplier: $e^{\hat{q}} = 4.6667$

### Reliability Classification Distribution (Frozen Test)
- **HIGH:** `43.7%` (2,951 queries)
- **MEDIUM:** `24.8%` (1,674 queries)
- **LOW:** `31.5%` (2,130 queries)

---

## 7. Service Integration Status

- **API Framework:** FastAPI + Uvicorn (`src/api/server.py`)
- **Endpoints:**
  - `GET /health`
  - `POST /predict`
- **Currency Presentation Layer:**
  - Internal engine retains native USD calculations
  - Presentation conversion to INR applied strictly at API response boundary via `USD_TO_INR_RATE`
  - Full statistical auditability maintained via `rawUSD` response payload
