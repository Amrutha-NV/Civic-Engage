"""
CivicEngage Procurement Benchmark ML
Step 37 Conformal Price Range Calibration

PURPOSE:
  Calibrate a statistically defensible Split Conformal Prediction layer
  around the existing frozen T3 point predictor (step37_t3_ranker.joblib).

RULES & CONSTRAINTS:
  - ZERO access to the frozen test set (2025-01-02 onward, N=6,755).
  - T3 point predictor remains 100% UNCHANGED.
  - The benchmark point prediction is ALWAYS the selected rank-1 historical candidate's real UNIT_PRICE.
  - Calibration is performed strictly on development validation data (2023-2024, N=10,560).
  - Primary calibration split: Temporal split (Calibration: 2023, N=5,491; Evaluation: 2024, N=5,069).

MATHEMATICAL FORMULATION:
  Given strictly positive prices (UNIT_PRICE > 0), the conformal nonconformity score
  is formulated in the multiplicative / log-ratio domain:

    s_i = |ln(y_i) - ln(y_hat_i)| = |ln(y_i / y_hat_i)|

  For target coverage level (1 - alpha) with calibration sample size n_cal:
    q_hat = Quantile({s_i}, ceil((n_cal + 1) * (1 - alpha)) / n_cal)

  The conformal prediction interval for any future point prediction y_hat is:
    lower_bound = y_hat * exp(-q_hat)
    upper_bound = y_hat * exp(q_hat)

  Properties:
    1. Lower bound is strictly positive (never <= 0, no arbitrary negative price truncation).
    2. Multiplicative scale invariance across all price regimes ($1 to $100,000+).
    3. Distribution-free coverage guarantee: P(Y in [lower, upper]) >= 1 - alpha under exchangeability.

OUTPUT ARTIFACT:
  models/step37_t3_conformal.joblib
"""

import math
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Path setup
ml_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ml_root / "src" / "ranking"))
sys.path.insert(0, str(ml_root / "src" / "normalization"))

from quantity_temporal_ranker import Step30FeatureExtractor


def compute_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Compute finite-sample adjusted conformal quantile:
    q_hat = Quantile_{ ceil((n+1)*(1-alpha)) / n } (scores)
    """
    n = len(scores)
    level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
    # 100 * level percentile
    return float(np.percentile(scores, level * 100.0))


def compute_asymmetric_conformal_quantiles(
    s_low: np.ndarray, s_high: np.ndarray, alpha: float
) -> tuple:
    """
    Asymmetric conformal quantiles split alpha equally: alpha/2 for low, alpha/2 for high.
    """
    n = len(s_low)
    level = min(1.0, math.ceil((n + 1) * (1.0 - alpha / 2.0)) / n)
    q_low = float(np.percentile(s_low, level * 100.0))
    q_high = float(np.percentile(s_high, level * 100.0))
    return q_low, q_high


def evaluate_conformal_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    q_hat: float,
    asymmetric: bool = False,
    q_low: float = 0.0,
    q_high: float = 0.0,
) -> dict:
    """
    Evaluate empirical coverage and interval widths.
    """
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred) & (y_true > 0) & (y_pred > 0)
    y_t = y_true[valid]
    y_p = y_pred[valid]
    n = len(y_t)

    if not asymmetric:
        lower = y_p * np.exp(-q_hat)
        upper = y_p * np.exp(q_hat)
    else:
        lower = y_p * np.exp(-q_low)
        upper = y_p * np.exp(q_high)

    covered = (y_t >= lower) & (y_t <= upper)
    emp_coverage = float(np.mean(covered) * 100.0)

    # Absolute widths
    widths = upper - lower
    avg_width = float(np.mean(widths))
    med_width = float(np.median(widths))

    # Relative widths (width / y_hat)
    rel_widths = (upper - lower) / y_p * 100.0
    avg_rel_width = float(np.mean(rel_widths))
    med_rel_width = float(np.median(rel_widths))

    # Small price edge cases (y_p < $10)
    small_mask = y_p < 10.0
    small_cov = float(np.mean(covered[small_mask]) * 100.0) if np.sum(small_mask) > 0 else 100.0

    # Large price edge cases (y_p > $1,000)
    large_mask = y_p > 1000.0
    large_cov = float(np.mean(covered[large_mask]) * 100.0) if np.sum(large_mask) > 0 else 100.0

    return {
        "n": n,
        "empirical_coverage_pct": round(emp_coverage, 2),
        "avg_width": round(avg_width, 2),
        "med_width": round(med_width, 2),
        "avg_rel_width_pct": round(avg_rel_width, 2),
        "med_rel_width_pct": round(med_rel_width, 2),
        "small_price_coverage_pct": round(small_cov, 2),
        "large_price_coverage_pct": round(large_cov, 2),
        "min_lower_bound": round(float(np.min(lower)), 4),
        "strictly_positive": bool(np.all(lower > 0)),
    }


def main():
    print("=" * 80, flush=True)
    print("CIVICENGAGE PROCUREMENT BENCHMARK ML", flush=True)
    print("STEP 37: SPLIT CONFORMAL PREDICTION CALIBRATION", flush=True)
    print("=" * 80, flush=True)

    t0 = time.time()
    cache_dir = ml_root / "data" / "cache" / "step36"
    model_path = ml_root / "models" / "step37_t3_ranker.joblib"
    conformal_save_path = ml_root / "models" / "step37_t3_conformal.joblib"
    val_preds_cache = ml_root / "data" / "cache" / "val_preds_t3.npy"

    assert model_path.exists(), f"Model not found: {model_path}"

    # ------------------------------------------------------------------ #
    # 1. Load canonical dataset & partition dates                        #
    # ------------------------------------------------------------------ #
    print("\n[Phase 1/5] Loading canonical dataset & verifying partition locks...", flush=True)
    parquet_path = ml_root / "data" / "processed_v2" / "unit_price_training_dataset.parquet"
    df_all = pd.read_parquet(parquet_path)
    df_all["award_date_parsed"] = pd.to_datetime(df_all["award_date_parsed"])
    df_all = df_all.sort_values(["award_date_parsed", "PURCHASE_ORDER"]).reset_index(drop=True)
    df_all["row_id"] = np.arange(len(df_all))

    # STRICT partition verification
    frozen_mask = (df_all["award_date_parsed"] >= pd.to_datetime("2025-01-01")).values
    val_mask = ((df_all["award_date_parsed"] >= pd.to_datetime("2023-01-01")) &
                (df_all["award_date_parsed"] <= pd.to_datetime("2024-12-31"))).values
    n_frozen = int(np.sum(frozen_mask))
    n_val = int(np.sum(val_mask))

    print(f"  * Validation records (2023-2024): N = {n_val:,}", flush=True)
    print(f"  * Frozen test records (>= 2025):  N = {n_frozen:,} [STRICTLY LOCKED - ZERO ACCESS]", flush=True)
    assert n_val == 10560, f"Expected 10,560 val rows, got {n_val}"

    df_val = df_all[val_mask].copy().reset_index(drop=True)
    y_true_val = df_val["target_unit_price"].values
    val_dates = df_val["award_date_parsed"].values

    # Temporal partition within development validation:
    # Calibration set: 2023-01-01 to 2023-12-31
    # Evaluation set:  2024-01-01 to 2024-12-31
    cal_mask_temporal = (df_val["award_date_parsed"] <= pd.to_datetime("2023-12-31")).values
    eval_mask_temporal = (df_val["award_date_parsed"] >= pd.to_datetime("2024-01-01")).values
    n_cal_temporal = int(np.sum(cal_mask_temporal))
    n_eval_temporal = int(np.sum(eval_mask_temporal))

    print(f"  * Calibration partition (2023):  N = {n_cal_temporal:,}", flush=True)
    print(f"  * Evaluation partition  (2024):  N = {n_eval_temporal:,}", flush=True)

    # ------------------------------------------------------------------ #
    # 2. Obtain frozen T3 predictions on all validation queries          #
    # ------------------------------------------------------------------ #
    print("\n[Phase 2/5] Obtaining T3 ranker predictions on validation set...", flush=True)

    if val_preds_cache.exists():
        print(f"  * Loading cached T3 validation predictions: {val_preds_cache.name}", flush=True)
        val_preds = np.load(val_preds_cache)
    else:
        print("  * Computing T3 validation predictions from cached feature matrix...", flush=True)
        t3_ranker = joblib.load(model_path)
        X_val_t0 = np.load(cache_dir / "X_val_t0.npy")
        val_groups = np.load(cache_dir / "val_groups.npy").tolist()
        val_pools_raw = joblib.load(cache_dir / "val_pools.joblib")

        # PO metadata
        po_comm_qty = df_all.groupby(["PURCHASE_ORDER", "COMMODITY"])["quantity_numeric"].sum().reset_index()
        po_tot_qty_s = po_comm_qty.groupby("PURCHASE_ORDER")["quantity_numeric"].transform("sum")
        po_comm_qty["share_sq"] = (po_comm_qty["quantity_numeric"] / np.maximum(po_tot_qty_s, 1e-4)) ** 2
        po_hhi_series = po_comm_qty.groupby("PURCHASE_ORDER")["share_sq"].sum().rename("po_hhi")
        df_all = df_all.merge(po_hhi_series, on="PURCHASE_ORDER", how="left")

        quantities = df_all["quantity_numeric"].values.astype(np.float32)
        po_total_qtys = df_all.groupby("PURCHASE_ORDER")["quantity_numeric"].transform("sum").values.astype(np.float32)
        po_max_qty = df_all.groupby("PURCHASE_ORDER")["quantity_numeric"].transform("max").values.astype(np.float32)
        is_primary_item = (quantities >= (po_max_qty - 1e-4)).astype(np.float32)
        po_hhis = df_all["po_hhi"].fillna(1.0).values.astype(np.float32)
        commodities = df_all["COMMODITY"].values
        po_names = df_all["PURCHASE_ORDER"].values
        po_comm_sets = df_all.groupby("PURCHASE_ORDER")["COMMODITY"].apply(set).to_dict()

        po_name_by_rid = {int(rid): str(po_names[rid]) for rid in range(len(po_names))}
        val_rids = df_all.loc[val_mask, "row_id"].values
        for q_idx, q_rid in enumerate(val_rids):
            po_n = po_name_by_rid.get(int(q_rid), "")
            for c in val_pools_raw[q_idx]:
                c["__q_po_name__"] = po_n

        # Extract 6 T3 features
        t3_list = []
        for q_idx, q_rid in enumerate(val_rids):
            cands = val_pools_raw[q_idx]
            if len(cands) == 0:
                continue
            q_hhi = float(po_hhis[q_rid])
            q_is_prim = float(is_primary_item[q_rid])
            q_qty = float(quantities[q_rid])
            q_tot_qty = float(po_total_qtys[q_rid])
            q_other = max(0.0, q_tot_qty - q_qty)
            q_po_name = cands[0].get("__q_po_name__", "")
            q_comm_set = po_comm_sets.get(q_po_name, set())

            for c in cands:
                c_rid = c["row_id"]
                c_hhi = float(po_hhis[c_rid])
                c_is_prim = float(is_primary_item[c_rid])
                c_comm = commodities[c_rid]
                t3_list.append([
                    q_hhi,
                    q_is_prim,
                    math.log1p(q_other),
                    1.0 if c_comm in q_comm_set else 0.0,
                    abs(q_hhi - c_hhi),
                    1.0 if (q_is_prim > 0.5 and c_is_prim > 0.5) else 0.0,
                ])
        X_val_t3_sub = np.array(t3_list, dtype=np.float32)
        X_val_t3 = np.hstack([X_val_t0, X_val_t3_sub])

        # Predict
        val_scores_flat = t3_ranker.predict(X_val_t3)
        val_preds = np.zeros(n_val, dtype=np.float32)
        offset = 0
        for i in range(n_val):
            k = val_groups[i]
            if k == 0:
                continue
            q_sc = val_scores_flat[offset: offset + k]
            offset += k
            best_idx = int(np.argmax(q_sc))
            val_preds[i] = float(val_pools_raw[i][best_idx].get("target_unit_price", 0.0))

        np.save(val_preds_cache, val_preds)
        print(f"  * Cached validation predictions saved: {val_preds_cache.name}", flush=True)

    # Verification: Check reproduced validation accuracy
    apes_val = np.abs(val_preds - y_true_val) / np.maximum(y_true_val, 1e-4) * 100.0
    acc_10 = float(np.mean(apes_val <= 10.0) * 100.0)
    print(f"  * Verified T3 point accuracy on validation set: {acc_10:.4f}% (+-10% target)", flush=True)
    assert abs(acc_10 - 37.57) < 0.1, f"Prediction mismatch! Got {acc_10:.4f}%"

    # ------------------------------------------------------------------ #
    # 3. Formulate Nonconformity Scores & Calibrate Quantiles            #
    # ------------------------------------------------------------------ #
    print("\n[Phase 3/5] Computing Nonconformity Scores & Conformal Quantiles...", flush=True)

    # Multiplicative Log-Ratio Nonconformity Scores
    # s_i = |ln(y_i) - ln(y_hat_i)|
    log_y = np.log(np.maximum(y_true_val, 1e-4))
    log_y_hat = np.log(np.maximum(val_preds, 1e-4))
    s_sym = np.abs(log_y - log_y_hat)

    # Asymmetric scores
    s_low = np.maximum(0.0, log_y_hat - log_y)     # over-prediction penalty (controls lower bound)
    s_high = np.maximum(0.0, log_y - log_y_hat)    # under-prediction penalty (controls upper bound)

    # Calibration on 2023 subset
    s_sym_cal = s_sym[cal_mask_temporal]
    s_low_cal = s_low[cal_mask_temporal]
    s_high_cal = s_high[cal_mask_temporal]

    # Target coverage levels to analyze: 80%, 85%, 90%
    coverage_levels = [0.80, 0.85, 0.90]
    calibration_results = []

    print("\n  --- TEMPORAL CALIBRATION: Calibrated on 2023 (N=5,491) -> Evaluated on 2024 (N=5,069) ---", flush=True)
    for cov in coverage_levels:
        alpha = 1.0 - cov

        # 1. Symmetric Log-Ratio Conformal
        q_sym = compute_conformal_quantile(s_sym_cal, alpha)
        eval_metrics = evaluate_conformal_intervals(
            y_true_val[eval_mask_temporal],
            val_preds[eval_mask_temporal],
            q_sym,
            asymmetric=False,
        )

        # Multiplicative multiplier factor
        mult_factor = float(np.exp(q_sym))

        res = {
            "method": "Log-Ratio Symmetric Split Conformal",
            "target_coverage_pct": int(cov * 100),
            "alpha": alpha,
            "q_hat": round(q_sym, 4),
            "multiplier_factor": round(mult_factor, 4),
            "lower_multiplier": round(1.0 / mult_factor, 4),
            "upper_multiplier": round(mult_factor, 4),
            "eval_coverage_pct": eval_metrics["empirical_coverage_pct"],
            "eval_avg_rel_width_pct": eval_metrics["avg_rel_width_pct"],
            "eval_med_rel_width_pct": eval_metrics["med_rel_width_pct"],
            "eval_avg_width": eval_metrics["avg_width"],
            "eval_med_width": eval_metrics["med_width"],
            "small_price_cov_pct": eval_metrics["small_price_coverage_pct"],
            "large_price_cov_pct": eval_metrics["large_price_coverage_pct"],
            "strictly_positive": eval_metrics["strictly_positive"],
        }
        calibration_results.append(res)

        print(f"\n  [Target {int(cov*100)}% Coverage (alpha={alpha:.2f})]")
        print(f"    * Conformal log quantile q_hat: {q_sym:.4f} (Multipliers: [{1.0/mult_factor:.4f} * y_hat, {mult_factor:.4f} * y_hat])")
        print(f"    * Empirical Coverage on 2024:    {eval_metrics['empirical_coverage_pct']}% (Target: {int(cov*100)}%)")
        print(f"    * Median Relative Width:         {eval_metrics['med_rel_width_pct']}%")
        print(f"    * Average Width:                 ${eval_metrics['avg_width']:,.2f} (Median: ${eval_metrics['med_width']:,.2f})")
        print(f"    * Small Price (<$10) Coverage:   {eval_metrics['small_price_coverage_pct']}%")
        print(f"    * Large Price (>$1,000) Coverage:{eval_metrics['large_price_coverage_pct']}%")
        print(f"    * Lower bounds strictly positive: {eval_metrics['strictly_positive']}")

    # Also compute 80% and 90% quantiles on full validation set for production artifact
    print("\n  --- FULL VALIDATION (N=10,560) PRODUCTION CALIBRATION ---", flush=True)
    prod_quantiles = {}
    for cov in coverage_levels:
        alpha = 1.0 - cov
        q_prod = compute_conformal_quantile(s_sym, alpha)
        full_eval = evaluate_conformal_intervals(
            y_true_val, val_preds, q_prod, asymmetric=False
        )
        prod_quantiles[f"{int(cov*100)}pct"] = {
            "target_coverage": cov,
            "alpha": alpha,
            "q_hat": round(q_prod, 4),
            "multiplier": round(float(np.exp(q_prod)), 4),
            "lower_multiplier": round(float(1.0 / np.exp(q_prod)), 4),
            "upper_multiplier": round(float(np.exp(q_prod)), 4),
            "empirical_coverage_pct": full_eval["empirical_coverage_pct"],
            "med_width": full_eval["med_width"],
            "avg_width": full_eval["avg_width"],
        }
        print(f"    * {int(cov*100)}% Coverage: q_hat = {q_prod:.4f} -> Multipliers: [{1.0/np.exp(q_prod):.4f} * y_hat, {np.exp(q_prod):.4f} * y_hat], Emp Coverage = {full_eval['empirical_coverage_pct']}%")

    # Primary selected configuration for procurement benchmark:
    # 80% coverage is the standard procurement confidence band (balanced width vs coverage)
    # with 90% available as high-confidence conservative band
    primary_cov = "80pct"
    selected_q_hat = prod_quantiles[primary_cov]["q_hat"]
    selected_mult = prod_quantiles[primary_cov]["multiplier"]

    # ------------------------------------------------------------------ #
    # 4. Serialize Conformal Calibration Artifact                        #
    # ------------------------------------------------------------------ #
    print("\n[Phase 4/5] Serializing Conformal Calibration Artifact...", flush=True)

    conformal_artifact = {
        "artifact_type": "SplitConformalPredictionInterval",
        "model_version": "Step37_T3_CoOccurrence_Ranker",
        "calibration_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "frozen_test_access_count": 0,
        "calibration_method": "multiplicative_log_ratio_split_conformal",
        "formula": {
            "nonconformity_score": "s_i = |ln(y_i) - ln(y_hat_i)|",
            "lower_bound": "lowerBound = max(0.01, benchmarkUnitPrice * exp(-q_hat))",
            "upper_bound": "upperBound = benchmarkUnitPrice * exp(q_hat)",
        },
        "primary_configuration": {
            "default_coverage_level": 0.80,
            "alpha": 0.20,
            "q_hat": selected_q_hat,
            "lower_multiplier": round(float(1.0 / selected_mult), 4),
            "upper_multiplier": round(float(selected_mult), 4),
            "calibration_sample_size": n_val,
            "calibration_dataset_range": "2023-01-01 to 2024-12-31",
            "empirical_coverage_pct": prod_quantiles[primary_cov]["empirical_coverage_pct"],
            "median_interval_width": prod_quantiles[primary_cov]["med_width"],
            "average_interval_width": prod_quantiles[primary_cov]["avg_width"],
        },
        "available_coverage_quantiles": prod_quantiles,
        "temporal_evaluation_2023_to_2024": calibration_results,
    }

    joblib.dump(conformal_artifact, conformal_save_path, compress=3)
    artifact_size_kb = conformal_save_path.stat().st_size / 1024
    print(f"  * Conformal calibration artifact saved: {conformal_save_path} ({artifact_size_kb:.1f} KB)", flush=True)

    # ------------------------------------------------------------------ #
    # 5. Verification & Example Range Demonstration                      #
    # ------------------------------------------------------------------ #
    print("\n[Phase 5/5] Demonstrating Conformal Intervals on Representative Benchmarks...", flush=True)
    example_benchmarks = [
        ("Office Supplies (Pen/Tape)", 2.45),
        ("Medical Supplies (Sterile Box)", 8.69),
        ("Safety Vest / PPE", 45.00),
        ("Laptop Computer 15-inch", 727.96),
        ("Enterprise Server / Network Switch", 4850.00),
        ("Heavy Industrial Pump / Valve", 28500.00),
    ]

    print("\n  Representative Conformal Ranges (80% Default Coverage, q_hat={:.4f}):".format(selected_q_hat))
    print("  " + "-" * 75)
    print(f"  {'Item Category':<35} {'Benchmark':<12} {'Conformal Expected Range':<25}")
    print("  " + "-" * 75)
    for cat, p in example_benchmarks:
        low = p * math.exp(-selected_q_hat)
        high = p * math.exp(selected_q_hat)
        print(f"  {cat:<35} ${p:>9,.2f}   ${low:>8,.2f} -- ${high:>8,.2f}")
    print("  " + "-" * 75)

    total_time = time.time() - t0
    print(f"\n[Conformal Calibration Complete in {total_time:.1f}s]")
    print(f"  Artifact:    {conformal_save_path}")
    print(f"  Frozen test: STRICTLY LOCKED (zero access)")


if __name__ == "__main__":
    main()
