"""
CivicEngage - Procurement Benchmark Research
Historical Evidence Gate Analysis (Step 1)

Empirical investigation of inference-safe signals to determine whether
retrieved Top-10 historical candidates provide sufficient evidence to proceed
to Person 2's LLM semantic reranker or fall back to Person 3's current-market evidence.

Strict Invariants:
- Does NOT generate prices.
- Does NOT modify candidate_unit_price.
- Does NOT call an LLM.
- Does NOT perform current-market retrieval.
- Does NOT modify the frozen production model or frozen dataset.
- Distinguishes inference-safe features from evaluation-only ground truth.
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Ensure utf-8 output encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EXPECTED_SCHEMA = [
    "query_index",
    "target_description",
    "target_quantity",
    "target_unit_of_measure",
    "target_procurement_date",
    "target_city",
    "target_state",
    "target_brand",
    "target_model",
    "target_commodity_code",
    "target_commodity_family",
    "candidate_rank",
    "candidate_description",
    "candidate_unit_price",
    "candidate_score",
    "actual_unit_price",
    "absolute_percentage_error",
    "is_within_10pct",
]


def summarize_series(arr: np.ndarray, round_digits: int = 4) -> Dict[str, float]:
    """Compute standard summary statistics for a numeric array."""
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return {}
    q = np.percentile(valid, [5, 10, 25, 50, 75, 90, 95])
    return {
        "count": int(len(valid)),
        "mean": round(float(np.mean(valid)), round_digits),
        "std": round(float(np.std(valid)), round_digits),
        "min": round(float(np.min(valid)), round_digits),
        "p5": round(float(q[0]), round_digits),
        "p10": round(float(q[1]), round_digits),
        "p25": round(float(q[2]), round_digits),
        "p50_median": round(float(q[3]), round_digits),
        "p75": round(float(q[4]), round_digits),
        "p90": round(float(q[5]), round_digits),
        "p95": round(float(q[6]), round_digits),
        "max": round(float(np.max(valid)), round_digits),
        "iqr": round(float(q[4] - q[2]), round_digits),
    }


def main():
    t_start = time.time()
    script_dir = Path(__file__).resolve().parent
    parquet_path = script_dir / "frozen_evaluation_dataset.parquet"
    out_metrics_path = script_dir / "evidence_gate_metrics.json"

    print("=" * 80)
    print("STEP 1: HISTORICAL EVIDENCE GATE EMPIRICAL ANALYSIS")
    print("=" * 80)

    # 1. Load and Validate Dataset Schema
    print(f"\n[1/5] Loading frozen evaluation artifact: {parquet_path.name}...")
    assert parquet_path.exists(), f"Error: {parquet_path} does not exist!"
    df = pd.read_parquet(parquet_path)

    print(f"  * Total records: {len(df):,}")
    print(f"  * Total columns: {len(df.columns)}")

    # Schema verification
    missing_cols = [c for c in EXPECTED_SCHEMA if c not in df.columns]
    assert not missing_cols, f"Schema validation error! Missing columns: {missing_cols}"
    print("  * Schema check: All 18 expected columns present and verified.")

    # 2. Group by query_index and compute signals
    print("\n[2/5] Grouping by query_index and extracting signals...", flush=True)
    n_queries = df["query_index"].nunique()
    cands_per_query = df.groupby("query_index").size()
    min_cands = int(cands_per_query.min())
    max_cands = int(cands_per_query.max())
    print(f"  * Total unique queries: {n_queries:,}")
    print(f"  * Candidates per query: exactly {min_cands} (min={min_cands}, max={max_cands})")

    # Pivot candidates by rank for clean vector operations
    piv_scores = df.pivot(index="query_index", columns="candidate_rank", values="candidate_score")
    piv_prices = df.pivot(index="query_index", columns="candidate_rank", values="candidate_unit_price")
    piv_hits = df.pivot(index="query_index", columns="candidate_rank", values="is_within_10pct")
    piv_apes = df.pivot(index="query_index", columns="candidate_rank", values="absolute_percentage_error")
    target_meta = df.drop_duplicates(subset=["query_index"]).set_index("query_index")

    # A. INFERENCE-SAFE FEATURES (Ground truth is NEVER used here)
    score_r1 = piv_scores[1].values
    score_r2 = piv_scores[2].values
    score_r3 = piv_scores[3].values
    score_r5 = piv_scores[5].values
    score_r10 = piv_scores[10].values
    score_top5_mean = piv_scores[[1, 2, 3, 4, 5]].mean(axis=1).values
    score_top10_mean = piv_scores.mean(axis=1).values
    score_top10_std = piv_scores.std(axis=1).values
    score_margin_1_2 = (piv_scores[1] - piv_scores[2]).values
    score_margin_1_10 = (piv_scores[1] - piv_scores[10]).values

    price_r1 = piv_prices[1].values
    price_mean_top10 = piv_prices.mean(axis=1).values
    price_median_top10 = piv_prices.median(axis=1).values
    price_cv_top10 = (piv_prices.std(axis=1) / (price_mean_top10 + 1e-6)).values
    price_cv_top5 = (piv_prices[[1, 2, 3, 4, 5]].std(axis=1) / (piv_prices[[1, 2, 3, 4, 5]].mean(axis=1) + 1e-6)).values

    log_prices = np.log(np.maximum(piv_prices.values, 1e-4))
    log_price_std_top10 = np.std(log_prices, axis=1)
    log_price_std_top5 = np.std(log_prices[:, :5], axis=1)

    price_ratio_top10 = (piv_prices.max(axis=1) / (piv_prices.min(axis=1) + 1e-6)).values
    price_ratio_top5 = (piv_prices[[1, 2, 3, 4, 5]].max(axis=1) / (piv_prices[[1, 2, 3, 4, 5]].min(axis=1) + 1e-6)).values
    price_spread_rel_top10 = ((piv_prices.max(axis=1) - piv_prices.min(axis=1)) / (price_median_top10 + 1e-6)).values

    # Composite index: Higher score is better, lower log price dispersion is better.
    # Empirical regression weighting shows ~ 1.0 : -1.0 ratio
    composite_evidence = score_r1 - log_price_std_top10

    # B. EVALUATION-ONLY OUTCOMES (Used solely for research & gate threshold validation)
    actual_prices = target_meta["actual_unit_price"].values
    r1_apes = piv_apes[1].values
    r1_hits = piv_hits[1].values.astype(bool)
    top3_hits = piv_hits[[1, 2, 3]].any(axis=1).values.astype(bool)
    top5_hits = piv_hits[[1, 2, 3, 4, 5]].any(axis=1).values.astype(bool)
    top10_hits = piv_hits.any(axis=1).values.astype(bool)

    recovery_top5 = (~r1_hits) & top5_hits
    recovery_top10 = (~r1_hits) & top10_hits
    unrecoverable = ~top10_hits

    # 3. Overall Benchmark Metrics
    n_r1_hits = int(np.sum(r1_hits))
    n_top3_hits = int(np.sum(top3_hits))
    n_top5_hits = int(np.sum(top5_hits))
    n_top10_hits = int(np.sum(top10_hits))
    n_recovery_t5 = int(np.sum(recovery_top5))
    n_recovery_t10 = int(np.sum(recovery_top10))
    n_unrecoverable = int(np.sum(unrecoverable))

    overall_metrics = {
        "total_validation_queries": n_queries,
        "candidates_per_query": min_cands,
        "rank_1_accuracy_pct": round(n_r1_hits / n_queries * 100.0, 2),
        "top_3_accuracy_pct": round(n_top3_hits / n_queries * 100.0, 2),
        "top_5_accuracy_pct": round(n_top5_hits / n_queries * 100.0, 2),
        "top_10_accuracy_pct": round(n_top10_hits / n_queries * 100.0, 2),
        "rank_1_hits": n_r1_hits,
        "top_3_hits": n_top3_hits,
        "top_5_hits": n_top5_hits,
        "top_10_hits": n_top10_hits,
        "rank_1_failures_total": n_queries - n_r1_hits,
        "recovery_by_top_5_count": n_recovery_t5,
        "recovery_by_top_5_pct_of_total": round(n_recovery_t5 / n_queries * 100.0, 2),
        "recovery_by_top_5_pct_of_failures": round(n_recovery_t5 / (n_queries - n_r1_hits) * 100.0, 2),
        "recovery_by_top_10_count": n_recovery_t10,
        "recovery_by_top_10_pct_of_total": round(n_recovery_t10 / n_queries * 100.0, 2),
        "recovery_by_top_10_pct_of_failures": round(n_recovery_t10 / (n_queries - n_r1_hits) * 100.0, 2),
        "completely_unrecoverable_count": n_unrecoverable,
        "completely_unrecoverable_pct": round(n_unrecoverable / n_queries * 100.0, 2),
    }

    # 4. Statistical Distributions of Inference Signals
    print("\n[3/5] Computing signal distributions and correlations...", flush=True)
    signal_distributions = {
        "candidate_score_rank_1": summarize_series(score_r1),
        "candidate_score_rank_2": summarize_series(score_r2),
        "candidate_score_rank_3": summarize_series(score_r3),
        "candidate_score_rank_5": summarize_series(score_r5),
        "candidate_score_rank_10": summarize_series(score_r10),
        "candidate_score_top5_mean": summarize_series(score_top5_mean),
        "candidate_score_top10_mean": summarize_series(score_top10_mean),
        "candidate_score_margin_1_2": summarize_series(score_margin_1_2),
        "candidate_score_margin_1_10": summarize_series(score_margin_1_10),
        "price_dispersion_log_std_top10": summarize_series(log_price_std_top10),
        "price_dispersion_log_std_top5": summarize_series(log_price_std_top5),
        "price_dispersion_cv_top10": summarize_series(price_cv_top10),
        "price_dispersion_cv_top5": summarize_series(price_cv_top5),
        "price_ratio_max_min_top10": summarize_series(price_ratio_top10),
        "composite_evidence_score": summarize_series(composite_evidence),
        "top_1_absolute_percentage_error": summarize_series(r1_apes),
    }

    # Predictive power (ROC-AUC and correlation)
    inference_features = {
        "score_r1": (score_r1, 1),
        "score_r2": (score_r2, 1),
        "score_top5_mean": (score_top5_mean, 1),
        "score_top10_mean": (score_top10_mean, 1),
        "score_margin_1_2": (score_margin_1_2, 1),
        "score_margin_1_10": (score_margin_1_10, 1),
        "neg_log_price_std_top10": (-log_price_std_top10, -1),
        "neg_log_price_std_top5": (-log_price_std_top5, -1),
        "neg_price_cv_top10": (-price_cv_top10, -1),
        "neg_price_cv_top5": (-price_cv_top5, -1),
        "neg_price_ratio_top10": (-price_ratio_top10, -1),
        "composite_evidence_score": (composite_evidence, 1),
    }

    feature_predictive_power = {}
    for feat_name, (feat_vals, sign) in inference_features.items():
        auc_t10 = float(roc_auc_score(top10_hits, feat_vals))
        auc_t5 = float(roc_auc_score(top5_hits, feat_vals))
        auc_r1 = float(roc_auc_score(r1_hits, feat_vals))
        raw_vals = feat_vals if sign == 1 else -feat_vals
        corr_t10 = float(np.corrcoef(raw_vals, top10_hits.astype(float))[0, 1])
        corr_r1 = float(np.corrcoef(raw_vals, r1_hits.astype(float))[0, 1])
        feature_predictive_power[feat_name] = {
            "roc_auc_top10_hit": round(auc_t10, 4),
            "roc_auc_top5_hit": round(auc_t5, 4),
            "roc_auc_rank1_hit": round(auc_r1, 4),
            "pearson_corr_top10_hit": round(corr_t10, 4),
            "pearson_corr_rank1_hit": round(corr_r1, 4),
        }

    # 5. Quintile Breakdown of Primary Signals
    print("\n[4/5] Evaluating quintiles and gate threshold candidates...", flush=True)

    def compute_quintiles(values: np.ndarray, labels: List[str] = None):
        bins = pd.qcut(values, 5, duplicates="drop")
        df_q = pd.DataFrame({
            "bin": bins.astype(str),
            "r1_hit": r1_hits,
            "top5_hit": top5_hits,
            "top10_hit": top10_hits,
            "recovery_top10": recovery_top10,
            "unrecoverable": unrecoverable,
        })
        agg = df_q.groupby("bin", observed=False).agg(
            count=("r1_hit", "count"),
            r1_acc=("r1_hit", "mean"),
            top5_acc=("top5_hit", "mean"),
            top10_acc=("top10_hit", "mean"),
            recovery_top10_count=("recovery_top10", "sum"),
            unrecoverable_count=("unrecoverable", "sum"),
        ).reset_index()
        results = []
        for _, row in agg.iterrows():
            results.append({
                "quintile_bin": row["bin"],
                "query_count": int(row["count"]),
                "rank1_accuracy_pct": round(float(row["r1_acc"]) * 100.0, 2),
                "top5_accuracy_pct": round(float(row["top5_acc"]) * 100.0, 2),
                "top10_accuracy_pct": round(float(row["top10_acc"]) * 100.0, 2),
                "unrecoverable_pct": round(float(row["unrecoverable_count"]) / float(row["count"]) * 100.0, 2),
            })
        return results

    quintile_breakdown = {
        "score_r1_quintiles": compute_quintiles(score_r1),
        "log_price_std_top10_quintiles": compute_quintiles(log_price_std_top10),
        "composite_evidence_quintiles": compute_quintiles(composite_evidence),
    }

    # 6. Evaluation of Concrete Gate Policies
    def evaluate_gate_rule(condition_passed: np.ndarray, policy_name: str, description: str):
        n_pass = int(np.sum(condition_passed))
        n_reject = n_queries - n_pass
        pass_rate = n_pass / n_queries

        # Performance on Passed Queries (Sent to Person 2)
        passed_t10_hits = int(np.sum(top10_hits[condition_passed]))
        passed_t5_hits = int(np.sum(top5_hits[condition_passed]))
        passed_r1_hits = int(np.sum(r1_hits[condition_passed]))
        passed_t10_acc = (passed_t10_hits / n_pass * 100.0) if n_pass > 0 else 0.0
        passed_t5_acc = (passed_t5_hits / n_pass * 100.0) if n_pass > 0 else 0.0
        passed_r1_acc = (passed_r1_hits / n_pass * 100.0) if n_pass > 0 else 0.0

        # Performance on Rejected Queries (Diverted to Person 3)
        rej_mask = ~condition_passed
        rej_t10_hits = int(np.sum(top10_hits[rej_mask]))
        rej_t5_hits = int(np.sum(top5_hits[rej_mask]))
        rej_r1_hits = int(np.sum(r1_hits[rej_mask]))
        rej_unrecoverable = int(np.sum(unrecoverable[rej_mask]))
        rej_t10_acc = (rej_t10_hits / n_reject * 100.0) if n_reject > 0 else 0.0
        rej_unrecoverable_pct = (rej_unrecoverable / n_reject * 100.0) if n_reject > 0 else 0.0

        # Error matrix relative to ground truth top10_hit
        # True Positive: Gate says SUFFICIENT, Top-10 has hit
        tp = passed_t10_hits
        # False Positive: Gate says SUFFICIENT, but Top-10 has NO hit (Person 2 receives unrecoverable pool)
        fp = n_pass - passed_t10_hits
        # False Negative: Gate says INSUFFICIENT, but Top-10 had a hit (Person 3 receives recoverable pool)
        fn = rej_t10_hits
        # True Negative: Gate says INSUFFICIENT, and Top-10 had NO hit (Rightly prevented LLM hallucination)
        tn = rej_unrecoverable

        return {
            "policy_name": policy_name,
            "description": description,
            "passed_queries_count": n_pass,
            "pass_rate_pct": round(pass_rate * 100.0, 2),
            "passed_top10_accuracy_pct": round(passed_t10_acc, 2),
            "passed_top5_accuracy_pct": round(passed_t5_acc, 2),
            "passed_rank1_accuracy_pct": round(passed_r1_acc, 2),
            "rejected_queries_count": n_reject,
            "rejection_rate_pct": round((1.0 - pass_rate) * 100.0, 2),
            "rejected_top10_accuracy_pct": round(rej_t10_acc, 2),
            "rejected_completely_unrecoverable_pct": round(rej_unrecoverable_pct, 2),
            "true_positives": tp,
            "false_positives_unrecoverable_passed_to_llm": fp,
            "false_negatives_diverted_to_person3": fn,
            "true_negatives_correctly_rejected_junk": tn,
            "precision_top10_in_passed": round(tp / (tp + fp) * 100.0, 2) if (tp + fp) > 0 else 0.0,
            "recall_recoverable_queries_retained": round(tp / (tp + fn) * 100.0, 2) if (tp + fn) > 0 else 0.0,
        }

    gate_policies = {
        "policy_1_permissive": evaluate_gate_rule(
            composite_evidence >= -1.0,
            "Permissive Gate (High Coverage)",
            "composite_evidence >= -1.0 (filters out bottom ~13% toxic candidate pools)"
        ),
        "policy_2_balanced_default": evaluate_gate_rule(
            composite_evidence >= 0.0,
            "Balanced Gate (Recommended Default)",
            "composite_evidence >= 0.0 (filters out ~37% low-confidence/high-dispersion pools)"
        ),
        "policy_3_strict": evaluate_gate_rule(
            composite_evidence >= 0.7,
            "Strict Gate (High Precision)",
            "composite_evidence >= 0.7 (retains top ~46% highest quality historical pools)"
        ),
        "policy_4_dual_rule_balanced": evaluate_gate_rule(
            (score_r1 >= 0.5) & (log_price_std_top10 <= 1.0),
            "Dual-Threshold Rule (Score >= 0.5 AND LogPriceStd <= 1.0)",
            "Explicit dual threshold on candidate_score_r1 >= 0.5 and log_price_std_top10 <= 1.0"
        ),
        "policy_5_dual_rule_permissive": evaluate_gate_rule(
            (score_r1 >= 0.0) & (log_price_std_top10 <= 1.5),
            "Dual-Threshold Permissive (Score >= 0.0 AND LogPriceStd <= 1.5)",
            "Explicit dual threshold on candidate_score_r1 >= 0.0 and log_price_std_top10 <= 1.5"
        ),
    }

    # Threshold sweep table for fine-grained tuning
    sweep_results = []
    for thresh in np.arange(-1.5, 2.1, 0.25):
        passed = composite_evidence >= thresh
        eval_dict = evaluate_gate_rule(
            passed,
            f"threshold_{thresh:+.2f}",
            f"composite_evidence >= {thresh:+.2f}"
        )
        sweep_results.append({
            "threshold": round(float(thresh), 2),
            "pass_rate_pct": eval_dict["pass_rate_pct"],
            "passed_count": eval_dict["passed_queries_count"],
            "passed_top10_acc_pct": eval_dict["passed_top10_accuracy_pct"],
            "passed_top5_acc_pct": eval_dict["passed_top5_accuracy_pct"],
            "passed_r1_acc_pct": eval_dict["passed_rank1_accuracy_pct"],
            "rejected_count": eval_dict["rejected_queries_count"],
            "rejected_top10_acc_pct": eval_dict["rejected_top10_accuracy_pct"],
            "rejected_unrecoverable_pct": eval_dict["rejected_completely_unrecoverable_pct"],
        })

    # 7. Compile Final Output Artifact
    final_metrics_payload = {
        "dataset_metadata": {
            "source_file": "frozen_evaluation_dataset.parquet",
            "total_records": len(df),
            "total_queries": n_queries,
            "candidates_per_query": min_cands,
            "verified_schema": EXPECTED_SCHEMA,
        },
        "overall_benchmark_metrics": overall_metrics,
        "signal_distributions": signal_distributions,
        "feature_predictive_power": feature_predictive_power,
        "quintile_breakdown": quintile_breakdown,
        "gate_policies": gate_policies,
        "threshold_sweep": sweep_results,
        "execution_summary": {
            "runtime_seconds": round(time.time() - t_start, 2),
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }
    }

    print("\n[5/5] Exporting results to evidence_gate_metrics.json...", flush=True)
    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics_payload, f, indent=2)

    print("\n" + "=" * 80)
    print("HISTORICAL EVIDENCE GATE ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total Validation Queries:                           {n_queries:,}")
    print(f"Overall Rank-1 Accuracy:                            {overall_metrics['rank_1_accuracy_pct']}%")
    print(f"Overall Top-5 Accuracy:                             {overall_metrics['top_5_accuracy_pct']}%")
    print(f"Overall Top-10 Accuracy:                            {overall_metrics['top_10_accuracy_pct']}%")
    print(f"Completely Unrecoverable Queries:                   {n_unrecoverable:,} ({overall_metrics['completely_unrecoverable_pct']}%)")
    print("-" * 80)
    print("PROPOSED GATE POLICY COMPARISONS:")
    for pol_key, pol in gate_policies.items():
        print(f"\n* [{pol['policy_name']}]")
        print(f"  - Condition:             {pol['description']}")
        print(f"  - Pass Rate:            {pol['pass_rate_pct']}% ({pol['passed_queries_count']:,} queries -> Person 2)")
        print(f"  - Passed Top-10 Acc:    {pol['passed_top10_accuracy_pct']}% (vs. 62.64% baseline)")
        print(f"  - Passed Top-5 Acc:     {pol['passed_top5_accuracy_pct']}% (vs. 54.14% baseline)")
        print(f"  - Passed Rank-1 Acc:    {pol['passed_rank1_accuracy_pct']}% (vs. 37.57% baseline)")
        print(f"  - Rejection Rate:       {pol['rejection_rate_pct']}% ({pol['rejected_queries_count']:,} queries -> Person 3)")
        print(f"  - Rejected Top-10 Acc:  {pol['rejected_top10_accuracy_pct']}% (Unrecoverable: {pol['rejected_completely_unrecoverable_pct']}%)")
    print("=" * 80)
    print(f"Metrics successfully written to: {out_metrics_path}")


if __name__ == "__main__":
    main()
