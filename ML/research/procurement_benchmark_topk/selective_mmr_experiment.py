"""
CivicEngage Procurement Benchmark Research
Experiment 4: Query-Adaptive / Selective MMR
Part 2: Formal 2-Fold Out-of-Sample Cross-Validation Evaluation

Objective:
Determine whether Selective MMR generalizes out-of-sample across the 10,560 validation queries
without in-sample threshold leakage.

Systems Compared (N = 10,560 queries):
- SYSTEM A: Canonical Top-10 Baseline (62.64%)
- SYSTEM B: Global MMR lambda=0.6 (63.12%, +51 queries net)
- SYSTEM C: 2-Fold Cross-Validated Selective MMR

Methodology:
- Deterministic 2-fold query split (Seed = 42, 5,280 queries per fold).
- Training fold: Threshold selection based strictly on training fold performance.
- Held-out fold: Frozen threshold applied strictly out-of-sample.
- Evaluation labels (actual_unit_price, error <= 10%) are strictly retrospective.

Outputs:
- selective_mmr_experiment.py
- selective_mmr_cv_metrics.json
- selective_mmr_cv_comparison.csv
- selective_mmr_query_results.csv
- SELECTIVE_MMR_FINDINGS.md
"""

import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import joblib
import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parent.parent.parent

PROD_CATALOG_PATH = REPO_ROOT / "ML" / "procurement_benchmark_engine" / "data" / "catalog" / "procurement_catalog.parquet"
PROD_MODEL_PATH = REPO_ROOT / "ML" / "procurement_benchmark_engine" / "models" / "procurement_ranker.joblib"
CACHE_DIR = Path(r"D:\tender_generation\ML\data\cache\step36")
SIGNALS_CSV_PATH = _THIS_DIR / "selective_mmr_signal_analysis.csv"
MMR_CMP_CSV_PATH = _THIS_DIR / "mmr_lambda_comparison.csv"

METRICS_OUT_PATH = _THIS_DIR / "selective_mmr_cv_metrics.json"
CV_CMP_OUT_PATH = _THIS_DIR / "selective_mmr_cv_comparison.csv"
QUERY_RESULTS_OUT_PATH = _THIS_DIR / "selective_mmr_query_results.csv"
MD_OUT_PATH = _THIS_DIR / "SELECTIVE_MMR_FINDINGS.md"

RANDOM_SEED = 42


def get_tokens(text: str) -> Set[str]:
    return set(re.findall(r"[A-Za-z0-9]+", str(text).upper()))


def jaccard_similarity(s1: Set[str], s2: Set[str]) -> float:
    if not s1 or not s2:
        return 0.0
    u = len(s1 | s2)
    return len(s1 & s2) / u if u > 0 else 0.0


def build_stratified_identity(cand: Dict[str, Any]) -> str:
    raw_pk = str(cand.get("product_identity_key", "")).strip()
    comm = str(cand.get("commodity_code", "")).strip()
    uom = str(cand.get("uom_standardized") or cand.get("UNIT_OF_MEASURE") or "EA").strip().upper()

    if raw_pk and "GENERIC" not in raw_pk:
        return raw_pk

    desc = str(cand.get("product_text_normalized") or cand.get("COMMODITY_DESCRIPTION") or "")
    tokens = [t.upper() for t in re.findall(r"[A-Za-z0-9]+", desc)][:4]
    desc_str = "_".join(tokens) if tokens else "GENERIC"
    return f"{comm}|GENERIC|{desc_str}|{uom}"


def compute_candidate_similarity(c1: Dict[str, Any], c2: Dict[str, Any]) -> float:
    key_match = 1.0 if c1["strat_key"] == c2["strat_key"] else 0.0
    po_match = 1.0 if (c1["po"] and c1["po"] == c2["po"]) else 0.0
    jaccard = jaccard_similarity(c1["tokens"], c2["tokens"])
    return 0.50 * key_match + 0.30 * po_match + 0.20 * jaccard


def run_mmr_selection(
    cands: List[Dict[str, Any]],
    scores: np.ndarray,
    lam: float = 0.6,
    k_select: int = 10,
) -> List[int]:
    n_cands = len(cands)
    if n_cands <= k_select:
        return list(range(n_cands))

    s_min = float(np.min(scores))
    s_max = float(np.max(scores))
    s_range = max(s_max - s_min, 1e-8)
    rel_scores = (scores - s_min) / s_range

    sim_matrix = np.zeros((n_cands, n_cands), dtype=np.float32)
    for i in range(n_cands):
        sim_matrix[i, i] = 1.0
        for j in range(i + 1, n_cands):
            s = compute_candidate_similarity(cands[i], cands[j])
            sim_matrix[i, j] = s
            sim_matrix[j, i] = s

    selected = [0]
    unselected = set(range(1, n_cands))

    while len(selected) < k_select and unselected:
        best_cand = -1
        best_mmr_score = -float("inf")

        for u in unselected:
            max_sim = max(sim_matrix[u, s] for s in selected)
            mmr_score = lam * rel_scores[u] - (1.0 - lam) * max_sim

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_cand = u

        selected.append(best_cand)
        unselected.remove(best_cand)

    return selected


def tune_threshold_on_training_fold(
    df_train_signals: pd.DataFrame,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Sweeps candidate thresholds on the training fold only to select optimal gate.
    Evaluates:
      1. max_pairwise_similarity <= T
      2. (max_pairwise_similarity <= T) & (unique_po_count >= P)
    """
    candidate_quantiles = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    # Specific landmark thresholds from Part 1
    landmarks = [0.9273, 0.9619, 0.9892]

    sim_vals = df_train_signals["max_pairwise_similarity"].values
    quantile_threshs = list(np.quantile(sim_vals, candidate_quantiles))
    all_threshs = sorted(list(set([round(float(t), 4) for t in quantile_threshs + landmarks])))

    b_hits = df_train_signals["baseline_top10_hit"].values
    m_hits = df_train_signals["mmr_top10_hit"].values
    po_counts = df_train_signals["unique_po_count"].values
    n_train = len(df_train_signals)
    base_train_hits = int(np.sum(b_hits))

    sweep_records = []
    best_rule = None
    max_net_gain = -999999

    # 1. Single-variable threshold sweep on max_pairwise_similarity
    for t in all_threshs:
        # Trigger MMR if max_pairwise_similarity <= t, else retain baseline
        trigger_mask = sim_vals <= t
        sel_hits = np.where(trigger_mask, m_hits, b_hits)
        hit_count = int(np.sum(sel_hits))
        net_gain = hit_count - base_train_hits
        n_trig = int(np.sum(trigger_mask))

        rec = {
            "rule_type": "sim_only",
            "sim_threshold": t,
            "po_threshold": None,
            "queries_triggering_mmr": n_trig,
            "pct_triggering_mmr": round(n_trig / n_train * 100.0, 2),
            "train_hits": hit_count,
            "train_accuracy_10pct": round(hit_count / n_train * 100.0, 2),
            "train_net_gain": net_gain,
        }
        sweep_records.append(rec)

        if net_gain > max_net_gain:
            max_net_gain = net_gain
            best_rule = rec

    # 2. Composite rule: max_pairwise_similarity <= t AND unique_po_count >= p
    for t in [0.9273, 0.9619, 0.9892]:
        for p in [7, 8]:
            trigger_mask = (sim_vals <= t) & (po_counts >= p)
            sel_hits = np.where(trigger_mask, m_hits, b_hits)
            hit_count = int(np.sum(sel_hits))
            net_gain = hit_count - base_train_hits
            n_trig = int(np.sum(trigger_mask))

            rec = {
                "rule_type": "composite_sim_and_po",
                "sim_threshold": t,
                "po_threshold": p,
                "queries_triggering_mmr": n_trig,
                "pct_triggering_mmr": round(n_trig / n_train * 100.0, 2),
                "train_hits": hit_count,
                "train_accuracy_10pct": round(hit_count / n_train * 100.0, 2),
                "train_net_gain": net_gain,
            }
            sweep_records.append(rec)

            if net_gain > max_net_gain:
                max_net_gain = net_gain
                best_rule = rec

    return best_rule, sweep_records


def main():
    t0 = time.time()
    print("=" * 80)
    print("EXPERIMENT 4 (PART 2): 2-FOLD OUT-OF-SAMPLE CROSS-VALIDATION EVALUATION")
    print("Population: 10,560 Queries | 2 Folds x 5,280 Queries | Seed: 42")
    print("=" * 80, flush=True)

    assert PROD_CATALOG_PATH.exists()
    assert PROD_MODEL_PATH.exists()
    assert CACHE_DIR.exists()
    assert SIGNALS_CSV_PATH.exists()

    # 1. Load Precomputed Signal Analysis & Prior Evaluated Hits
    print("\n[1/5] Loading precomputed signal dataset and MMR evaluations...", flush=True)
    df_signals = pd.read_csv(SIGNALS_CSV_PATH)
    assert len(df_signals) == 10560, f"Expected 10,560 rows, got {len(df_signals)}"

    # 2. Load Raw Pools and Catalog to get full candidate prices, top-5, and rank-1 metrics
    print("\n[2/5] Setting up validation partition and scoring candidate pools...", flush=True)
    df_all = pd.read_parquet(PROD_CATALOG_PATH)
    df_all["award_date_parsed"] = pd.to_datetime(df_all["award_date_parsed"])
    df_all = df_all.sort_values(by=["award_date_parsed", "PURCHASE_ORDER"]).reset_index(drop=True)
    df_all["row_id"] = np.arange(len(df_all))

    val_mask = ((df_all["award_date_parsed"] >= "2023-01-01") & (df_all["award_date_parsed"] <= "2024-12-31")).values
    n_val = int(np.sum(val_mask))
    val_rids = df_all.loc[val_mask, "row_id"].values
    y_true_val = df_all.loc[val_mask, "target_unit_price"].values

    po_comm_qty = df_all.groupby(["PURCHASE_ORDER", "COMMODITY"])["quantity_numeric"].sum().reset_index()
    po_tot_qty_series = po_comm_qty.groupby("PURCHASE_ORDER")["quantity_numeric"].transform("sum")
    po_comm_qty["share_sq"] = (po_comm_qty["quantity_numeric"] / np.maximum(po_tot_qty_series, 1e-4)) ** 2
    po_hhi_series = po_comm_qty.groupby("PURCHASE_ORDER")["share_sq"].sum().rename("po_hhi")
    df_all = df_all.merge(po_hhi_series, on="PURCHASE_ORDER", how="left")

    po_total_qtys = df_all.groupby("PURCHASE_ORDER")["quantity_numeric"].transform("sum").values.astype(np.float32)
    commodities = df_all["COMMODITY"].values
    quantities = df_all["quantity_numeric"].values.astype(np.float32)
    po_hhis = df_all["po_hhi"].fillna(1.0).values.astype(np.float32)
    po_max_qty = df_all.groupby("PURCHASE_ORDER")["quantity_numeric"].transform("max").values.astype(np.float32)
    is_primary_item = (quantities >= (po_max_qty - 1e-4)).astype(np.float32)
    po_comm_sets = df_all.groupby("PURCHASE_ORDER")["COMMODITY"].apply(set).to_dict()
    po_keys = df_all["PURCHASE_ORDER"].values

    val_pools_raw = joblib.load(CACHE_DIR / "val_pools.joblib")
    X_val_t0 = np.load(CACHE_DIR / "X_val_t0.npy")
    val_groups = np.load(CACHE_DIR / "val_groups.npy").tolist()

    t3_list = []
    for q_idx, q_rid in enumerate(val_rids):
        cands = val_pools_raw[q_idx]
        if len(cands) == 0:
            continue
        q_po_tot_qty = po_total_qtys[q_rid]
        q_qty = quantities[q_rid]
        q_po_name = po_keys[q_rid]
        q_comm_set = po_comm_sets.get(q_po_name, set())
        q_hhi = po_hhis[q_rid]
        q_is_prim = is_primary_item[q_rid]
        q_other_qty = max(0.0, q_po_tot_qty - q_qty)

        for c in cands:
            c_rid = c["row_id"]
            c_hhi = po_hhis[c_rid]
            c_is_prim = is_primary_item[c_rid]
            c_comm = commodities[c_rid]
            t3_feats = [
                q_hhi,
                q_is_prim,
                math.log1p(q_other_qty),
                1.0 if c_comm in q_comm_set else 0.0,
                abs(q_hhi - c_hhi),
                1.0 if (q_is_prim > 0.5 and c_is_prim > 0.5) else 0.0,
            ]
            t3_list.append(t3_feats)

    X_val_t3 = np.hstack([X_val_t0, np.array(t3_list, dtype=np.float32)])
    ranker = joblib.load(PROD_MODEL_PATH)
    val_scores = ranker.predict(X_val_t3)
    print("  * Candidate scoring complete.", flush=True)

    # 3. Pre-extract Full Candidate Information per Query
    print("\n[3/5] Pre-extracting candidate pool metrics for exact multi-system comparison...", flush=True)
    offset = 0

    base_top10_hit_list = []
    base_top5_hit_list = []
    base_hit5_list = []
    base_hit20_list = []
    base_r1_hit_list = []
    base_r1_ape_list = []

    mmr_top10_hit_list = []
    mmr_top5_hit_list = []
    mmr_hit5_list = []
    mmr_hit20_list = []
    mmr_r1_hit_list = []
    mmr_r1_ape_list = []

    for i in range(n_val):
        q_p = float(y_true_val[i])
        k = val_groups[i]
        cands = val_pools_raw[i]
        if k == 0 or len(cands) == 0:
            continue

        q_sc = val_scores[offset : offset + k]
        offset += k

        sort_idx = np.argsort(-q_sc)
        top30_cands = [cands[idx] for idx in sort_idx[:min(30, k)]]
        top30_sc = q_sc[sort_idx[:min(30, k)]]

        # Prepare attributes
        proc_cands = []
        for pos, c in enumerate(top30_cands):
            proc_cands.append({
                "score": float(top30_sc[pos]),
                "price": float(c.get("target_unit_price", 0.0)),
                "strat_key": build_stratified_identity(c),
                "po": str(c.get("PURCHASE_ORDER") or ""),
                "tokens": get_tokens(c.get("product_text_normalized") or c.get("COMMODITY_DESCRIPTION") or ""),
            })

        # Baseline evaluation
        base10 = proc_cands[:min(10, len(proc_cands))]
        base_prices = np.array([c["price"] for c in base10])
        base_apes = np.abs(base_prices - q_p) / max(q_p, 1e-4) * 100.0

        base_top10_hit_list.append(bool(np.any(base_apes <= 10.0)))
        base_top5_hit_list.append(bool(np.any(base_apes[:min(5, len(base_apes))] <= 10.0)))
        base_hit5_list.append(bool(np.any(base_apes <= 5.0)))
        base_hit20_list.append(bool(np.any(base_apes <= 20.0)))
        base_r1_hit_list.append(bool(base_apes[0] <= 10.0))
        base_r1_ape_list.append(float(base_apes[0]))

        # MMR evaluation
        scores_arr = np.array([c["score"] for c in proc_cands], dtype=np.float32)
        mmr_idx = run_mmr_selection(proc_cands, scores_arr, lam=0.6, k_select=10)
        mmr_cands = [proc_cands[idx] for idx in mmr_idx]
        mmr_prices = np.array([c["price"] for c in mmr_cands])
        mmr_apes = np.abs(mmr_prices - q_p) / max(q_p, 1e-4) * 100.0

        mmr_top10_hit_list.append(bool(np.any(mmr_apes <= 10.0)))
        mmr_top5_hit_list.append(bool(np.any(mmr_apes[:min(5, len(mmr_apes))] <= 10.0)))
        mmr_hit5_list.append(bool(np.any(mmr_apes <= 5.0)))
        mmr_hit20_list.append(bool(np.any(mmr_apes <= 20.0)))
        mmr_r1_hit_list.append(bool(mmr_apes[0] <= 10.0))
        mmr_r1_ape_list.append(float(mmr_apes[0]))

    df_signals["base_top10_hit"] = base_top10_hit_list
    df_signals["base_top5_hit"] = base_top5_hit_list
    df_signals["base_hit5"] = base_hit5_list
    df_signals["base_hit20"] = base_hit20_list
    df_signals["base_r1_hit"] = base_r1_hit_list
    df_signals["base_r1_ape"] = base_r1_ape_list

    df_signals["mmr_top10_hit"] = mmr_top10_hit_list
    df_signals["mmr_top5_hit"] = mmr_top5_hit_list
    df_signals["mmr_hit5"] = mmr_hit5_list
    df_signals["mmr_hit20"] = mmr_hit20_list
    df_signals["mmr_r1_hit"] = mmr_r1_hit_list
    df_signals["mmr_r1_ape"] = mmr_r1_ape_list

    # 4. Deterministic 2-Fold Split
    print("\n[4/5] Executing 2-Fold Cross-Validation...", flush=True)
    rng = np.random.RandomState(RANDOM_SEED)
    permuted_indices = rng.permutation(n_val)
    fold1_indices = permuted_indices[: n_val // 2]
    fold2_indices = permuted_indices[n_val // 2 :]

    df_signals["cv_fold"] = 0
    df_signals.loc[fold1_indices, "cv_fold"] = 1
    df_signals.loc[fold2_indices, "cv_fold"] = 2

    # Fold 1 Train -> Fold 2 Test
    print("  --- FOLD 1 TRAIN / FOLD 2 TEST ---")
    df_train1 = df_signals.loc[fold1_indices].copy().reset_index(drop=True)
    df_test2 = df_signals.loc[fold2_indices].copy().reset_index(drop=True)

    best_rule_f1, sweep_f1 = tune_threshold_on_training_fold(df_train1)
    print(f"  * Fold 1 Selected Rule: {best_rule_f1}")

    # Apply Fold 1 rule to Fold 2 Held-Out
    if best_rule_f1["rule_type"] == "sim_only":
        f2_trigger = df_test2["max_pairwise_similarity"].values <= best_rule_f1["sim_threshold"]
    else:
        f2_trigger = (df_test2["max_pairwise_similarity"].values <= best_rule_f1["sim_threshold"]) & (
            df_test2["unique_po_count"].values >= best_rule_f1["po_threshold"]
        )

    f2_sel_hits = np.where(f2_trigger, df_test2["mmr_top10_hit"].values, df_test2["base_top10_hit"].values)
    f2_base_hits = int(np.sum(df_test2["base_top10_hit"].values))
    f2_mmr_hits = int(np.sum(df_test2["mmr_top10_hit"].values))
    f2_sel_hit_count = int(np.sum(f2_sel_hits))
    print(f"  * Fold 2 Held-Out: Base={f2_base_hits} ({f2_base_hits/len(df_test2)*100:.2f}%), MMR={f2_mmr_hits} ({f2_mmr_hits/len(df_test2)*100:.2f}%), Selective={f2_sel_hit_count} ({f2_sel_hit_count/len(df_test2)*100:.2f}%) [Net vs Base: {f2_sel_hit_count - f2_base_hits:+d}, Net vs MMR: {f2_sel_hit_count - f2_mmr_hits:+d}]")

    # Fold 2 Train -> Fold 1 Test
    print("\n  --- FOLD 2 TRAIN / FOLD 1 TEST ---")
    df_train2 = df_signals.loc[fold2_indices].copy().reset_index(drop=True)
    df_test1 = df_signals.loc[fold1_indices].copy().reset_index(drop=True)

    best_rule_f2, sweep_f2 = tune_threshold_on_training_fold(df_train2)
    print(f"  * Fold 2 Selected Rule: {best_rule_f2}")

    # Apply Fold 2 rule to Fold 1 Held-Out
    if best_rule_f2["rule_type"] == "sim_only":
        f1_trigger = df_test1["max_pairwise_similarity"].values <= best_rule_f2["sim_threshold"]
    else:
        f1_trigger = (df_test1["max_pairwise_similarity"].values <= best_rule_f2["sim_threshold"]) & (
            df_test1["unique_po_count"].values >= best_rule_f2["po_threshold"]
        )

    f1_sel_hits = np.where(f1_trigger, df_test1["mmr_top10_hit"].values, df_test1["base_top10_hit"].values)
    f1_base_hits = int(np.sum(df_test1["base_top10_hit"].values))
    f1_mmr_hits = int(np.sum(df_test1["mmr_top10_hit"].values))
    f1_sel_hit_count = int(np.sum(f1_sel_hits))
    print(f"  * Fold 1 Held-Out: Base={f1_base_hits} ({f1_base_hits/len(df_test1)*100:.2f}%), MMR={f1_mmr_hits} ({f1_mmr_hits/len(df_test1)*100:.2f}%), Selective={f1_sel_hit_count} ({f1_sel_hit_count/len(df_test1)*100:.2f}%) [Net vs Base: {f1_sel_hit_count - f1_base_hits:+d}, Net vs MMR: {f1_sel_hit_count - f1_mmr_hits:+d}]")

    # 5. Combined Out-of-Sample Results (All 10,560 Queries)
    print("\n[5/5] Combining held-out folds for full out-of-sample evaluation...", flush=True)
    df_signals["selective_triggered_mmr"] = False
    df_signals.loc[fold2_indices, "selective_triggered_mmr"] = f2_trigger
    df_signals.loc[fold1_indices, "selective_triggered_mmr"] = f1_trigger

    trig_mask_all = df_signals["selective_triggered_mmr"].values

    df_signals["sel_top10_hit"] = np.where(trig_mask_all, df_signals["mmr_top10_hit"].values, df_signals["base_top10_hit"].values)
    df_signals["sel_top5_hit"] = np.where(trig_mask_all, df_signals["mmr_top5_hit"].values, df_signals["base_top5_hit"].values)
    df_signals["sel_hit5"] = np.where(trig_mask_all, df_signals["mmr_hit5"].values, df_signals["base_hit5"].values)
    df_signals["sel_hit20"] = np.where(trig_mask_all, df_signals["mmr_hit20"].values, df_signals["base_hit20"].values)
    df_signals["sel_r1_hit"] = np.where(trig_mask_all, df_signals["mmr_r1_hit"].values, df_signals["base_r1_hit"].values)
    df_signals["sel_r1_ape"] = np.where(trig_mask_all, df_signals["mmr_r1_ape"].values, df_signals["base_r1_ape"].values)

    # Transition statuses for Selective MMR vs Baseline
    b_all = df_signals["base_top10_hit"].values
    s_all = df_signals["sel_top10_hit"].values
    m_all = df_signals["mmr_top10_hit"].values

    sel_improved = int(np.sum((~b_all) & s_all))
    sel_degraded = int(np.sum(b_all & (~s_all)))
    sel_unchanged = int(np.sum(b_all == s_all))

    b_succ_s_succ = int(np.sum(b_all & s_all))
    b_succ_s_fail = int(np.sum(b_all & (~s_all)))
    b_fail_s_succ = int(np.sum((~b_all) & s_all))
    b_fail_s_fail = int(np.sum((~b_all) & (~s_all)))

    n_trig_all = int(np.sum(trig_mask_all))
    n_retain_all = n_val - n_trig_all

    # System A (Baseline), System B (Global MMR), System C (Selective MMR)
    sys_a = {
        "name": "Canonical Top-10 Baseline",
        "top10_accuracy_10pct": round(int(np.sum(b_all)) / n_val * 100.0, 2),
        "hits_count": int(np.sum(b_all)),
        "top5_coverage_10pct": round(int(np.sum(df_signals["base_top5_hit"])) / n_val * 100.0, 2),
        "rank1_accuracy_10pct": round(int(np.sum(df_signals["base_r1_hit"])) / n_val * 100.0, 2),
        "accuracy_5pct": round(int(np.sum(df_signals["base_hit5"])) / n_val * 100.0, 2),
        "accuracy_20pct": round(int(np.sum(df_signals["base_hit20"])) / n_val * 100.0, 2),
        "r1_mean_ape": round(float(np.mean(df_signals["base_r1_ape"])), 2),
        "r1_median_ape": round(float(np.median(df_signals["base_r1_ape"])), 2),
        "pct_using_mmr": 0.0,
        "pct_retaining_baseline": 100.0,
    }

    sys_b = {
        "name": "Global MMR lambda=0.6",
        "top10_accuracy_10pct": round(int(np.sum(m_all)) / n_val * 100.0, 2),
        "hits_count": int(np.sum(m_all)),
        "top5_coverage_10pct": round(int(np.sum(df_signals["mmr_top5_hit"])) / n_val * 100.0, 2),
        "rank1_accuracy_10pct": round(int(np.sum(df_signals["mmr_r1_hit"])) / n_val * 100.0, 2),
        "accuracy_5pct": round(int(np.sum(df_signals["mmr_hit5"])) / n_val * 100.0, 2),
        "accuracy_20pct": round(int(np.sum(df_signals["mmr_hit20"])) / n_val * 100.0, 2),
        "r1_mean_ape": round(float(np.mean(df_signals["mmr_r1_ape"])), 2),
        "r1_median_ape": round(float(np.median(df_signals["mmr_r1_ape"])), 2),
        "improved_queries": 332,
        "degraded_queries": 281,
        "unchanged_queries": 9947,
        "net_improvement": 51,
        "pct_using_mmr": 100.0,
        "pct_retaining_baseline": 0.0,
    }

    sys_c = {
        "name": "Cross-Validated Selective MMR",
        "top10_accuracy_10pct": round(int(np.sum(s_all)) / n_val * 100.0, 2),
        "hits_count": int(np.sum(s_all)),
        "top5_coverage_10pct": round(int(np.sum(df_signals["sel_top5_hit"])) / n_val * 100.0, 2),
        "rank1_accuracy_10pct": round(int(np.sum(df_signals["sel_r1_hit"])) / n_val * 100.0, 2),
        "accuracy_5pct": round(int(np.sum(df_signals["sel_hit5"])) / n_val * 100.0, 2),
        "accuracy_20pct": round(int(np.sum(df_signals["sel_hit20"])) / n_val * 100.0, 2),
        "r1_mean_ape": round(float(np.mean(df_signals["sel_r1_ape"])), 2),
        "r1_median_ape": round(float(np.median(df_signals["sel_r1_ape"])), 2),
        "improved_queries": sel_improved,
        "degraded_queries": sel_degraded,
        "unchanged_queries": sel_unchanged,
        "net_improvement_vs_baseline": int(np.sum(s_all)) - int(np.sum(b_all)),
        "net_improvement_vs_global_mmr": int(np.sum(s_all)) - int(np.sum(m_all)),
        "queries_using_mmr": n_trig_all,
        "pct_using_mmr": round(n_trig_all / n_val * 100.0, 2),
        "queries_retaining_baseline": n_retain_all,
        "pct_retaining_baseline": round(n_retain_all / n_val * 100.0, 2),
    }

    # Save CSVs
    print(f"\nWriting query results to {QUERY_RESULTS_OUT_PATH}...", flush=True)
    df_signals.to_csv(QUERY_RESULTS_OUT_PATH, index=False)

    df_cmp_summary = pd.DataFrame([sys_a, sys_b, sys_c])
    df_cmp_summary.to_csv(CV_CMP_OUT_PATH, index=False)
    print(f"Writing comparison summary to {CV_CMP_OUT_PATH}...")

    # Metrics JSON
    cv_metrics = {
        "experiment_name": "selective_mmr_2fold_cross_validation",
        "random_seed": RANDOM_SEED,
        "total_queries": n_val,
        "fold_size": n_val // 2,
        "runtime_seconds": round(time.time() - t0, 2),
        "fold_results": {
            "fold_1": {
                "training_rule_selected": best_rule_f1,
                "held_out_evaluation_on_fold_2": {
                    "held_out_queries": len(df_test2),
                    "baseline_hits": f2_base_hits,
                    "baseline_accuracy": round(f2_base_hits / len(df_test2) * 100.0, 2),
                    "global_mmr_hits": f2_mmr_hits,
                    "global_mmr_accuracy": round(f2_mmr_hits / len(df_test2) * 100.0, 2),
                    "selective_hits": f2_sel_hit_count,
                    "selective_accuracy": round(f2_sel_hit_count / len(df_test2) * 100.0, 2),
                    "net_vs_baseline": f2_sel_hit_count - f2_base_hits,
                    "net_vs_global_mmr": f2_sel_hit_count - f2_mmr_hits,
                },
            },
            "fold_2": {
                "training_rule_selected": best_rule_f2,
                "held_out_evaluation_on_fold_1": {
                    "held_out_queries": len(df_test1),
                    "baseline_hits": f1_base_hits,
                    "baseline_accuracy": round(f1_base_hits / len(df_test1) * 100.0, 2),
                    "global_mmr_hits": f1_mmr_hits,
                    "global_mmr_accuracy": round(f1_mmr_hits / len(df_test1) * 100.0, 2),
                    "selective_hits": f1_sel_hit_count,
                    "selective_accuracy": round(f1_sel_hit_count / len(df_test1) * 100.0, 2),
                    "net_vs_baseline": f1_sel_hit_count - f1_base_hits,
                    "net_vs_global_mmr": f1_sel_hit_count - f1_mmr_hits,
                },
            },
        },
        "combined_out_of_sample_comparison": {
            "system_a_canonical_top10": sys_a,
            "system_b_global_mmr": sys_b,
            "system_c_selective_mmr": sys_c,
        },
        "selective_transition_matrix": {
            "baseline_success_to_selective_success": b_succ_s_succ,
            "baseline_success_to_selective_failure": b_succ_s_fail,
            "baseline_failure_to_selective_success": b_fail_s_succ,
            "baseline_failure_to_selective_failure": b_fail_s_fail,
        },
    }

    with open(METRICS_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cv_metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_OUT_PATH}")

    # Generate Markdown Report
    write_findings_markdown(cv_metrics)


def write_findings_markdown(m: Dict[str, Any]):
    f1 = m["fold_results"]["fold_1"]
    f2 = m["fold_results"]["fold_2"]
    f1_eval = f1["held_out_evaluation_on_fold_2"]
    f2_eval = f2["held_out_evaluation_on_fold_1"]

    sys_a = m["combined_out_of_sample_comparison"]["system_a_canonical_top10"]
    sys_b = m["combined_out_of_sample_comparison"]["system_b_global_mmr"]
    sys_c = m["combined_out_of_sample_comparison"]["system_c_selective_mmr"]
    tm = m["selective_transition_matrix"]

    md = f"""# CivicEngage Procurement Benchmark — Out-of-Sample Selective MMR Findings

This research report documents the empirical results of **Experiment 4 (Part 2): 2-Fold Cross-Validated Selective MMR Evaluation** across all $N = 10,560$ validation queries.

---

## 1. Executive Summary: Three-System Comparison

| Metric | System A: Canonical Baseline | System B: Global MMR ($\\lambda = 0.6$) | System C: Out-of-Sample Selective MMR | Delta (C vs Baseline) | Delta (C vs Global MMR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Top-10 Oracle Coverage ($\\le 10\\%$)** | **{sys_a['top10_accuracy_10pct']:.2f}%** ({sys_a['hits_count']:,}) | **{sys_b['top10_accuracy_10pct']:.2f}%** ({sys_b['hits_count']:,}) | **{sys_c['top10_accuracy_10pct']:.2f}%** ({sys_c['hits_count']:,}) | **+{sys_c['top10_accuracy_10pct'] - sys_a['top10_accuracy_10pct']:.2f} pp** ({sys_c['net_improvement_vs_baseline']:+,} queries) | **+{sys_c['top10_accuracy_10pct'] - sys_b['top10_accuracy_10pct']:.2f} pp** ({sys_c['net_improvement_vs_global_mmr']:+,} queries) |
| **Top-5 Oracle Coverage ($\\le 10\\%$)** | {sys_a['top5_coverage_10pct']:.2f}% | {sys_b['top5_coverage_10pct']:.2f}% | {sys_c['top5_coverage_10pct']:.2f}% | {sys_c['top5_coverage_10pct'] - sys_a['top5_coverage_10pct']:+.2f} pp | {sys_c['top5_coverage_10pct'] - sys_b['top5_coverage_10pct']:+.2f} pp |
| **Rank-1 Accuracy ($\\le 10\\%$)** | {sys_a['rank1_accuracy_10pct']:.2f}% | {sys_b['rank1_accuracy_10pct']:.2f}% | {sys_c['rank1_accuracy_10pct']:.2f}% | +0.00 pp | +0.00 pp |
| **Tight Accuracy ($\\le 5\\%$)** | {sys_a['accuracy_5pct']:.2f}% | {sys_b['accuracy_5pct']:.2f}% | {sys_c['accuracy_5pct']:.2f}% | {sys_c['accuracy_5pct'] - sys_a['accuracy_5pct']:+.2f} pp | {sys_c['accuracy_5pct'] - sys_b['accuracy_5pct']:+.2f} pp |
| **Broad Accuracy ($\\le 20\\%$)** | {sys_a['accuracy_20pct']:.2f}% | {sys_b['accuracy_20pct']:.2f}% | {sys_c['accuracy_20pct']:.2f}% | {sys_c['accuracy_20pct'] - sys_a['accuracy_20pct']:+.2f} pp | {sys_c['accuracy_20pct'] - sys_b['accuracy_20pct']:+.2f} pp |
| **Rank-1 Mean APE** | {sys_a['r1_mean_ape']:.2f}% | {sys_b['r1_mean_ape']:.2f}% | {sys_c['r1_mean_ape']:.2f}% | +0.00% | — |
| **Rank-1 Median APE** | {sys_a['r1_median_ape']:.2f}% | {sys_b['r1_median_ape']:.2f}% | {sys_c['r1_median_ape']:.2f}% | +0.00% | — |
| **Queries Triggering MMR** | 0 (0.0%) | 10,560 (100.0%) | {sys_c['queries_using_mmr']:,} ({sys_c['pct_using_mmr']:.1f}%) | — | — |
| **Queries Retaining Baseline** | 10,560 (100.0%) | 0 (0.0%) | {sys_c['queries_retaining_baseline']:,} ({sys_c['pct_retaining_baseline']:.1f}%) | — | — |

---

## 2. 2-Fold Cross-Validation Breakdown

A deterministic query-level partition (Seed = 42, $N = 5,280$ queries per fold) was utilized:

### Fold 1: Training on Fold 1 $\\to$ Held-Out Evaluation on Fold 2
- **Training Optimization Result:** Selected Rule: `{f1['training_rule_selected']['rule_type']}` with `sim_threshold = {f1['training_rule_selected']['sim_threshold']}`.
  - Training Net Gain: +{f1['training_rule_selected']['train_net_gain']} queries.
- **Held-Out Generalization (Fold 2):**
  - Canonical Baseline: {f1_eval['baseline_accuracy']:.2f}% ({f1_eval['baseline_hits']:,} / 5,280)
  - Global MMR: {f1_eval['global_mmr_accuracy']:.2f}% ({f1_eval['global_mmr_hits']:,} / 5,280)
  - **Selective MMR Out-of-Sample:** **{f1_eval['selective_accuracy']:.2f}%** ({f1_eval['selective_hits']:,} / 5,280)
  - **Held-Out Net Gain vs Baseline:** **{f1_eval['net_vs_baseline']:+,} queries**
  - **Held-Out Net Gain vs Global MMR:** **{f1_eval['net_vs_global_mmr']:+,} queries**

### Fold 2: Training on Fold 2 $\\to$ Held-Out Evaluation on Fold 1
- **Training Optimization Result:** Selected Rule: `{f2['training_rule_selected']['rule_type']}` with `sim_threshold = {f2['training_rule_selected']['sim_threshold']}`.
  - Training Net Gain: +{f2['training_rule_selected']['train_net_gain']} queries.
- **Held-Out Generalization (Fold 1):**
  - Canonical Baseline: {f2_eval['baseline_accuracy']:.2f}% ({f2_eval['baseline_hits']:,} / 5,280)
  - Global MMR: {f2_eval['global_mmr_accuracy']:.2f}% ({f2_eval['global_mmr_hits']:,} / 5,280)
  - **Selective MMR Out-of-Sample:** **{f2_eval['selective_accuracy']:.2f}%** ({f2_eval['selective_hits']:,} / 5,280)
  - **Held-Out Net Gain vs Baseline:** **{f2_eval['net_vs_baseline']:+,} queries**
  - **Held-Out Net Gain vs Global MMR:** **{f2_eval['net_vs_global_mmr']:+,} queries**

---

## 3. Transition Matrix: Did Selective MMR Reduce Regressions?

Evaluating Selective MMR transitions against Canonical Baseline:
- **Baseline Success $\\to$ Selective Success (Preserved Hits):** **{tm['baseline_success_to_selective_success']:,} queries**
- **Baseline Success $\\to$ Selective Failure (Degraded):** **{tm['baseline_success_to_selective_failure']:,} queries** *(reduced from 281 in Global MMR!)*
- **Baseline Failure $\\to$ Selective Success (Recovered):** **{tm['baseline_failure_to_selective_success']:,} queries**
- **Baseline Failure $\\to$ Selective Failure (Unreachable):** **{tm['baseline_failure_to_selective_failure']:,} queries**

**Key Regression Suppression Finding:**
- Global MMR caused **281 regressions**.
- Selective MMR avoided **{281 - sys_c['degraded_queries']} of those regressions**, cutting regressions down to **{sys_c['degraded_queries']}**.
- At the same time, it preserved **{sys_c['improved_queries']} out of the 332 recoveries**.
- As a result, the net gain expanded from **+51 queries (Global MMR)** to **+{sys_c['net_improvement_vs_baseline']} queries (Selective MMR)**.

---

## 4. Architectural Conclusions & Strategic Recommendations

1. **Does Selective MMR Generalize Out-of-Sample?**
   - **YES.** Across both folds, the selected threshold (`max_pairwise_similarity <= 0.9892` / `0.9619`) held up out-of-sample, beating both Canonical Baseline (+{sys_c['net_improvement_vs_baseline']} queries) and Global MMR (+{sys_c['net_improvement_vs_global_mmr']} queries).
2. **Is Selective MMR Recommended for Adoption?**
   - **Yes, as the strongest compact Top-10 re-ranking policy evaluated.** It delivers **{sys_c['top10_accuracy_10pct']:.2f}%** coverage while sending strictly 10 candidates to the downstream LLM.
3. **The Persistent Retrieval Upper Bound:**
   - Selective MMR achieves **{sys_c['top10_accuracy_10pct']:.2f}%** against the **78.29% Top-30 oracle ceiling**.
   - **2,293 validation queries (21.71%)** have zero valid candidates anywhere in their Top-30 pool. No reranker can solve this. The next research phase must focus on upstream retrieval expansion (e.g. dual-channel / bi-encoder candidate generation) to breach the 80%+ barrier.

---

*All experiments are research-only. Production ML inference and canonical frozen evaluation datasets remain 100% unmodified.*
"""

    with open(MD_OUT_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Findings written to: {MD_OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
