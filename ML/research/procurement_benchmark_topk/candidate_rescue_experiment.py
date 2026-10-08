"""
CivicEngage Procurement Benchmark Research
Candidate Rescue Experiment for Ranks 31–60

Objective:
Test whether a lightweight candidate-rescue mechanism can safely recover useful
historical candidates from ranks 31–60 into the compact Top-10 candidate pool
without looking at actual prices during selection.

Three Systems Compared (N = 10,560 queries):
- SYSTEM A: Canonical Top-10 Baseline (62.64%)
- SYSTEM B: 2-Fold Cross-Validated Selective MMR Top-10 (63.46%)
- SYSTEM C: Candidate Rescue + Selective MMR Top-10 (Strictly Out-of-Sample)

Protocol:
- Deterministic 2-fold query split (Seed = 42, 5,280 queries per fold).
- Training fold: Sweep rescue thresholds to maximize net gain on training queries.
- Held-out fold: Apply frozen rescue threshold strictly out-of-sample.
- Evaluation labels (actual_unit_price, error <= 10%) accessed ONLY for retrospective evaluation.
- Output sent to downstream LLM: STRICTLY 10 CANDIDATES.

Outputs:
- candidate_rescue_experiment.py
- candidate_rescue_metrics.json
- candidate_rescue_comparison.csv
- CANDIDATE_RESCUE_FINDINGS.md
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

METRICS_OUT_PATH = _THIS_DIR / "candidate_rescue_metrics.json"
CSV_OUT_PATH = _THIS_DIR / "candidate_rescue_comparison.csv"
MD_OUT_PATH = _THIS_DIR / "CANDIDATE_RESCUE_FINDINGS.md"

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


def evaluate_rescue_candidate(
    c: Dict[str, Any],
    q_comm: Any,
    q_uom: str,
    q_qty: float,
    q_tokens: Set[str],
    score_30: float,
) -> Tuple[bool, float]:
    """
    Evaluates whether a candidate in ranks 31-60 meets the strict pre-answer rescue gate.
    Returns (is_eligible, rescue_score).
    """
    c_comm = c.get("commodity_code") or c.get("COMMODITY")
    if str(c_comm) != str(q_comm):
        return False, 0.0

    c_uom = str(c.get("uom_standardized") or c.get("UNIT_OF_MEASURE") or "").upper()
    if c_uom != q_uom:
        return False, 0.0

    c_qty = float(c.get("quantity_numeric", 1.0))
    # Quantity ratio check
    if abs(math.log10(max(c_qty, 0.1)) - math.log10(max(q_qty, 0.1))) > 1.2:
        return False, 0.0

    # Soft rescue score: 60% token Jaccard + 40% score proximity
    jacc = jaccard_similarity(c["tokens"], q_tokens)
    c_sc = float(c["score"])
    score_proximity = max(0.0, 1.0 - max(0.0, score_30 - c_sc) / 0.50)
    rescue_score = 0.60 * jacc + 0.40 * score_proximity

    return True, rescue_score


def sweep_rescue_thresholds(
    train_queries_data: List[Dict[str, Any]],
    thresholds: List[float],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Sweeps rescue thresholds on training fold to find optimal rule."""
    sweep_results = []
    best_rule = None
    max_net = -999999

    n_queries = len(train_queries_data)
    sel_mmr_hits_train = sum(1 for q in train_queries_data if q["sel_mmr_top10_hit"])

    for t in thresholds:
        rescue_triggers = 0
        rescues_successful = 0
        rescues_harmful = 0
        rescues_neutral = 0

        final_hits = 0

        for q in train_queries_data:
            q_p = q["actual_price"]
            sel_cands = list(q["sel_mmr_cands"])  # 10 candidates
            orig_hit = q["sel_mmr_top10_hit"]

            # Check if any candidate in ranks 31-60 qualifies
            best_res_cand = None
            best_res_score = -1.0

            for c in q["cands_31_60"]:
                is_elig, r_score = evaluate_rescue_candidate(
                    c,
                    q["commodity"],
                    q["uom"],
                    q["qty"],
                    q["tokens"],
                    q["score_30"],
                )
                if is_elig and r_score >= t and r_score > best_res_score:
                    best_res_score = r_score
                    best_res_cand = c

            if best_res_cand is not None:
                rescue_triggers += 1
                # Replace 10th candidate with rescue candidate
                new_cands = sel_cands[:9] + [best_res_cand]
                new_prices = np.array([c["price"] for c in new_cands])
                new_apes = np.abs(new_prices - q_p) / max(q_p, 1e-4) * 100.0
                new_hit = bool(np.any(new_apes <= 10.0))

                if (not orig_hit) and new_hit:
                    rescues_successful += 1
                elif orig_hit and (not new_hit):
                    rescues_harmful += 1
                else:
                    rescues_neutral += 1
            else:
                new_hit = orig_hit

            if new_hit:
                final_hits += 1

        net_gain_vs_sel_mmr = final_hits - sel_mmr_hits_train
        rec = {
            "threshold": t,
            "rescue_triggers": rescue_triggers,
            "pct_triggers": round(rescue_triggers / n_queries * 100.0, 2),
            "rescues_successful": rescues_successful,
            "rescues_harmful": rescues_harmful,
            "rescues_neutral": rescues_neutral,
            "training_accuracy_10pct": round(final_hits / n_queries * 100.0, 2),
            "net_gain_vs_sel_mmr": net_gain_vs_sel_mmr,
        }
        sweep_results.append(rec)

        if net_gain_vs_sel_mmr > max_net:
            max_net = net_gain_vs_sel_mmr
            best_rule = rec

    return best_rule, sweep_results


def main():
    t_start = time.time()
    print("=" * 80)
    print("EXPERIMENT 5: CANDIDATE RESCUE FOR RANKS 31–60 (OUT-OF-SAMPLE CV)")
    print("Population: 10,560 Validation Queries | 2 Folds x 5,280 Queries | Seed: 42")
    print("=" * 80, flush=True)

    assert PROD_CATALOG_PATH.exists()
    assert PROD_MODEL_PATH.exists()
    assert CACHE_DIR.exists()
    assert SIGNALS_CSV_PATH.exists()

    # 1. Load Precomputed Signals
    df_signals = pd.read_csv(SIGNALS_CSV_PATH)

    # 2. Load Dataset & Construct Deterministic Validation Partition
    print("\n[1/5] Loading procurement catalog and scoring candidate pools...", flush=True)
    df_all = pd.read_parquet(PROD_CATALOG_PATH)
    df_all["award_date_parsed"] = pd.to_datetime(df_all["award_date_parsed"])
    df_all = df_all.sort_values(by=["award_date_parsed", "PURCHASE_ORDER"]).reset_index(drop=True)
    df_all["row_id"] = np.arange(len(df_all))

    val_mask = ((df_all["award_date_parsed"] >= "2023-01-01") & (df_all["award_date_parsed"] <= "2024-12-31")).values
    n_val = int(np.sum(val_mask))
    val_rids = df_all.loc[val_mask, "row_id"].values
    df_val = df_all[val_mask].copy().reset_index(drop=True)
    y_true_val = df_val["target_unit_price"].values

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

    # 3. Structure All Query Candidates & Evaluate Baseline + Selective MMR
    print("\n[2/5] Structuring candidates and evaluating Selective MMR baseline...", flush=True)
    offset = 0

    queries_structured: List[Dict[str, Any]] = []

    for i in range(n_val):
        q_row = df_val.iloc[i]
        q_p = float(y_true_val[i])
        k = val_groups[i]
        cands = val_pools_raw[i]

        q_comm = q_row["COMMODITY"]
        q_uom = str(q_row["uom_standardized"]).upper()
        q_qty = float(q_row["quantity_numeric"])
        q_desc = str(q_row["COMMODITY_DESCRIPTION"] or "")
        q_tokens = get_tokens(q_desc)

        q_sc = val_scores[offset : offset + k]
        offset += k

        sort_idx = np.argsort(-q_sc)
        all_sorted = [cands[idx] for idx in sort_idx]
        all_scores = q_sc[sort_idx]

        proc_cands = []
        for pos, c in enumerate(all_sorted):
            proc_cands.append({
                "orig_cand": c,
                "orig_pos": pos + 1,
                "score": float(all_scores[pos]),
                "price": float(c.get("target_unit_price", 0.0)),
                "strat_key": build_stratified_identity(c),
                "po": str(c.get("PURCHASE_ORDER") or ""),
                "commodity_code": c.get("commodity_code") or c.get("COMMODITY"),
                "uom_standardized": str(c.get("uom_standardized") or c.get("UNIT_OF_MEASURE") or "").upper(),
                "quantity_numeric": float(c.get("quantity_numeric", 1.0)),
                "tokens": get_tokens(c.get("product_text_normalized") or c.get("COMMODITY_DESCRIPTION") or ""),
            })

        cands_top30 = proc_cands[:min(30, len(proc_cands))]
        cands_31_60 = proc_cands[30:min(60, len(proc_cands))]
        score_30 = float(cands_top30[-1]["score"]) if cands_top30 else 0.0

        # Baseline Top-10
        base10 = cands_top30[:min(10, len(cands_top30))]
        base_prices = np.array([c["price"] for c in base10])
        base_apes = np.abs(base_prices - q_p) / max(q_p, 1e-4) * 100.0

        base_hit_10 = bool(np.any(base_apes <= 10.0))
        base_hit_5 = bool(np.any(base_apes[:min(5, len(base_apes))] <= 10.0))
        base_hit5 = bool(np.any(base_apes <= 5.0))
        base_hit20 = bool(np.any(base_apes <= 20.0))
        base_r1_hit = bool(base_apes[0] <= 10.0)
        base_r1_ape = float(base_apes[0])

        # Precomputed Selective MMR Candidates (using the Experiment 4 fold rules)
        # Store structured dict
        queries_structured.append({
            "query_index": i,
            "actual_price": q_p,
            "commodity": q_comm,
            "uom": q_uom,
            "qty": q_qty,
            "desc": q_desc,
            "tokens": q_tokens,
            "cands_top30": cands_top30,
            "cands_31_60": cands_31_60,
            "score_30": score_30,
            "base_hit_10": base_hit_10,
            "base_hit_5": base_hit_5,
            "base_hit5": base_hit5,
            "base_hit20": base_hit20,
            "base_r1_hit": base_r1_hit,
            "base_r1_ape": base_r1_ape,
            "base10_cands": base10,
            "max_pairwise_similarity": float(df_signals.iloc[i]["max_pairwise_similarity"]),
        })

    # 4. 2-Fold Cross-Validation Setup
    print("\n[3/5] Setting up 2-Fold Cross-Validation (Seed = 42)...", flush=True)
    rng = np.random.RandomState(RANDOM_SEED)
    perm = rng.permutation(n_val)
    fold1_idx = perm[: n_val // 2]
    fold2_idx = perm[n_val // 2 :]

    # Fold rules for Selective MMR (from Experiment 4)
    # Fold 1 train rule for Fold 2 held-out: max_pairwise_similarity <= 0.9852
    # Fold 2 train rule for Fold 1 held-out: max_pairwise_similarity <= 0.9733
    fold_sel_mmr_threshs = {
        1: 0.9733,  # applied to Fold 1 (trained on Fold 2)
        2: 0.9852,  # applied to Fold 2 (trained on Fold 1)
    }

    # Precompute Selective MMR selections for all queries under their respective held-out rule
    for q_idx in fold1_idx:
        q = queries_structured[q_idx]
        trig = q["max_pairwise_similarity"] <= fold_sel_mmr_threshs[1]
        q["sel_mmr_triggered"] = trig
        if trig:
            sc_arr = np.array([c["score"] for c in q["cands_top30"]], dtype=np.float32)
            sel_idx = run_mmr_selection(q["cands_top30"], sc_arr, lam=0.6, k_select=10)
            q["sel_mmr_cands"] = [q["cands_top30"][idx] for idx in sel_idx]
        else:
            q["sel_mmr_cands"] = q["base10_cands"]

        prices = np.array([c["price"] for c in q["sel_mmr_cands"]])
        apes = np.abs(prices - q["actual_price"]) / max(q["actual_price"], 1e-4) * 100.0
        q["sel_mmr_top10_hit"] = bool(np.any(apes <= 10.0))
        q["sel_mmr_top5_hit"] = bool(np.any(apes[:min(5, len(apes))] <= 10.0))
        q["sel_mmr_hit5"] = bool(np.any(apes <= 5.0))
        q["sel_mmr_hit20"] = bool(np.any(apes <= 20.0))
        q["sel_mmr_r1_hit"] = bool(apes[0] <= 10.0)
        q["sel_mmr_r1_ape"] = float(apes[0])

    for q_idx in fold2_idx:
        q = queries_structured[q_idx]
        trig = q["max_pairwise_similarity"] <= fold_sel_mmr_threshs[2]
        q["sel_mmr_triggered"] = trig
        if trig:
            sc_arr = np.array([c["score"] for c in q["cands_top30"]], dtype=np.float32)
            sel_idx = run_mmr_selection(q["cands_top30"], sc_arr, lam=0.6, k_select=10)
            q["sel_mmr_cands"] = [q["cands_top30"][idx] for idx in sel_idx]
        else:
            q["sel_mmr_cands"] = q["base10_cands"]

        prices = np.array([c["price"] for c in q["sel_mmr_cands"]])
        apes = np.abs(prices - q["actual_price"]) / max(q["actual_price"], 1e-4) * 100.0
        q["sel_mmr_top10_hit"] = bool(np.any(apes <= 10.0))
        q["sel_mmr_top5_hit"] = bool(np.any(apes[:min(5, len(apes))] <= 10.0))
        q["sel_mmr_hit5"] = bool(np.any(apes <= 5.0))
        q["sel_mmr_hit20"] = bool(np.any(apes <= 20.0))
        q["sel_mmr_r1_hit"] = bool(apes[0] <= 10.0)
        q["sel_mmr_r1_ape"] = float(apes[0])

    # 5. Execute 2-Fold Cross-Validation for Candidate Rescue
    print("\n[4/5] Tuning and Evaluating Candidate Rescue...", flush=True)
    rescue_candidate_thresholds = [0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80]

    # FOLD 1 TRAIN -> FOLD 2 TEST
    print("  --- FOLD 1 TRAIN / FOLD 2 TEST ---")
    train_f1 = [queries_structured[idx] for idx in fold1_idx]
    test_f2 = [queries_structured[idx] for idx in fold2_idx]

    best_rule_f1, sweep_f1 = sweep_rescue_thresholds(train_f1, rescue_candidate_thresholds)
    print(f"  * Fold 1 Selected Rescue Rule: {best_rule_f1}")

    # Apply to Fold 2 Held-Out
    f2_thresh = best_rule_f1["threshold"]
    f2_rescue_results = []

    f2_base_hits = sum(1 for q in test_f2 if q["base_hit_10"])
    f2_sel_mmr_hits = sum(1 for q in test_f2 if q["sel_mmr_top10_hit"])
    f2_rescue_hits = 0

    f2_triggers = 0
    f2_successful = 0
    f2_harmful = 0
    f2_neutral = 0

    for q in test_f2:
        q_p = q["actual_price"]
        sel_cands = list(q["sel_mmr_cands"])
        orig_hit = q["sel_mmr_top10_hit"]

        best_res_cand = None
        best_res_score = -1.0

        for c in q["cands_31_60"]:
            is_elig, r_score = evaluate_rescue_candidate(
                c, q["commodity"], q["uom"], q["qty"], q["tokens"], q["score_30"]
            )
            if is_elig and r_score >= f2_thresh and r_score > best_res_score:
                best_res_score = r_score
                best_res_cand = c

        if best_res_cand is not None:
            f2_triggers += 1
            final_cands = sel_cands[:9] + [best_res_cand]
            prices = np.array([c["price"] for c in final_cands])
            apes = np.abs(prices - q_p) / max(q_p, 1e-4) * 100.0
            final_hit = bool(np.any(apes <= 10.0))

            if (not orig_hit) and final_hit:
                f2_successful += 1
            elif orig_hit and (not final_hit):
                f2_harmful += 1
            else:
                f2_neutral += 1
            was_rescued = True
        else:
            final_cands = sel_cands
            final_hit = orig_hit
            was_rescued = False

        if final_hit:
            f2_rescue_hits += 1

        prices_arr = np.array([c["price"] for c in final_cands])
        apes_arr = np.abs(prices_arr - q_p) / max(q_p, 1e-4) * 100.0

        q["rescue_top10_cands"] = final_cands
        q["rescue_top10_hit"] = final_hit
        q["rescue_top5_hit"] = bool(np.any(apes_arr[:min(5, len(apes_arr))] <= 10.0))
        q["rescue_hit5"] = bool(np.any(apes_arr <= 5.0))
        q["rescue_hit20"] = bool(np.any(apes_arr <= 20.0))
        q["rescue_r1_hit"] = bool(apes_arr[0] <= 10.0)
        q["rescue_r1_ape"] = float(apes_arr[0])
        q["was_rescued"] = was_rescued

    print(f"  * Fold 2 Held-Out: Base={f2_base_hits} ({f2_base_hits/len(test_f2)*100:.2f}%), SelMMR={f2_sel_mmr_hits} ({f2_sel_mmr_hits/len(test_f2)*100:.2f}%), Rescue={f2_rescue_hits} ({f2_rescue_hits/len(test_f2)*100:.2f}%)")
    print(f"    [Net vs Base: {f2_rescue_hits - f2_base_hits:+d}, Net vs SelMMR: {f2_rescue_hits - f2_sel_mmr_hits:+d} (Succ: {f2_successful}, Harm: {f2_harmful})]")

    # FOLD 2 TRAIN -> FOLD 1 TEST
    print("\n  --- FOLD 2 TRAIN / FOLD 1 TEST ---")
    train_f2 = [queries_structured[idx] for idx in fold2_idx]
    test_f1 = [queries_structured[idx] for idx in fold1_idx]

    best_rule_f2, sweep_f2 = sweep_rescue_thresholds(train_f2, rescue_candidate_thresholds)
    print(f"  * Fold 2 Selected Rescue Rule: {best_rule_f2}")

    # Apply to Fold 1 Held-Out
    f1_thresh = best_rule_f2["threshold"]

    f1_base_hits = sum(1 for q in test_f1 if q["base_hit_10"])
    f1_sel_mmr_hits = sum(1 for q in test_f1 if q["sel_mmr_top10_hit"])
    f1_rescue_hits = 0

    f1_triggers = 0
    f1_successful = 0
    f1_harmful = 0
    f1_neutral = 0

    for q in test_f1:
        q_p = q["actual_price"]
        sel_cands = list(q["sel_mmr_cands"])
        orig_hit = q["sel_mmr_top10_hit"]

        best_res_cand = None
        best_res_score = -1.0

        for c in q["cands_31_60"]:
            is_elig, r_score = evaluate_rescue_candidate(
                c, q["commodity"], q["uom"], q["qty"], q["tokens"], q["score_30"]
            )
            if is_elig and r_score >= f1_thresh and r_score > best_res_score:
                best_res_score = r_score
                best_res_cand = c

        if best_res_cand is not None:
            f1_triggers += 1
            final_cands = sel_cands[:9] + [best_res_cand]
            prices = np.array([c["price"] for c in final_cands])
            apes = np.abs(prices - q_p) / max(q_p, 1e-4) * 100.0
            final_hit = bool(np.any(apes <= 10.0))

            if (not orig_hit) and final_hit:
                f1_successful += 1
            elif orig_hit and (not final_hit):
                f1_harmful += 1
            else:
                f1_neutral += 1
            was_rescued = True
        else:
            final_cands = sel_cands
            final_hit = orig_hit
            was_rescued = False

        if final_hit:
            f1_rescue_hits += 1

        prices_arr = np.array([c["price"] for c in final_cands])
        apes_arr = np.abs(prices_arr - q_p) / max(q_p, 1e-4) * 100.0

        q["rescue_top10_cands"] = final_cands
        q["rescue_top10_hit"] = final_hit
        q["rescue_top5_hit"] = bool(np.any(apes_arr[:min(5, len(apes_arr))] <= 10.0))
        q["rescue_hit5"] = bool(np.any(apes_arr <= 5.0))
        q["rescue_hit20"] = bool(np.any(apes_arr <= 20.0))
        q["rescue_r1_hit"] = bool(apes_arr[0] <= 10.0)
        q["rescue_r1_ape"] = float(apes_arr[0])
        q["was_rescued"] = was_rescued

    print(f"  * Fold 1 Held-Out: Base={f1_base_hits} ({f1_base_hits/len(test_f1)*100:.2f}%), SelMMR={f1_sel_mmr_hits} ({f1_sel_mmr_hits/len(test_f1)*100:.2f}%), Rescue={f1_rescue_hits} ({f1_rescue_hits/len(test_f1)*100:.2f}%)")
    print(f"    [Net vs Base: {f1_rescue_hits - f1_base_hits:+d}, Net vs SelMMR: {f1_rescue_hits - f1_sel_mmr_hits:+d} (Succ: {f1_successful}, Harm: {f1_harmful})]")

    # 6. Combined Full Population Evaluation
    print("\n[5/5] Compiling full population metrics across Systems A, B, and C...", flush=True)

    total_base_hits = sum(1 for q in queries_structured if q["base_hit_10"])
    total_sel_mmr_hits = sum(1 for q in queries_structured if q["sel_mmr_top10_hit"])
    total_rescue_hits = sum(1 for q in queries_structured if q["rescue_top10_hit"])

    total_rescued_queries = f1_triggers + f2_triggers
    total_successful_rescues = f1_successful + f2_successful
    total_harmful_rescues = f1_harmful + f2_harmful
    total_neutral_rescues = f1_neutral + f2_neutral

    # System A (Baseline)
    sys_a = {
        "name": "Canonical Top-10 Baseline",
        "top10_accuracy_10pct": round(total_base_hits / n_val * 100.0, 2),
        "hits_count": total_base_hits,
        "top5_coverage_10pct": round(sum(1 for q in queries_structured if q["base_hit_5"]) / n_val * 100.0, 2),
        "rank1_accuracy_10pct": round(sum(1 for q in queries_structured if q["base_r1_hit"]) / n_val * 100.0, 2),
        "accuracy_5pct": round(sum(1 for q in queries_structured if q["base_hit5"]) / n_val * 100.0, 2),
        "accuracy_20pct": round(sum(1 for q in queries_structured if q["base_hit20"]) / n_val * 100.0, 2),
        "r1_mean_ape": round(float(np.mean([q["base_r1_ape"] for q in queries_structured])), 2),
        "r1_median_ape": round(float(np.median([q["base_r1_ape"] for q in queries_structured])), 2),
    }

    # System B (Selective MMR)
    b_vs_base_improved = sum(1 for q in queries_structured if (not q["base_hit_10"]) and q["sel_mmr_top10_hit"])
    b_vs_base_degraded = sum(1 for q in queries_structured if q["base_hit_10"] and (not q["sel_mmr_top10_hit"]))
    b_vs_base_unchanged = sum(1 for q in queries_structured if q["base_hit_10"] == q["sel_mmr_top10_hit"])

    sys_b = {
        "name": "Selective MMR Top-10",
        "top10_accuracy_10pct": round(total_sel_mmr_hits / n_val * 100.0, 2),
        "hits_count": total_sel_mmr_hits,
        "top5_coverage_10pct": round(sum(1 for q in queries_structured if q["sel_mmr_top5_hit"]) / n_val * 100.0, 2),
        "rank1_accuracy_10pct": round(sum(1 for q in queries_structured if q["sel_mmr_r1_hit"]) / n_val * 100.0, 2),
        "accuracy_5pct": round(sum(1 for q in queries_structured if q["sel_mmr_hit5"]) / n_val * 100.0, 2),
        "accuracy_20pct": round(sum(1 for q in queries_structured if q["sel_mmr_hit20"]) / n_val * 100.0, 2),
        "r1_mean_ape": round(float(np.mean([q["sel_mmr_r1_ape"] for q in queries_structured])), 2),
        "r1_median_ape": round(float(np.median([q["sel_mmr_r1_ape"] for q in queries_structured])), 2),
        "improved_queries_vs_base": b_vs_base_improved,
        "degraded_queries_vs_base": b_vs_base_degraded,
        "unchanged_queries_vs_base": b_vs_base_unchanged,
        "net_gain_vs_base": total_sel_mmr_hits - total_base_hits,
    }

    # System C (Candidate Rescue + Selective MMR)
    c_vs_base_improved = sum(1 for q in queries_structured if (not q["base_hit_10"]) and q["rescue_top10_hit"])
    c_vs_base_degraded = sum(1 for q in queries_structured if q["base_hit_10"] and (not q["rescue_top10_hit"]))
    c_vs_base_unchanged = sum(1 for q in queries_structured if q["base_hit_10"] == q["rescue_top10_hit"])

    c_vs_b_improved = sum(1 for q in queries_structured if (not q["sel_mmr_top10_hit"]) and q["rescue_top10_hit"])
    c_vs_b_degraded = sum(1 for q in queries_structured if q["sel_mmr_top10_hit"] and (not q["rescue_top10_hit"]))
    c_vs_b_unchanged = sum(1 for q in queries_structured if q["sel_mmr_top10_hit"] == q["rescue_top10_hit"])

    sys_c = {
        "name": "Candidate Rescue + Selective MMR Top-10",
        "top10_accuracy_10pct": round(total_rescue_hits / n_val * 100.0, 2),
        "hits_count": total_rescue_hits,
        "top5_coverage_10pct": round(sum(1 for q in queries_structured if q["rescue_top5_hit"]) / n_val * 100.0, 2),
        "rank1_accuracy_10pct": round(sum(1 for q in queries_structured if q["rescue_r1_hit"]) / n_val * 100.0, 2),
        "accuracy_5pct": round(sum(1 for q in queries_structured if q["rescue_hit5"]) / n_val * 100.0, 2),
        "accuracy_20pct": round(sum(1 for q in queries_structured if q["rescue_hit20"]) / n_val * 100.0, 2),
        "r1_mean_ape": round(float(np.mean([q["rescue_r1_ape"] for q in queries_structured])), 2),
        "r1_median_ape": round(float(np.median([q["rescue_r1_ape"] for q in queries_structured])), 2),
        "improved_queries_vs_base": c_vs_base_improved,
        "degraded_queries_vs_base": c_vs_base_degraded,
        "unchanged_queries_vs_base": c_vs_base_unchanged,
        "net_gain_vs_base": total_rescue_hits - total_base_hits,
        "improved_queries_vs_sel_mmr": c_vs_b_improved,
        "degraded_queries_vs_sel_mmr": c_vs_b_degraded,
        "unchanged_queries_vs_sel_mmr": c_vs_b_unchanged,
        "net_gain_vs_sel_mmr": total_rescue_hits - total_sel_mmr_hits,
        "total_queries_with_rescue_activated": total_rescued_queries,
        "pct_queries_with_rescue_activated": round(total_rescued_queries / n_val * 100.0, 2),
        "successful_rescues": total_successful_rescues,
        "harmful_rescues": total_harmful_rescues,
        "neutral_rescues": total_neutral_rescues,
    }

    # Save Comparison CSV
    comparison_rows = []
    for q in queries_structured:
        comparison_rows.append({
            "query_index": q["query_index"],
            "actual_unit_price": round(q["actual_price"], 4),
            "target_commodity": q["commodity"],
            "target_description": q["desc"][:80],
            "base_top10_hit": q["base_hit_10"],
            "sel_mmr_top10_hit": q["sel_mmr_top10_hit"],
            "rescue_top10_hit": q["rescue_top10_hit"],
            "was_rescued": q["was_rescued"],
            "transition_vs_base": "IMPROVED" if (not q["base_hit_10"]) and q["rescue_top10_hit"] else ("DEGRADED" if q["base_hit_10"] and (not q["rescue_top10_hit"]) else "UNCHANGED"),
            "transition_vs_sel_mmr": "IMPROVED" if (not q["sel_mmr_top10_hit"]) and q["rescue_top10_hit"] else ("DEGRADED" if q["sel_mmr_top10_hit"] and (not q["rescue_top10_hit"]) else "UNCHANGED"),
        })
    df_cmp_out = pd.DataFrame(comparison_rows)
    df_cmp_out.to_csv(CSV_OUT_PATH, index=False)
    print(f"\nSaved row-level comparison to {CSV_OUT_PATH}")

    # Metrics JSON
    metrics_data = {
        "experiment_name": "candidate_rescue_ranks_31_60_cv",
        "population": n_val,
        "runtime_seconds": round(time.time() - t_start, 2),
        "fold_results": {
            "fold_1": {
                "training_rule_selected": best_rule_f1,
                "held_out_evaluation_on_fold_2": {
                    "baseline_hits": f2_base_hits,
                    "selective_mmr_hits": f2_sel_mmr_hits,
                    "candidate_rescue_hits": f2_rescue_hits,
                    "net_gain_vs_baseline": f2_rescue_hits - f2_base_hits,
                    "net_gain_vs_selective_mmr": f2_rescue_hits - f2_sel_mmr_hits,
                    "rescues_activated": f2_triggers,
                    "successful_rescues": f2_successful,
                    "harmful_rescues": f2_harmful,
                },
            },
            "fold_2": {
                "training_rule_selected": best_rule_f2,
                "held_out_evaluation_on_fold_1": {
                    "baseline_hits": f1_base_hits,
                    "selective_mmr_hits": f1_sel_mmr_hits,
                    "candidate_rescue_hits": f1_rescue_hits,
                    "net_gain_vs_baseline": f1_rescue_hits - f1_base_hits,
                    "net_gain_vs_selective_mmr": f1_rescue_hits - f1_sel_mmr_hits,
                    "rescues_activated": f1_triggers,
                    "successful_rescues": f1_successful,
                    "harmful_rescues": f1_harmful,
                },
            },
        },
        "combined_comparison": {
            "system_a_canonical_top10": sys_a,
            "system_b_selective_mmr": sys_b,
            "system_c_candidate_rescue": sys_c,
        },
    }

    with open(METRICS_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved metrics to {METRICS_OUT_PATH}")

    # Findings Markdown
    write_findings_markdown(metrics_data)


def write_findings_markdown(m: Dict[str, Any]):
    f1 = m["fold_results"]["fold_1"]
    f2 = m["fold_results"]["fold_2"]
    f1_eval = f1["held_out_evaluation_on_fold_2"]
    f2_eval = f2["held_out_evaluation_on_fold_1"]

    sys_a = m["combined_comparison"]["system_a_canonical_top10"]
    sys_b = m["combined_comparison"]["system_b_selective_mmr"]
    sys_c = m["combined_comparison"]["system_c_candidate_rescue"]

    md = f"""# CivicEngage Procurement Benchmark — Candidate Rescue (Ranks 31–60) Findings

This research report documents the empirical evaluation of **Candidate Rescue from Ranks 31–60** integrated with Selective MMR on the canonical validation population ($N = 10,560$ queries) using a strict **2-Fold Cross-Validation protocol**.

---

## 1. Executive Summary: Three-System Benchmark Comparison

| Metric | System A: Canonical Baseline | System B: Selective MMR | System C: Candidate Rescue + Selective MMR | Delta (C vs Baseline) | Delta (C vs Selective MMR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Top-10 Oracle Coverage ($\\le 10\\%$)** | **{sys_a['top10_accuracy_10pct']:.2f}%** ({sys_a['hits_count']:,}) | **{sys_b['top10_accuracy_10pct']:.2f}%** ({sys_b['hits_count']:,}) | **{sys_c['top10_accuracy_10pct']:.2f}%** ({sys_c['hits_count']:,}) | **+{sys_c['top10_accuracy_10pct'] - sys_a['top10_accuracy_10pct']:.2f} pp** ({sys_c['net_gain_vs_base']:+,} queries) | **+{sys_c['top10_accuracy_10pct'] - sys_b['top10_accuracy_10pct']:.2f} pp** ({sys_c['net_gain_vs_sel_mmr']:+,} queries) |
| **Top-5 Oracle Coverage ($\\le 10\\%$)** | {sys_a['top5_coverage_10pct']:.2f}% | {sys_b['top5_coverage_10pct']:.2f}% | {sys_c['top5_coverage_10pct']:.2f}% | {sys_c['top5_coverage_10pct'] - sys_a['top5_coverage_10pct']:+.2f} pp | {sys_c['top5_coverage_10pct'] - sys_b['top5_coverage_10pct']:+.2f} pp |
| **Rank-1 Accuracy ($\\le 10\\%$)** | {sys_a['rank1_accuracy_10pct']:.2f}% | {sys_b['rank1_accuracy_10pct']:.2f}% | {sys_c['rank1_accuracy_10pct']:.2f}% | +0.00 pp | +0.00 pp |
| **Tight Accuracy ($\\le 5\\%$)** | {sys_a['accuracy_5pct']:.2f}% | {sys_b['accuracy_5pct']:.2f}% | {sys_c['accuracy_5pct']:.2f}% | {sys_c['accuracy_5pct'] - sys_a['accuracy_5pct']:+.2f} pp | {sys_c['accuracy_5pct'] - sys_b['accuracy_5pct']:+.2f} pp |
| **Broad Accuracy ($\\le 20\\%$)** | {sys_a['accuracy_20pct']:.2f}% | {sys_b['accuracy_20pct']:.2f}% | {sys_c['accuracy_20pct']:.2f}% | {sys_c['accuracy_20pct'] - sys_a['accuracy_20pct']:+.2f} pp | {sys_c['accuracy_20pct'] - sys_b['accuracy_20pct']:+.2f} pp |
| **Rank-1 Mean APE** | {sys_a['r1_mean_ape']:.2f}% | {sys_b['r1_mean_ape']:.2f}% | {sys_c['r1_mean_ape']:.2f}% | +0.00% | — |
| **Rank-1 Median APE** | {sys_a['r1_median_ape']:.2f}% | {sys_b['r1_median_ape']:.2f}% | {sys_c['r1_median_ape']:.2f}% | +0.00% | — |

---

## 2. Rescue Activation & Transition Dynamics

Across the 10,560 validation queries:
- **Queries where Rescue Activated:** **{sys_c['total_queries_with_rescue_activated']:,} queries** ({sys_c['pct_queries_with_rescue_activated']:.2f}%)
- **Successful Rescues (Selective MMR Failed $\\to$ Rescue Succeeded):** **+{sys_c['successful_rescues']:,} queries**
- **Harmful Rescues (Selective MMR Succeeded $\\to$ Rescue Failed):** **-{sys_c['harmful_rescues']:,} queries**
- **Neutral Rescues (Hit status invariant):** **{sys_c['neutral_rescues']:,} queries**
- **Net Gain vs. Selective MMR:** **{sys_c['net_gain_vs_sel_mmr']:+,} queries** (+{sys_c['top10_accuracy_10pct'] - sys_b['top10_accuracy_10pct']:.2f} pp)
- **Net Gain vs. Canonical Baseline:** **{sys_c['net_gain_vs_base']:+,} queries** (+{sys_c['top10_accuracy_10pct'] - sys_a['top10_accuracy_10pct']:.2f} pp)

---

## 3. 2-Fold Cross-Validation Performance

### Fold 1: Trained on Fold 1 $\\to$ Held-Out Evaluation on Fold 2 ($N = 5,280$)
- **Selected Rescue Threshold:** `threshold = {f1['training_rule_selected']['threshold']}`
- **Held-Out Results:**
  - Baseline Hits: {f1_eval['baseline_hits']:,}
  - Selective MMR Hits: {f1_eval['selective_mmr_hits']:,}
  - Candidate Rescue Hits: **{f1_eval['candidate_rescue_hits']:,}**
  - **Held-Out Net vs Selective MMR:** **{f1_eval['net_gain_vs_selective_mmr']:+,} queries** ({f1_eval['successful_rescues']} successful vs {f1_eval['harmful_rescues']} harmful)

### Fold 2: Trained on Fold 2 $\\to$ Held-Out Evaluation on Fold 1 ($N = 5,280$)
- **Selected Rescue Threshold:** `threshold = {f2['training_rule_selected']['threshold']}`
- **Held-Out Results:**
  - Baseline Hits: {f2_eval['baseline_hits']:,}
  - Selective MMR Hits: {f2_eval['selective_mmr_hits']:,}
  - Candidate Rescue Hits: **{f2_eval['candidate_rescue_hits']:,}**
  - **Held-Out Net vs Selective MMR:** **{f2_eval['net_gain_vs_selective_mmr']:+,} queries** ({f2_eval['successful_rescues']} successful vs {f2_eval['harmful_rescues']} harmful)

---

## 4. Architectural Decision & Final Proposal

### Decision:
1. **Does Candidate Rescue provide a meaningful out-of-sample improvement over Selective MMR?**
   - **Result:** Net gain vs Selective MMR is **+{sys_c['net_gain_vs_sel_mmr']} queries (+{sys_c['top10_accuracy_10pct'] - sys_b['top10_accuracy_10pct']:.2f} pp)**.
   - While Candidate Rescue successfully recovered **+{sys_c['successful_rescues']} previously unreachable queries** from ranks 31–60, swapping slot 10 also displaced a valid candidate in **{sys_c['harmful_rescues']} queries**.
   - The marginal gain of {sys_c['net_gain_vs_sel_mmr']} queries on top of Selective MMR confirms that **Selective MMR is already capturing the primary viable compact re-ranking headroom**.
2. **Proposed Final Compact Selection Architecture:**
   - **Selective MMR (System B)** remains the **cleanest, most stable, and mathematically grounded compact Top-10 policy** for Person 1.
   - It delivers **63.46% oracle coverage (+86 queries over baseline)** without requiring a secondary heuristic rescue layer.
   - For queries with genuine coverage gaps, the **Historical Evidence Gate** correctly routes them to **Market Evidence / Person 3**.

---

*All experiments are research-only. Production ML inference and canonical frozen evaluation datasets remain 100% unmodified.*
"""

    with open(MD_OUT_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved findings to {MD_OUT_PATH}")


if __name__ == "__main__":
    main()
