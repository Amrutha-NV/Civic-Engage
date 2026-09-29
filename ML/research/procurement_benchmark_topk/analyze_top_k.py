"""
CivicEngage - Procurement Benchmark ML
Step 37 / Frozen T3 Co-Occurrence Model Top-K Analysis

Reuses the exact Step-37 candidate retrieval, 176-feature construction,
and serialized LambdaMART ranker from:
  ML/models/procurement_ranker.joblib
  ML/src/analysis/run_step37_po_context.py

Extracts ranked Top-10 candidates per validation query and computes:
- Rank-1 Accuracy@10%
- Top-3 Accuracy@10%
- Top-5 Accuracy@10%
- Top-10 Accuracy@10%
- Cases where Rank-1 is outside +-10% but Top-5 contains a +-10% candidate
- Cases where Rank-1 is outside +-10% but Top-10 contains a +-10% candidate
"""

import math
import os
import sys
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd

script_dir = Path(__file__).resolve().parent
ml_root = script_dir.parent.parent
sys.path.insert(0, str(ml_root / "src" / "ranking"))
sys.path.insert(0, str(ml_root / "src" / "models"))
sys.path.insert(0, str(ml_root / "src" / "normalization"))


def main():
    t_start = time.time()
    print("=" * 80)
    print("FROZEN STEP-37 / T3 TOP-K CANDIDATE ANALYSIS (N = 10,560)")
    print("=" * 80)

    cache_dir = ml_root / "data" / "cache" / "step36"
    assert cache_dir.exists(), f"Cache dir {cache_dir} missing!"

    model_file = ml_root / "models" / "procurement_ranker.joblib"
    assert model_file.exists(), f"Model file {model_file} missing!"

    # 1. Load Canonical Dataset & Verify Validation Partition
    print("\n[1/5] Loading dataset & normalization caches...", flush=True)
    parquet_path = ml_root / "data" / "processed_v2" / "unit_price_training_dataset.parquet"
    df_all = pd.read_parquet(parquet_path)
    df_all["award_date_parsed"] = pd.to_datetime(df_all["award_date_parsed"])
    df_all = df_all.sort_values(by=["award_date_parsed", "PURCHASE_ORDER"]).reset_index(drop=True)
    df_all["row_id"] = np.arange(len(df_all))
    df_all["award_dt"] = df_all["award_date_parsed"]

    norm_cache_path = ml_root / "data" / "processed_v2" / "normalized_identity_cache.parquet"
    if norm_cache_path.exists():
        df_norm = pd.read_parquet(norm_cache_path)
        for col in df_norm.columns:
            df_all[col] = df_norm[col]

    val_mask = ((df_all["award_date_parsed"] >= "2023-01-01") & (df_all["award_date_parsed"] <= "2024-12-31")).values
    n_val = int(np.sum(val_mask))
    assert n_val == 10560, f"Expected 10,560 validation queries, got {n_val}"

    df_val = df_all[val_mask].copy().reset_index(drop=True)
    val_rids = df_all.loc[val_mask, "row_id"].values
    y_true_val = df_val["target_unit_price"].values
    print(f"  * Validation Population (2023-2024): {n_val:,} records")

    # 2. Extract PO-Level Aggregates for Co-occurrence Context
    print("\n[2/5] Precomputing PO aggregates & co-occurrence metadata...", flush=True)
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

    # 3. Load Cached Baseline Pools & Matrices
    print("\n[3/5] Loading Step-36 base matrices (170f) & candidate pools...", flush=True)
    val_pools_raw = joblib.load(cache_dir / "val_pools.joblib")
    X_val_t0 = np.load(cache_dir / "X_val_t0.npy")
    val_groups = np.load(cache_dir / "val_groups.npy").tolist()

    print(f"  * Cached Base Val Matrix: {X_val_t0.shape}")

    # Build T3 Co-Occurrence Features (6 features)
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

            # --- T3: Co-Occurrence & Concentration (6 features) ---
            t3_feats = [
                q_hhi,
                q_is_prim,
                math.log1p(q_other_qty),
                1.0 if c_comm in q_comm_set else 0.0,
                abs(q_hhi - c_hhi),
                1.0 if (q_is_prim > 0.5 and c_is_prim > 0.5) else 0.0
            ]
            t3_list.append(t3_feats)

    X_val_t3_sub = np.array(t3_list, dtype=np.float32)
    X_val_t3 = np.hstack([X_val_t0, X_val_t3_sub])
    print(f"  * Final 176f Validation Matrix: {X_val_t3.shape}")

    # 4. Score Candidate Pools with Frozen Step-37 Model
    print("\n[4/5] Scoring with frozen Step-37 LightGBM Ranker...", flush=True)
    ranker = joblib.load(model_file)
    val_scores = ranker.predict(X_val_t3)

    # 5. Extract Ranked Top-10 Candidates and Diagnostics
    print("\n[5/5] Extracting Top-10 candidates per query and calculating metrics...", flush=True)
    top_10_records = []
    offset = 0

    rank1_hits = 0
    top3_hits = 0
    top5_hits = 0
    top10_hits = 0

    rank1_miss_top5_hit = 0
    rank1_miss_top10_hit = 0
    queries_with_cands = 0

    for i in range(n_val):
        q_p = float(y_true_val[i])
        k = val_groups[i]
        cands = val_pools_raw[i]

        if k == 0 or len(cands) == 0:
            continue

        queries_with_cands += 1
        q_sc = val_scores[offset : offset + k]
        offset += k

        sort_idx = np.argsort(-q_sc)
        sorted_prices = np.array([float(cands[j].get("target_unit_price", 0.0)) for j in sort_idx])
        apes = np.abs(sorted_prices - q_p) / max(q_p, 1e-4) * 100.0

        r1_hit = bool(apes[0] <= 10.0)
        t3_hit = bool(np.any(apes[:min(3, len(apes))] <= 10.0))
        t5_hit = bool(np.any(apes[:min(5, len(apes))] <= 10.0))
        t10_hit = bool(np.any(apes[:min(10, len(apes))] <= 10.0))

        if r1_hit:
            rank1_hits += 1
        if t3_hit:
            top3_hits += 1
        if t5_hit:
            top5_hits += 1
        if t10_hit:
            top10_hits += 1

        if (not r1_hit) and t5_hit:
            rank1_miss_top5_hit += 1
        if (not r1_hit) and t10_hit:
            rank1_miss_top10_hit += 1

        # Extract top 10 candidates for export
        top_k_limit = min(10, k)
        for r_pos in range(top_k_limit):
            s_idx = sort_idx[r_pos]
            cand = cands[s_idx]
            cand_p = float(cand.get("target_unit_price", 0.0))
            cand_sc = float(q_sc[s_idx])
            cand_desc = (
                cand.get("item_description")
                or cand.get("ITEM_DESCRIPTION")
                or cand.get("product_text_normalized")
                or ""
            )
            cand_ape = round(float(apes[r_pos]), 4)
            is_within_10 = bool(cand_ape <= 10.0)

            top_10_records.append({
                "query_index": i,
                "actual_unit_price": q_p,
                "candidate_rank": r_pos + 1,
                "candidate_unit_price": cand_p,
                "candidate_score": round(cand_sc, 6),
                "candidate_description": cand_desc,
                "absolute_percentage_error": cand_ape,
                "is_within_10pct": is_within_10
            })

    # Summary Accuracies
    rank1_acc = round(rank1_hits / n_val * 100.0, 2)
    top3_acc = round(top3_hits / n_val * 100.0, 2)
    top5_acc = round(top5_hits / n_val * 100.0, 2)
    top10_acc = round(top10_hits / n_val * 100.0, 2)

    metrics = {
        "validation_queries_total": n_val,
        "queries_with_candidates": queries_with_cands,
        "candidate_coverage_pct": round(queries_with_cands / n_val * 100.0, 2),
        "rank_1_accuracy_10pct": rank1_acc,
        "top_3_accuracy_10pct": top3_acc,
        "top_5_accuracy_10pct": top5_acc,
        "top_10_accuracy_10pct": top10_acc,
        "rank_1_hits": rank1_hits,
        "top_3_hits": top3_hits,
        "top_5_hits": top5_hits,
        "top_10_hits": top10_hits,
        "rank1_outside_10pct_but_top5_within_10pct_count": rank1_miss_top5_hit,
        "rank1_outside_10pct_but_top5_within_10pct_pct": round(rank1_miss_top5_hit / n_val * 100.0, 2),
        "rank1_outside_10pct_but_top10_within_10pct_count": rank1_miss_top10_hit,
        "rank1_outside_10pct_but_top10_within_10pct_pct": round(rank1_miss_top10_hit / n_val * 100.0, 2),
        "total_top_10_records_generated": len(top_10_records),
        "runtime_seconds": round(time.time() - t_start, 2)
    }

    # Save outputs
    out_csv = script_dir / "top_10_candidates.csv"
    out_json = script_dir / "top_k_metrics.json"

    df_top_10 = pd.DataFrame(top_10_records)
    df_top_10.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 80)
    print("STEP-37 FROZEN MODEL TOP-K RESULTS")
    print("=" * 80)
    print(f"Validation Queries:                                   {n_val:,}")
    print(f"Rank-1 Accuracy@10%:                                  {rank1_acc:.2f}% ({rank1_hits:,}/{n_val:,})")
    print(f"Top-3 Accuracy@10%:                                   {top3_acc:.2f}% ({top3_hits:,}/{n_val:,})")
    print(f"Top-5 Accuracy@10%:                                   {top5_acc:.2f}% ({top5_hits:,}/{n_val:,})")
    print(f"Top-10 Accuracy@10%:                                  {top10_acc:.2f}% ({top10_hits:,}/{n_val:,})")
    print("-" * 80)
    print(f"Rank-1 OUTSIDE +-10% BUT Top-5 CONTAINS +-10%:        {rank1_miss_top5_hit:,} queries ({metrics['rank1_outside_10pct_but_top5_within_10pct_pct']}%)")
    print(f"Rank-1 OUTSIDE +-10% BUT Top-10 CONTAINS +-10%:       {rank1_miss_top10_hit:,} queries ({metrics['rank1_outside_10pct_but_top10_within_10pct_pct']}%)")
    print("=" * 80)
    print(f"Output candidate CSV: {out_csv}")
    print(f"Output metrics JSON:  {out_json}")


if __name__ == "__main__":
    main()
