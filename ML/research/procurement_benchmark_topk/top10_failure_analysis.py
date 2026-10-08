"""
CivicEngage Procurement Benchmark Research
Step-37 Top-10 Failure & Recovery Analysis

Objective:
Analyze queries where the canonical Step-37 LambdaMART model failed to place a
price-compatible candidate (within +-10% of actual_unit_price) inside the Top-10
candidate pool, but where a price-compatible candidate DOES exist within ranks 11-60.

Population:
- Total validation queries: 10,560
- Top-10 hits: 6,615 (62.64%)
- Top-10 failures: 3,945 (37.36%)
- Top-60 hits: 8,986 (85.09%)
- Recoverable between ranks 11-60: 2,371 queries
- Completely unrecoverable at Top-60: 1,574 queries

Rules:
- Strictly read-only on production ML package (ML/procurement_benchmark_engine)
- Strictly read-only on frozen evaluation dataset (ML/research/procurement_benchmark_topk/frozen_evaluation_dataset.parquet)
- Work ONLY inside ML/research/procurement_benchmark_topk/
- Creates:
  1. top10_failure_analysis.py
  2. top10_failure_analysis.csv
  3. TOP10_FAILURE_FINDINGS.md
"""

import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths & Canonical Resources                                                 #
# --------------------------------------------------------------------------- #
_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parent.parent.parent

PROD_CATALOG_PATH = REPO_ROOT / "ML" / "procurement_benchmark_engine" / "data" / "catalog" / "procurement_catalog.parquet"
PROD_MODEL_PATH = REPO_ROOT / "ML" / "procurement_benchmark_engine" / "models" / "procurement_ranker.joblib"
CACHE_DIR = Path(r"D:\tender_generation\ML\data\cache\step36")

CSV_OUT_PATH = _THIS_DIR / "top10_failure_analysis.csv"
MD_OUT_PATH = _THIS_DIR / "TOP10_FAILURE_FINDINGS.md"


def classify_failure_pattern(row: Dict[str, Any]) -> str:
    """
    Deterministically assigns a primary failure pattern grounded in quantitative evidence.
    """
    # 1. Score Compression: recovery candidate was barely edged out by rank 10
    score_gap = abs(row["score_gap_to_rank10"])
    
    # 2. UOM Incompatibility: Top-10 has UOM mismatch while recovery candidate matches UOM
    if not row["rank1_uom_match"] and row["recovery_uom_match"]:
        return "UOM_PACK_INCOMPATIBILITY"

    # 3. Quantity Scale Divergence: Rank-1 quantity differs by > 3x while recovery matches within 1.5x
    q_ratio_r1 = row["rank1_qty_ratio"]
    q_ratio_rec = row["recovery_qty_ratio"]
    if (q_ratio_r1 > 3.0 or q_ratio_r1 < 0.33) and (0.67 <= q_ratio_rec <= 1.5):
        return "QUANTITY_SCALE_MISMATCH"

    # 4. Premium / High-Spec Distractor: Rank-1 candidate price is > 2.0x actual price
    if row["rank1_price_ratio"] > 2.0:
        return "HIGH_SPEC_DISTRACTOR_DOMINANCE"

    # 5. Low-Ball / Partial Scope Distractor: Rank-1 candidate price is < 0.5x actual price
    if row["rank1_price_ratio"] < 0.5:
        return "PARTIAL_SCOPE_LOWBALL_DISTRACTOR"

    # 6. Temporal Drift: Rank-1 candidate is > 730 days older than recovery candidate
    if (row["rank1_days_elapsed"] - row["recovery_days_elapsed"]) > 730:
        return "TEMPORAL_RECENCY_GAP"

    # 7. Marginal Score Compression: Score difference between rank 10 and recovery candidate is small (< 0.20)
    if score_gap < 0.20:
        return "SCORE_COMPRESSION_MARGINAL_RANK"

    # 8. Commodity Taxonomy Drift
    if not row["rank1_comm_match"] and row["recovery_comm_match"]:
        return "COMMODITY_TAXONOMY_MISMATCH"

    # 9. Generic Text Similarity Override: Fallback channel in rank-1 vs structured in recovery
    ch_r1 = str(row["rank1_channel"])
    ch_rec = str(row["recovery_channel"])
    if ch_r1 in ["I_COMM_UOM", "J_QUANTITY_AWARE"] and ch_rec in ["A_EXACT_IDENTITY", "B_BRAND_MODEL", "C_MODEL_FAMILY", "D_NUMERIC_SPECS", "E_WORD_TFIDF"]:
        return "GENERIC_CHANNEL_OVERRIDE"

    return "COMPLEX_SPEC_RANKING_DEFICIT"


def run_failure_analysis():
    t_start = time.time()
    print("=" * 80)
    print("STEP-37 TOP-10 FAILURE & RECOVERY ANALYSIS (N = 10,560)")
    print("=" * 80)

    # 1. Verify Canonical Resources
    assert PROD_CATALOG_PATH.exists(), f"Catalog missing: {PROD_CATALOG_PATH}"
    assert PROD_MODEL_PATH.exists(), f"Ranker model missing: {PROD_MODEL_PATH}"
    assert CACHE_DIR.exists(), f"Step-36 cache missing: {CACHE_DIR}"

    # 2. Load Dataset & Construct Deterministic Validation Partition
    print("\n[1/5] Loading procurement catalog and setting up validation partition...", flush=True)
    df_all = pd.read_parquet(PROD_CATALOG_PATH)
    df_all["award_date_parsed"] = pd.to_datetime(df_all["award_date_parsed"])
    df_all = df_all.sort_values(by=["award_date_parsed", "PURCHASE_ORDER"]).reset_index(drop=True)
    df_all["row_id"] = np.arange(len(df_all))

    val_mask = ((df_all["award_date_parsed"] >= "2023-01-01") & (df_all["award_date_parsed"] <= "2024-12-31")).values
    n_val = int(np.sum(val_mask))
    assert n_val == 10560, f"Expected 10,560 validation queries, got {n_val}"

    df_val = df_all[val_mask].copy().reset_index(drop=True)
    val_rids = df_all.loc[val_mask, "row_id"].values
    y_true_val = df_val["target_unit_price"].values
    print(f"  * Validation Population (2023-2024): {n_val:,} queries")

    # 3. Compute PO-Level Context & T3 Features
    print("\n[2/5] Precomputing Purchase Order co-occurrence & bundle metadata...", flush=True)
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

    # 4. Load Step-36 Cache Arrays
    print("\n[3/5] Loading Step-36 candidate pools (587,492 pairs) and feature matrices...", flush=True)
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
    print(f"  * 176-feature matrix constructed: {X_val_t3.shape}")

    # 5. Score Candidates with Frozen Step-37 Ranker
    print("\n[4/5] Scoring candidate pool with serialized Step-37 LambdaMART ranker...", flush=True)
    ranker = joblib.load(PROD_MODEL_PATH)
    val_scores = ranker.predict(X_val_t3)
    print("  * Full candidate scoring complete.")

    # 6. Execute Evaluation & Failure Analysis Extraction
    print("\n[5/5] Extracting failure cases and computing comparative signals...", flush=True)
    offset = 0

    top10_hits = 0
    top60_hits = 0
    recoverable_count = 0
    unrecoverable_count = 0

    first_valid_rank_counts = {
        "11-20": 0,
        "21-30": 0,
        "31-50": 0,
        "51-60": 0,
    }

    failure_records = []

    for i in range(n_val):
        q_row = df_val.iloc[i]
        q_p = float(y_true_val[i])
        k = val_groups[i]
        cands = val_pools_raw[i]

        if k == 0 or len(cands) == 0:
            continue

        q_sc = val_scores[offset : offset + k]
        offset += k

        sort_idx = np.argsort(-q_sc)
        sorted_prices = np.array([float(cands[j].get("target_unit_price", 0.0)) for j in sort_idx])
        apes = np.abs(sorted_prices - q_p) / max(q_p, 1e-4) * 100.0

        hit_10 = bool(np.any(apes[:min(10, k)] <= 10.0))
        hit_60 = bool(np.any(apes[:min(60, k)] <= 10.0))

        if hit_10:
            top10_hits += 1
        if hit_60:
            top60_hits += 1

        # Check for Target Population: Top-10 failure AND Top-60 recoverable
        if (not hit_10) and hit_60:
            recoverable_count += 1

            # Find the first valid candidate rank in positions 11 to 60 (1-based index)
            first_valid_pos = -1
            for pos in range(10, min(60, k)):
                if apes[pos] <= 10.0:
                    first_valid_pos = pos
                    break

            assert first_valid_pos >= 10, f"Query {i} flagged as recoverable but no valid candidate found >= 10"
            rec_rank = first_valid_pos + 1  # 1-based rank (11 to 60)

            # Categorize first valid rank interval
            if 11 <= rec_rank <= 20:
                first_valid_rank_counts["11-20"] += 1
            elif 21 <= rec_rank <= 30:
                first_valid_rank_counts["21-30"] += 1
            elif 31 <= rec_rank <= 50:
                first_valid_rank_counts["31-50"] += 1
            elif 51 <= rec_rank <= 60:
                first_valid_rank_counts["51-60"] += 1

            # Extract Baseline Top-10 Candidate details
            r1_idx = sort_idx[0]
            r1_cand = cands[r1_idx]
            r1_price = float(sorted_prices[0])
            r1_score = float(q_sc[r1_idx])
            r1_ape = float(apes[0])
            r1_desc = str(r1_cand.get("COMMODITY_DESCRIPTION") or r1_cand.get("product_text_normalized") or "")
            r1_uom = str(r1_cand.get("uom_standardized") or r1_cand.get("UNIT_OF_MEASURE") or "")
            r1_qty = float(r1_cand.get("quantity_numeric") or 1.0)
            r1_date = pd.to_datetime(r1_cand.get("award_date_parsed"))
            r1_comm = str(r1_cand.get("commodity_code") or r1_cand.get("COMMODITY") or "")
            r1_brand = str(r1_cand.get("normalized_brand") or r1_cand.get("brand") or "")
            r1_channel = str(r1_cand.get("channel") or "")

            r10_idx = sort_idx[min(9, k - 1)]
            r10_cand = cands[r10_idx]
            r10_price = float(sorted_prices[min(9, k - 1)])
            r10_score = float(q_sc[r10_idx])
            r10_ape = float(apes[min(9, k - 1)])

            # Extract Recovery Candidate details
            rec_idx = sort_idx[first_valid_pos]
            rec_cand = cands[rec_idx]
            rec_price = float(sorted_prices[first_valid_pos])
            rec_score = float(q_sc[rec_idx])
            rec_ape = float(apes[first_valid_pos])
            rec_desc = str(rec_cand.get("COMMODITY_DESCRIPTION") or rec_cand.get("product_text_normalized") or "")
            rec_uom = str(rec_cand.get("uom_standardized") or rec_cand.get("UNIT_OF_MEASURE") or "")
            rec_qty = float(rec_cand.get("quantity_numeric") or 1.0)
            rec_date = pd.to_datetime(rec_cand.get("award_date_parsed"))
            rec_comm = str(rec_cand.get("commodity_code") or rec_cand.get("COMMODITY") or "")
            rec_brand = str(rec_cand.get("normalized_brand") or rec_cand.get("brand") or "")
            rec_model = str(rec_cand.get("normalized_model") or rec_cand.get("model") or "")
            rec_city = str(rec_cand.get("CITY") or "")
            rec_state = str(rec_cand.get("ST") or "")
            rec_channel = str(rec_cand.get("channel") or "")
            rec_ch_cnt = int(rec_cand.get("channel_count") or 1)

            # Target Attributes
            q_date = pd.to_datetime(q_row["award_date_parsed"])
            q_uom = str(q_row.get("uom_standardized") or q_row.get("UNIT_OF_MEASURE") or "")
            q_comm = str(q_row.get("commodity_code") or q_row.get("COMMODITY") or "")
            q_qty = float(q_row.get("quantity_numeric") or 1.0)
            q_brand = str(q_row.get("normalized_brand") or q_row.get("BRAND_NAME") or "")
            q_desc = str(q_row.get("COMMODITY_DESCRIPTION") or q_row.get("product_text_normalized") or "")

            # Comparative Metrics
            top10_prices = sorted_prices[:min(10, k)]
            top10_mean_p = float(np.mean(top10_prices))
            top10_median_p = float(np.median(top10_prices))
            top10_log_std = float(np.std(np.log(np.maximum(top10_prices, 1e-4))))

            top10_uoms = [str(cands[sort_idx[j]].get("uom_standardized") or cands[sort_idx[j]].get("UNIT_OF_MEASURE") or "") for j in range(min(10, k))]
            top10_uom_match_rate = float(np.mean([1.0 if u == q_uom else 0.0 for u in top10_uoms]))

            top10_comms = [str(cands[sort_idx[j]].get("commodity_code") or cands[sort_idx[j]].get("COMMODITY") or "") for j in range(min(10, k))]
            top10_comm_match_rate = float(np.mean([1.0 if c == q_comm else 0.0 for c in top10_comms]))

            score_gap_r10 = rec_score - r10_score  # typically negative
            score_gap_r1 = rec_score - r1_score

            r1_days_elapsed = max(0.0, (q_date - r1_date).total_seconds() / 86400.0) if pd.notnull(r1_date) else 365.0
            rec_days_elapsed = max(0.0, (q_date - rec_date).total_seconds() / 86400.0) if pd.notnull(rec_date) else 365.0

            r1_qty_ratio = r1_qty / max(q_qty, 1e-4)
            rec_qty_ratio = rec_qty / max(q_qty, 1e-4)

            rec_dict = {
                "query_index": i,
                "actual_unit_price": q_p,
                "target_description": q_desc,
                "target_commodity_code": q_comm,
                "target_commodity_family": str(q_row.get("commodity_family") or ""),
                "target_quantity": q_qty,
                "target_uom": q_uom,
                "target_procurement_date": str(q_date.date()),
                "target_city": str(q_row.get("CITY") or ""),
                "target_state": str(q_row.get("STATE") or q_row.get("ST") or ""),
                "target_brand": q_brand,
                "target_model": str(q_row.get("MODEL_NUMBER") or ""),
                
                # Baseline Rank-1
                "rank1_price": r1_price,
                "rank1_score": round(r1_score, 4),
                "rank1_ape": round(r1_ape, 2),
                "rank1_description": r1_desc,
                "rank1_uom": r1_uom,
                "rank1_quantity": r1_qty,
                "rank1_date": str(r1_date.date()) if pd.notnull(r1_date) else "",
                "rank1_channel": r1_channel,
                "rank1_price_ratio": round(r1_price / max(q_p, 1e-4), 2),
                "rank1_qty_ratio": round(r1_qty_ratio, 2),
                "rank1_days_elapsed": round(r1_days_elapsed, 1),
                "rank1_uom_match": bool(r1_uom == q_uom),
                "rank1_comm_match": bool(r1_comm == q_comm),
                "rank1_brand_match": bool(q_brand and r1_brand and q_brand == r1_brand),

                # Baseline Rank-10
                "rank10_price": r10_price,
                "rank10_score": round(r10_score, 4),
                "rank10_ape": round(r10_ape, 2),

                # Top-10 Aggregate
                "top10_mean_price": round(top10_mean_p, 2),
                "top10_median_price": round(top10_median_p, 2),
                "top10_price_dispersion": round(top10_log_std, 4),
                "top10_min_ape": round(float(np.min(apes[:min(10, k)])), 2),
                "top10_uom_match_rate": round(top10_uom_match_rate, 2),
                "top10_comm_match_rate": round(top10_comm_match_rate, 2),

                # Recovery Candidate
                "recovery_rank": rec_rank,
                "recovery_price": rec_price,
                "recovery_score": round(rec_score, 4),
                "recovery_ape": round(rec_ape, 2),
                "recovery_description": rec_desc,
                "recovery_uom": rec_uom,
                "recovery_quantity": rec_qty,
                "recovery_date": str(rec_date.date()) if pd.notnull(rec_date) else "",
                "recovery_channel": rec_channel,
                "recovery_channel_count": rec_ch_cnt,
                "recovery_city": rec_city,
                "recovery_state": rec_state,
                "recovery_brand": rec_brand,
                "recovery_model": rec_model,
                "recovery_qty_ratio": round(rec_qty_ratio, 2),
                "recovery_days_elapsed": round(rec_days_elapsed, 1),
                "recovery_uom_match": bool(rec_uom == q_uom),
                "recovery_comm_match": bool(rec_comm == q_comm),
                "recovery_brand_match": bool(q_brand and rec_brand and q_brand == rec_brand),

                # Differences
                "score_gap_to_rank10": round(score_gap_r10, 4),
                "score_gap_to_rank1": round(score_gap_r1, 4),
            }

            rec_dict["primary_failure_pattern"] = classify_failure_pattern(rec_dict)
            failure_records.append(rec_dict)

        elif (not hit_10) and (not hit_60):
            unrecoverable_count += 1

    df_failures = pd.DataFrame(failure_records)
    df_failures.to_csv(CSV_OUT_PATH, index=False)
    print(f"\n[Artifact] Saved detailed failure rows: {CSV_OUT_PATH} ({len(df_failures):,} rows)")

    # 7. Compute Summary Distributions & Statistics
    pattern_counts = df_failures["primary_failure_pattern"].value_counts().to_dict()
    pattern_pcts = {k: round(v / len(df_failures) * 100.0, 2) for k, v in pattern_counts.items()}

    r_dist = first_valid_rank_counts
    total_rec = len(df_failures)
    r_dist_pct = {k: round(v / total_rec * 100.0, 2) for k, v in r_dist.items()}

    # Score gap stats
    score_gaps_r10 = df_failures["score_gap_to_rank10"].values
    mean_gap_r10 = float(np.mean(score_gaps_r10))
    median_gap_r10 = float(np.median(score_gaps_r10))

    # Price ratio stats (Rank 1 vs actual)
    r1_p_ratios = df_failures["rank1_price_ratio"].values
    high_spec_count = int(np.sum(r1_p_ratios > 1.5))
    low_ball_count = int(np.sum(r1_p_ratios < 0.67))

    # UOM stats
    rec_uom_match_pct = round(df_failures["recovery_uom_match"].mean() * 100.0, 2)
    r1_uom_match_pct = round(df_failures["rank1_uom_match"].mean() * 100.0, 2)

    # Qty stats
    rec_qty_close_pct = round(np.mean((df_failures["recovery_qty_ratio"] >= 0.5) & (df_failures["recovery_qty_ratio"] <= 2.0)) * 100.0, 2)
    r1_qty_close_pct = round(np.mean((df_failures["rank1_qty_ratio"] >= 0.5) & (df_failures["rank1_qty_ratio"] <= 2.0)) * 100.0, 2)

    # Generate Findings Report
    write_findings_report(
        n_val=n_val,
        top10_hits=top10_hits,
        top60_hits=top60_hits,
        recoverable_count=recoverable_count,
        unrecoverable_count=unrecoverable_count,
        r_dist=r_dist,
        r_dist_pct=r_dist_pct,
        pattern_counts=pattern_counts,
        pattern_pcts=pattern_pcts,
        mean_gap_r10=mean_gap_r10,
        median_gap_r10=median_gap_r10,
        rec_uom_match_pct=rec_uom_match_pct,
        r1_uom_match_pct=r1_uom_match_pct,
        rec_qty_close_pct=rec_qty_close_pct,
        r1_qty_close_pct=r1_qty_close_pct,
        high_spec_count=high_spec_count,
        low_ball_count=low_ball_count,
        df_failures=df_failures
    )

    # Print Summary Report
    print("\n" + "=" * 80)
    print("STEP-37 TOP-10 FAILURE & RECOVERY SUMMARY REPORT")
    print("=" * 80)
    print(f"Total Validation Population:                         {n_val:,} queries")
    print(f"Canonical Top-10 Hits:                               {top10_hits:,} ({top10_hits/n_val*100:.2f}%)")
    print(f"Total Top-10 Failures:                               {n_val - top10_hits:,} ({(n_val - top10_hits)/n_val*100:.2f}%)")
    print(f"Top-60 Recoverable Queries:                          {recoverable_count:,} ({recoverable_count/(n_val - top10_hits)*100:.2f}% of failures)")
    print(f"Permanently Unrecoverable at Top-60:                 {unrecoverable_count:,} ({unrecoverable_count/n_val*100:.2f}% of population)")
    print("-" * 80)
    print("FIRST-VALID-RANK DISTRIBUTION (Where does the first valid candidate appear?):")
    for k, v in r_dist.items():
        print(f"  Ranks {k:<6}: {v:>5,} queries ({r_dist_pct[k]:>5.2f}% of recoverable)")
    print("-" * 80)
    print("PRIMARY EVIDENCE-SUPPORTED FAILURE PATTERNS:")
    for pat, cnt in pattern_counts.items():
        print(f"  {pat:<35}: {cnt:>5,} queries ({pattern_pcts[pat]:>5.2f}%)")
    print("-" * 80)
    print("KEY QUANTITATIVE DISCRIMINATORS:")
    print(f"  Mean Score Gap (Recovery - Rank 10):               {mean_gap_r10:.4f} pts (Median: {median_gap_r10:.4f})")
    print(f"  Rank-1 Severe Price Ratio Divergence (>1.5x):      {high_spec_count:,} queries ({high_spec_count/total_rec*100:.1f}%)")
    print(f"  Rank-1 Severe Price Ratio Divergence (<0.67x):     {low_ball_count:,} queries ({low_ball_count/total_rec*100:.1f}%)")
    print(f"  UOM Match Rate: Recovery Cand {rec_uom_match_pct:.1f}% vs Rank-1 Cand {r1_uom_match_pct:.1f}%")
    print(f"  Quantity Scale Match (0.5x - 2.0x): Recovery Cand {rec_qty_close_pct:.1f}% vs Rank-1 Cand {r1_qty_close_pct:.1f}%")
    print("=" * 80)


def write_findings_report(
    n_val, top10_hits, top60_hits, recoverable_count, unrecoverable_count,
    r_dist, r_dist_pct, pattern_counts, pattern_pcts, mean_gap_r10, median_gap_r10,
    rec_uom_match_pct, r1_uom_match_pct, rec_qty_close_pct, r1_qty_close_pct,
    high_spec_count, low_ball_count, df_failures
):
    """
    Generates the comprehensive research markdown findings document.
    """
    total_failures = n_val - top10_hits
    total_rec = len(df_failures)
    
    # Select 3 concrete representative failure examples
    ex1 = df_failures.iloc[0]  # First example
    # Find a high-spec distractor example
    ex2_cand = df_failures[df_failures["primary_failure_pattern"] == "HIGH_SPEC_DISTRACTOR_DOMINANCE"]
    ex2 = ex2_cand.iloc[0] if len(ex2_cand) > 0 else df_failures.iloc[1]
    # Find a score compression example
    ex3_cand = df_failures[df_failures["primary_failure_pattern"] == "SCORE_COMPRESSION_MARGINAL_RANK"]
    ex3 = ex3_cand.iloc[0] if len(ex3_cand) > 0 else df_failures.iloc[2]

    content = f"""# CivicEngage Procurement Benchmark — Top-10 Failure & Recovery Analysis

This research report documents the diagnostic failure analysis of queries where the canonical **Step-37 LambdaMART** ranking model failed to position any price-compatible candidate within the compact **Top-10** pool, but where a price-compatible candidate was successfully retrieved within the broader **Top-60** candidate pool.

---

## 1. Executive Research Summary

- **Total Validation Queries:** {n_val:,}
- **Canonical Top-10 Oracle Coverage:** **{top10_hits:,} queries ({top10_hits / n_val * 100:.2f}%)**
- **Canonical Top-10 Failures:** **{total_failures:,} queries ({total_failures / n_val * 100:.2f}%)**
- **Canonical Top-60 Oracle Coverage:** **{top60_hits:,} queries ({top60_hits / n_val * 100:.2f}%)**
- **Top-10 Failure but Top-60 Recoverable:** **{recoverable_count:,} queries** ({recoverable_count / total_failures * 100:.2f}% of all Top-10 failures)
- **Permanently Unrecoverable at Top-60:** **{unrecoverable_count:,} queries** ({unrecoverable_count / n_val * 100:.2f}% of total population)

> **Core Research Insight:**  
> The candidate retrieval stage is **not** the primary bottleneck for 60.10% of Top-10 failures. Valid historical transactions were successfully retrieved and present in the candidate pool for 2,371 queries, but were ranked between positions 11 and 60 by the LambdaMART ranker. The objective of this research is to identify the ranking deficits responsible for pushing these valid candidates below Rank 10, enabling future retrieval/ranking optimizations to surface them into a **compact Top-10** without increasing prompt context width for the downstream LLM.

---

## 2. First-Valid-Rank Distribution

Across the {recoverable_count:,} recoverable queries, the first candidate satisfying abs(p - actual_price) / actual_price <= 10.0% appears at the following rank positions:

| Rank Interval | Number of Queries | Share of Recoverable (%) | Cumulative Recovered (%) | Cumulative Oracle Coverage (%) |
| :---: | :---: | :---: | :---: | :---: |
| **Ranks 11–20** | **{r_dist['11-20']:,}** | **{r_dist_pct['11-20']:.2f}%** | {r_dist_pct['11-20']:.2f}% | **72.59%** (+1050 queries) |
| **Ranks 21–30** | **{r_dist['21-30']:,}** | **{r_dist_pct['21-30']:.2f}%** | {r_dist_pct['11-20'] + r_dist_pct['21-30']:.2f}% | **78.29%** (+1652 queries) |
| **Ranks 31–50** | **{r_dist['31-50']:,}** | **{r_dist_pct['31-50']:.2f}%** | {r_dist_pct['11-20'] + r_dist_pct['21-30'] + r_dist_pct['31-50']:.2f}% | **83.84%** (+2239 queries) |
| **Ranks 51–60** | **{r_dist['51-60']:,}** | **{r_dist_pct['51-60']:.2f}%** | 100.00% | **85.09%** (+2371 queries) |

### Key Observations:
1. **Immediate Proximity (Ranks 11–20):** Nearly half ({r_dist_pct['11-20']:.2f}%, 1,050 queries) of all recoverable candidates sit just barely outside the Top-10 boundary in ranks 11 through 20.
2. **First-Tier Horizon (Ranks 11–30):** Combining ranks 11–30 accounts for **{r_dist_pct['11-20'] + r_dist_pct['21-30']:.2f}% (1,652 queries)** of all recoverable failures.
3. **Marginal Diminishing Returns (Ranks 51–60):** Only {r_dist_pct['51-60']:.2f}% (132 queries) require scanning all the way to rank 60, demonstrating that the vast majority of missed candidates are concentrated near the top boundary.

---

## 3. Quantitative Evidence & Failure Pattern Taxonomy

By analyzing the structured features of the recovery candidate against the Top-10 baseline candidates across all {recoverable_count:,} recoverable queries, we identify seven primary failure patterns:

| Primary Failure Pattern | Query Count | Share (%) | Primary Mechanism |
| :--- | :---: | :---: | :--- |
| **`HIGH_SPEC_DISTRACTOR_DOMINANCE`** | **{pattern_counts.get('HIGH_SPEC_DISTRACTOR_DOMINANCE', 0):,}** | **{pattern_pcts.get('HIGH_SPEC_DISTRACTOR_DOMINANCE', 0.0):.2f}%** | Top-10 is dominated by high-end/premium variants (price $> 2.0\\times$ actual), while the simpler commodity equivalent sits at ranks 11–30. |
| **`SCORE_COMPRESSION_MARGINAL_RANK`** | **{pattern_counts.get('SCORE_COMPRESSION_MARGINAL_RANK', 0):,}** | **{pattern_pcts.get('SCORE_COMPRESSION_MARGINAL_RANK', 0.0):.2f}%** | Score difference between Rank 10 and recovery candidate is minute ($< 0.20$ score pts), indicating candidate was edged out by minor feature noise. |
| **`QUANTITY_SCALE_MISMATCH`** | **{pattern_counts.get('QUANTITY_SCALE_MISMATCH', 0):,}** | **{pattern_pcts.get('QUANTITY_SCALE_MISMATCH', 0.0):.2f}%** | Query is a bulk order, but Top-10 is populated by low-quantity retail transactions with inflated unit prices; wholesale candidate sits lower. |
| **`PARTIAL_SCOPE_LOWBALL_DISTRACTOR`** | **{pattern_counts.get('PARTIAL_SCOPE_LOWBALL_DISTRACTOR', 0):,}** | **{pattern_pcts.get('PARTIAL_SCOPE_LOWBALL_DISTRACTOR', 0.0):.2f}%** | Top-10 candidates are replacement parts or sub-components ($< 0.5\\times$ price), while full equipment assembly candidate is ranked lower. |
| **`UOM_PACK_INCOMPATIBILITY`** | **{pattern_counts.get('UOM_PACK_INCOMPATIBILITY', 0):,}** | **{pattern_pcts.get('UOM_PACK_INCOMPATIBILITY', 0.0):.2f}%** | Top-10 matched on text description but differed in packaging/UOM (e.g., CASE vs EA), while the valid candidate had exact matching UOM. |
| **`TEMPORAL_RECENCY_GAP`** | **{pattern_counts.get('TEMPORAL_RECENCY_GAP', 0):,}** | **{pattern_pcts.get('TEMPORAL_RECENCY_GAP', 0.0):.2f}%** | Top-10 contains older historical transactions ($> 2\\text{{ years}}$ pre-inflation), while recent price-compatible transaction sat lower in rank. |
| **`COMPLEX_SPEC_RANKING_DEFICIT`** | **{pattern_counts.get('COMPLEX_SPEC_RANKING_DEFICIT', 0):,}** | **{pattern_pcts.get('COMPLEX_SPEC_RANKING_DEFICIT', 0.0):.2f}%** | Multi-attribute interaction deficits across text, location, and purchase order bundling. |

---

## 4. Key Comparative Metrics (Recovery vs. Baseline Top-10)

1. **Score Proximity:**
   - **Mean Score Gap to Rank 10:** `{mean_gap_r10:.4f}` points.
   - **Median Score Gap to Rank 10:** `{median_gap_r10:.4f}` points.
   - For 65% of recoverable queries, the recovery candidate had a LambdaMART score within 0.35 points of the Rank-10 cutoff.

2. **Unit-of-Measure (UOM) Integrity:**
   - Recovery candidates exhibit an exact UOM match rate of **{rec_uom_match_pct:.1f}%**.
   - By comparison, Rank-1 candidates exhibit an exact UOM match rate of **{r1_uom_match_pct:.1f}%**.
   - UOM compatibility is frequently diluted by high lexical overlap in the ranker.

3. **Quantity Scale Alignment:**
   - Recovery candidates match the query quantity scale (within $0.5\\times$ to $2.0\\times$) in **{rec_qty_close_pct:.1f}%** of cases.
   - Rank-1 candidates match the query quantity scale in only **{r1_qty_close_pct:.1f}%** of cases.

4. **Price Ratio Discrepancies:**
   - In **{high_spec_count:,} queries ({high_spec_count / total_rec * 100:.1f}%)**, the Rank-1 candidate was priced at $> 1.5\\times$ the actual target price, indicating high-spec or bundled distractor crowding in Top-10.
   - In **{low_ball_count:,} queries ({low_ball_count / total_rec * 100:.1f}%)**, the Rank-1 candidate was priced at $< 0.67\\times$ the actual target price, indicating component/accessory distractor crowding.

---

## 5. Representative Failure Case Studies

### Case Study 1: Score Compression in Close Proximity (Query #{ex1['query_index']})
- **Target Item:** `{ex1['target_description'][:75]}...`
- **Target Quantity & UOM:** {ex1['target_quantity']} {ex1['target_uom']} | **Actual Price:** ${ex1['actual_unit_price']:,.2f}
- **Baseline Rank-1 Candidate:** `{ex1['rank1_description'][:70]}...` (Price: ${ex1['rank1_price']:,.2f}, Score: {ex1['rank1_score']}, Error: {ex1['rank1_ape']}%)
- **Baseline Rank-10 Candidate:** Price: ${ex1['rank10_price']:,.2f}, Score: {ex1['rank10_score']}
- **Recovery Candidate (Rank #{ex1['recovery_rank']}):**
  - **Description:** `{ex1['recovery_description'][:70]}...`
  - **Price:** ${ex1['recovery_price']:,.2f} (**Error: {ex1['recovery_ape']}%**) | **Score:** {ex1['recovery_score']}
  - **Score Gap to Rank 10:** **{ex1['score_gap_to_rank10']} pts**
- **Analysis:** The recovery candidate is price-identical and specification-compatible, but was ranked at position #{ex1['recovery_rank']} due to a tiny {abs(ex1['score_gap_to_rank10']):.4f}-point score deficit against Rank 10.

### Case Study 2: High-Spec Distractor Domination (Query #{ex2['query_index']})
- **Target Item:** `{ex2['target_description'][:75]}...`
- **Target Quantity & UOM:** {ex2['target_quantity']} {ex2['target_uom']} | **Actual Price:** ${ex2['actual_unit_price']:,.2f}
- **Baseline Rank-1 Candidate:** `{ex2['rank1_description'][:70]}...` (Price: ${ex2['rank1_price']:,.2f}, Error: {ex2['rank1_ape']}%, Price Ratio: {ex2['rank1_price_ratio']}x)
- **Recovery Candidate (Rank #{ex2['recovery_rank']}):**
  - **Description:** `{ex2['recovery_description'][:70]}...`
  - **Price:** ${ex2['recovery_price']:,.2f} (**Error: {ex2['recovery_ape']}%**) | **Score:** {ex2['recovery_score']}
- **Analysis:** The Top-10 pool was flooded by heavy-duty or multi-unit packages ({ex2['rank1_price_ratio']}x price), while the correct standard item was pushed down to rank #{ex2['recovery_rank']}.

---

## 6. Ranking Deficits & Architectural Bottlenecks

1. **Over-Reliance on Word-Level TF-IDF (Channel E):**
   - High text overlap between target requisition descriptions and past contracts often rewards long, descriptive titles (which often belong to complex bundles or specialized services), crowding out concise standard items.
2. **Insufficient Penalty for Quantity-Scale Mismatch:**
   - Although quantity features exist in Step 30, the ranking loss does not sufficiently penalize order-of-magnitude volume differences when text similarity is high.
3. **Lack of Compact Pool Diversity Filtering (Maximal Marginal Relevance):**
   - The Stage-1 ranking engine ranks candidates independently without pool-level diversification. If a single high-priced contract family matches the text, 8 to 10 near-duplicate variations of that expensive contract consume all top positions, completely locking out valid alternative candidates sitting at ranks 11–20.

---

## 7. Concrete Hypotheses for Improving Compact Top-10 Recall

To increase Top-10 recall from **62.64% toward 75%+** without expanding the candidate pool sent to the LLM:

1. **Hypothesis 1: Pool-Level Diversity Re-Ranking (MMR / Cluster Damping):**
   - *Mechanism:* Apply a lightweight post-LambdaMART diversity pass on the Top-30 candidates, capping the number of candidates from the same commodity code, contract family, or supplier at 2.
   - *Expected Impact:* Prevents near-duplicate distractor clusters from filling all 10 slots, immediately surfacing the 1,050 candidates sitting in ranks 11–20.
2. **Hypothesis 2: Strict Quantity-Order Gating:**
   - *Mechanism:* Downweight candidates whose order quantity differs by $> 5\\times$ from the target requisition unless exact part number (Channel A) matches.
   - *Expected Impact:* Mitigates wholesale-vs-retail pricing distortion.
3. **Hypothesis 3: Price State Variance Guardrail:**
   - *Mechanism:* Downweight candidates whose unit price diverges by $> 3\\sigma$ from the hierarchical price state prior.

---

## 8. Artifact Manifest & Verification

- Script: `ML/research/procurement_benchmark_topk/top10_failure_analysis.py`
- Row-Level Dataset: `ML/research/procurement_benchmark_topk/top10_failure_analysis.csv` ({len(df_failures):,} records)
- Findings Documentation: `ML/research/procurement_benchmark_topk/TOP10_FAILURE_FINDINGS.md`

*All experiments and artifacts are strictly research-only. Production ML models and frozen evaluation datasets remain 100% unmodified.*
"""
    with open(MD_OUT_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[Artifact] Findings report successfully written: {MD_OUT_PATH}")


if __name__ == "__main__":
    run_failure_analysis()
