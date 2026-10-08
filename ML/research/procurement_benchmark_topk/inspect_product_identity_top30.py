"""
CivicEngage Procurement Benchmark Research
Experiment 2 - Product-Identity Diversity
Part 1: Inspection & Candidate Identity Analysis across Canonical Top-30 Pools

Questions to answer:
1. Available product identity fields and unique counts.
2. Missing / null rates across catalog, val pools, and Top-30.
3. Repetition statistics within Top-30 (identical product_identity_key, identity_key,
   normalized_product_name, commodity_code, brand+model).
4. Distribution of max identity frequency per query (queries with 2+, 3+, 5+ same identities).
5. Concrete examples of repeated identities.
6. Fallback analysis for missing/generic values.
"""

import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parent.parent.parent

PROD_CATALOG_PATH = REPO_ROOT / "ML" / "procurement_benchmark_engine" / "data" / "catalog" / "procurement_catalog.parquet"
PROD_MODEL_PATH = REPO_ROOT / "ML" / "procurement_benchmark_engine" / "models" / "procurement_ranker.joblib"
CACHE_DIR = Path(r"D:\tender_generation\ML\data\cache\step36")
OUT_JSON = _THIS_DIR / "product_identity_inspection.json"


def main():
    t0 = time.time()
    print("=" * 80)
    print("PART 1 INSPECTION: PRODUCT IDENTITY ACROSS CANONICAL TOP-30 CANDIDATE POOLS")
    print("=" * 80, flush=True)

    # 1. Inspect Full Catalog Columns & Missingness
    print("\n[1/4] Inspecting Catalog Parquet...", flush=True)
    df_cat = pd.read_parquet(PROD_CATALOG_PATH)
    n_cat = len(df_cat)
    print(f"Catalog total rows: {n_cat:,}")

    candidate_identity_fields = [
        "product_identity_key",
        "product_identity_confidence",
        "commodity_code",
        "commodity_family",
        "brand",
        "model",
        "product_text_normalized",
        "COMMODITY_DESCRIPTION",
        "EXTENDED_DESCRIPTION",
    ]

    catalog_field_stats = {}
    for col in candidate_identity_fields:
        if col in df_cat.columns:
            s = df_cat[col]
            # Handle 'nan' string or actual NaN
            null_count = int(s.isna().sum() + (s.astype(str) == "nan").sum() + (s.astype(str) == "None").sum())
            unique_count = int(s.nunique())
            catalog_field_stats[col] = {
                "null_count": null_count,
                "null_pct": round(null_count / n_cat * 100.0, 3),
                "unique_count": unique_count,
            }
            print(f"  Catalog [{col}]: nulls={null_count:,} ({null_count/n_cat*100:.2f}%), uniques={unique_count:,}")

    # 2. Setup Validation Partition & T3 Features
    print("\n[2/4] Setting up validation partition and scoring Top-30 pools...", flush=True)
    df_cat["award_date_parsed"] = pd.to_datetime(df_cat["award_date_parsed"])
    df_cat = df_cat.sort_values(by=["award_date_parsed", "PURCHASE_ORDER"]).reset_index(drop=True)
    df_cat["row_id"] = np.arange(len(df_cat))

    val_mask = ((df_cat["award_date_parsed"] >= "2023-01-01") & (df_cat["award_date_parsed"] <= "2024-12-31")).values
    n_val = int(np.sum(val_mask))
    val_rids = df_cat.loc[val_mask, "row_id"].values
    print(f"Validation queries: {n_val:,}")

    po_comm_qty = df_cat.groupby(["PURCHASE_ORDER", "COMMODITY"])["quantity_numeric"].sum().reset_index()
    po_tot_qty_series = po_comm_qty.groupby("PURCHASE_ORDER")["quantity_numeric"].transform("sum")
    po_comm_qty["share_sq"] = (po_comm_qty["quantity_numeric"] / np.maximum(po_tot_qty_series, 1e-4)) ** 2
    po_hhi_series = po_comm_qty.groupby("PURCHASE_ORDER")["share_sq"].sum().rename("po_hhi")
    df_cat = df_cat.merge(po_hhi_series, on="PURCHASE_ORDER", how="left")

    po_total_qtys = df_cat.groupby("PURCHASE_ORDER")["quantity_numeric"].transform("sum").values.astype(np.float32)
    commodities = df_cat["COMMODITY"].values
    quantities = df_cat["quantity_numeric"].values.astype(np.float32)
    po_hhis = df_cat["po_hhi"].fillna(1.0).values.astype(np.float32)
    po_max_qty = df_cat.groupby("PURCHASE_ORDER")["quantity_numeric"].transform("max").values.astype(np.float32)
    is_primary_item = (quantities >= (po_max_qty - 1e-4)).astype(np.float32)
    po_comm_sets = df_cat.groupby("PURCHASE_ORDER")["COMMODITY"].apply(set).to_dict()
    po_keys = df_cat["PURCHASE_ORDER"].values

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
    print("  * Candidate ranking scores computed.", flush=True)

    # 3. Analyze Top-30 Candidates Across All 10,560 Queries
    print("\n[3/4] Analyzing Product Identity distribution in Top-30...", flush=True)
    offset = 0

    total_top30_candidates = 0
    unique_product_identity_keys = set()
    unique_identity_keys = set()
    unique_norm_names = set()
    unique_commodity_codes = set()
    unique_brand_models = set()

    missing_prod_ident_key = 0
    generic_prod_ident_key = 0

    # Query level repetition distributions
    queries_with_rep_prod_key_ge2 = 0
    queries_with_rep_prod_key_ge3 = 0
    queries_with_rep_prod_key_ge5 = 0
    queries_with_rep_prod_key_ge10 = 0

    queries_with_rep_ident_key_ge2 = 0
    queries_with_rep_ident_key_ge3 = 0
    queries_with_rep_ident_key_ge5 = 0

    queries_with_rep_name_ge2 = 0
    queries_with_rep_name_ge3 = 0
    queries_with_rep_name_ge5 = 0

    queries_with_rep_comm_ge2 = 0
    queries_with_rep_comm_ge3 = 0
    queries_with_rep_comm_ge5 = 0

    queries_with_rep_bm_ge2 = 0
    queries_with_rep_bm_ge3 = 0
    queries_with_rep_bm_ge5 = 0

    # Candidate level repetition counts (within query)
    cand_count_rep_prod_key = 0
    cand_count_rep_ident_key = 0
    cand_count_rep_name = 0
    cand_count_rep_comm = 0
    cand_count_rep_bm = 0

    max_reps_prod_key = []
    repetition_examples = []

    for i in range(n_val):
        k = val_groups[i]
        cands = val_pools_raw[i]
        if k == 0 or len(cands) == 0:
            continue

        q_sc = val_scores[offset : offset + k]
        offset += k

        sort_idx = np.argsort(-q_sc)
        top30_cands = [cands[idx] for idx in sort_idx[:min(30, k)]]
        n_top = len(top30_cands)
        total_top30_candidates += n_top

        # Collect keys within query
        q_prod_keys = []
        q_ident_keys = []
        q_names = []
        q_comms = []
        q_bms = []

        for c in top30_cands:
            pk = str(c.get("product_identity_key", "")).strip()
            ik = str(c.get("identity_key", "")).strip()
            nm = str(c.get("normalized_product_name", "")).strip()
            cm = str(c.get("commodity_code", "")).strip()
            br = str(c.get("brand", "")).strip()
            md = str(c.get("model", "")).strip()
            bm = f"{br}|{md}" if (br not in ("", "nan", "None") or md not in ("", "nan", "None")) else "UNKNOWN_BM"

            if pk in ("", "nan", "None"):
                missing_prod_ident_key += 1
            if "|GENERIC|" in pk:
                generic_prod_ident_key += 1

            unique_product_identity_keys.add(pk)
            unique_identity_keys.add(ik)
            unique_norm_names.add(nm)
            unique_commodity_codes.add(cm)
            if bm != "UNKNOWN_BM":
                unique_brand_models.add(bm)

            q_prod_keys.append(pk)
            q_ident_keys.append(ik)
            q_names.append(nm)
            q_comms.append(cm)
            q_bms.append(bm)

        c_pk = Counter(q_prod_keys)
        c_ik = Counter(q_ident_keys)
        c_nm = Counter(q_names)
        c_cm = Counter(q_comms)
        # For brand+model, filter out unknown
        c_bm = Counter([x for x in q_bms if x != "UNKNOWN_BM"])

        max_pk = max(c_pk.values()) if c_pk else 0
        max_reps_prod_key.append(max_pk)

        # Repetition flags
        if any(v >= 2 for v in c_pk.values()): queries_with_rep_prod_key_ge2 += 1
        if any(v >= 3 for v in c_pk.values()): queries_with_rep_prod_key_ge3 += 1
        if any(v >= 5 for v in c_pk.values()): queries_with_rep_prod_key_ge5 += 1
        if any(v >= 10 for v in c_pk.values()): queries_with_rep_prod_key_ge10 += 1

        if any(v >= 2 for v in c_ik.values()): queries_with_rep_ident_key_ge2 += 1
        if any(v >= 3 for v in c_ik.values()): queries_with_rep_ident_key_ge3 += 1
        if any(v >= 5 for v in c_ik.values()): queries_with_rep_ident_key_ge5 += 1

        if any(v >= 2 for v in c_nm.values()): queries_with_rep_name_ge2 += 1
        if any(v >= 3 for v in c_nm.values()): queries_with_rep_name_ge3 += 1
        if any(v >= 5 for v in c_nm.values()): queries_with_rep_name_ge5 += 1

        if any(v >= 2 for v in c_cm.values()): queries_with_rep_comm_ge2 += 1
        if any(v >= 3 for v in c_cm.values()): queries_with_rep_comm_ge3 += 1
        if any(v >= 5 for v in c_cm.values()): queries_with_rep_comm_ge5 += 1

        if any(v >= 2 for v in c_bm.values()): queries_with_rep_bm_ge2 += 1
        if any(v >= 3 for v in c_bm.values()): queries_with_rep_bm_ge3 += 1
        if any(v >= 5 for v in c_bm.values()): queries_with_rep_bm_ge5 += 1

        # Count how many candidate positions are redundant (repeats after 1st appearance)
        cand_count_rep_prod_key += sum(v - 1 for v in c_pk.values() if v > 1)
        cand_count_rep_ident_key += sum(v - 1 for v in c_ik.values() if v > 1)
        cand_count_rep_name += sum(v - 1 for v in c_nm.values() if v > 1)
        cand_count_rep_comm += sum(v - 1 for v in c_cm.values() if v > 1)
        cand_count_rep_bm += sum(v - 1 for v in c_bm.values() if v > 1)

        # Store interesting repetition examples
        if len(repetition_examples) < 5 and max_pk >= 5:
            most_common_pk, count = c_pk.most_common(1)[0]
            # Gather descriptions of those repeated candidates
            repeated_descs = [
                (c.get("COMMODITY_DESCRIPTION"), c.get("PURCHASE_ORDER"), c.get("target_unit_price"))
                for c in top30_cands if str(c.get("product_identity_key", "")).strip() == most_common_pk
            ]
            repetition_examples.append({
                "query_index": i,
                "repeated_product_identity_key": most_common_pk,
                "repetition_count": count,
                "sample_candidates": repeated_descs[:4],
            })

    results = {
        "total_validation_queries": n_val,
        "total_top30_candidates_evaluated": total_top30_candidates,
        "mean_candidates_per_query_in_top30": round(total_top30_candidates / n_val, 2),
        "catalog_field_stats": catalog_field_stats,
        "unique_counts_in_top30": {
            "unique_product_identity_keys": len(unique_product_identity_keys),
            "unique_identity_keys": len(unique_identity_keys),
            "unique_normalized_product_names": len(unique_norm_names),
            "unique_commodity_codes": len(unique_commodity_codes),
            "unique_brand_models": len(unique_brand_models),
        },
        "missingness_in_top30": {
            "missing_product_identity_key_count": missing_prod_ident_key,
            "missing_product_identity_key_pct": round(missing_prod_ident_key / total_top30_candidates * 100.0, 4),
            "generic_product_identity_key_count": generic_prod_ident_key,
            "generic_product_identity_key_pct": round(generic_prod_ident_key / total_top30_candidates * 100.0, 2),
        },
        "candidate_level_redundancy_in_top30": {
            "redundant_candidates_by_product_identity_key": cand_count_rep_prod_key,
            "redundant_candidates_by_identity_key": cand_count_rep_ident_key,
            "redundant_candidates_by_normalized_name": cand_count_rep_name,
            "redundant_candidates_by_commodity_code": cand_count_rep_comm,
            "redundant_candidates_by_brand_model": cand_count_rep_bm,
        },
        "query_level_repetition_rates_in_top30": {
            "product_identity_key": {
                "queries_with_ge2_same": queries_with_rep_prod_key_ge2,
                "queries_with_ge2_pct": round(queries_with_rep_prod_key_ge2 / n_val * 100.0, 2),
                "queries_with_ge3_same": queries_with_rep_prod_key_ge3,
                "queries_with_ge3_pct": round(queries_with_rep_prod_key_ge3 / n_val * 100.0, 2),
                "queries_with_ge5_same": queries_with_rep_prod_key_ge5,
                "queries_with_ge5_pct": round(queries_with_rep_prod_key_ge5 / n_val * 100.0, 2),
                "queries_with_ge10_same": queries_with_rep_prod_key_ge10,
                "queries_with_ge10_pct": round(queries_with_rep_prod_key_ge10 / n_val * 100.0, 2),
            },
            "identity_key": {
                "queries_with_ge2_same": queries_with_rep_ident_key_ge2,
                "queries_with_ge2_pct": round(queries_with_rep_ident_key_ge2 / n_val * 100.0, 2),
                "queries_with_ge3_same": queries_with_rep_ident_key_ge3,
                "queries_with_ge3_pct": round(queries_with_rep_ident_key_ge3 / n_val * 100.0, 2),
                "queries_with_ge5_same": queries_with_rep_ident_key_ge5,
                "queries_with_ge5_pct": round(queries_with_rep_ident_key_ge5 / n_val * 100.0, 2),
            },
            "normalized_product_name": {
                "queries_with_ge2_same": queries_with_rep_name_ge2,
                "queries_with_ge2_pct": round(queries_with_rep_name_ge2 / n_val * 100.0, 2),
                "queries_with_ge3_same": queries_with_rep_name_ge3,
                "queries_with_ge3_pct": round(queries_with_rep_name_ge3 / n_val * 100.0, 2),
                "queries_with_ge5_same": queries_with_rep_name_ge5,
                "queries_with_ge5_pct": round(queries_with_rep_name_ge5 / n_val * 100.0, 2),
            },
            "commodity_code": {
                "queries_with_ge2_same": queries_with_rep_comm_ge2,
                "queries_with_ge2_pct": round(queries_with_rep_comm_ge2 / n_val * 100.0, 2),
                "queries_with_ge3_same": queries_with_rep_comm_ge3,
                "queries_with_ge3_pct": round(queries_with_rep_comm_ge3 / n_val * 100.0, 2),
                "queries_with_ge5_same": queries_with_rep_comm_ge5,
                "queries_with_ge5_pct": round(queries_with_rep_comm_ge5 / n_val * 100.0, 2),
            },
            "brand_model": {
                "queries_with_ge2_same": queries_with_rep_bm_ge2,
                "queries_with_ge2_pct": round(queries_with_rep_bm_ge2 / n_val * 100.0, 2),
                "queries_with_ge3_same": queries_with_rep_bm_ge3,
                "queries_with_ge3_pct": round(queries_with_rep_bm_ge3 / n_val * 100.0, 2),
                "queries_with_ge5_same": queries_with_rep_bm_ge5,
                "queries_with_ge5_pct": round(queries_with_rep_bm_ge5 / n_val * 100.0, 2),
            },
        },
        "repetition_examples": repetition_examples,
        "runtime_seconds": round(time.time() - t0, 2),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[4/4] Inspection complete in {results['runtime_seconds']}s.")
    print(f"Results saved to: {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
