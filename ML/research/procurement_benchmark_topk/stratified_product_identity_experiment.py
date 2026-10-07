"""
CivicEngage Procurement Benchmark Research
Experiment 2: Stratified Product-Identity Diversity Re-Ranking (Top-30 -> Compact Top-10)

Objective:
Test whether enforcing stratified product-identity diversity across the canonical Step-37
Top-30 candidates (maximum 2 candidates per stratified identity key) improves compact
Top-10 candidate recall/coverage on the frozen evaluation population (N = 10,560 queries).

Stratified Identity Representation:
- Specific (high/medium confidence, brand/model present):
    commodity_code | brand | model | uom
- Generic (GENERIC fallback):
    commodity_code | GENERIC | clean_description_tokens[:4] | uom

Selection Rule:
- Sequential traversal of ranks 1–30.
- Maximum 2 candidates per stratified product identity.
- Output strictly 10 compact candidates for downstream LLM.
- If diversity pruning exhausts candidates before 10 are filled, backfill with highest-ranked unused.
- Anti-leakage: actual_unit_price is NEVER accessed during selection.

Outputs:
- stratified_product_identity_experiment.py
- stratified_product_identity_metrics.json
- stratified_product_identity_comparison.csv
- STRATIFIED_PRODUCT_IDENTITY_FINDINGS.md
"""

import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parent.parent.parent

PROD_CATALOG_PATH = REPO_ROOT / "ML" / "procurement_benchmark_engine" / "data" / "catalog" / "procurement_catalog.parquet"
PROD_MODEL_PATH = REPO_ROOT / "ML" / "procurement_benchmark_engine" / "models" / "procurement_ranker.joblib"
CACHE_DIR = Path(r"D:\tender_generation\ML\data\cache\step36")

METRICS_OUT_PATH = _THIS_DIR / "stratified_product_identity_metrics.json"
CSV_OUT_PATH = _THIS_DIR / "stratified_product_identity_comparison.csv"
MD_OUT_PATH = _THIS_DIR / "STRATIFIED_PRODUCT_IDENTITY_FINDINGS.md"


def build_stratified_identity(cand: Dict[str, Any]) -> str:
    """Constructs the stratified product identity key for a candidate."""
    raw_pk = str(cand.get("product_identity_key", "")).strip()
    comm = str(cand.get("commodity_code", "")).strip()
    uom = str(cand.get("uom_standardized") or cand.get("UNIT_OF_MEASURE") or "EA").strip().upper()

    # If raw key is specific (not GENERIC), retain canonical specific composite
    if raw_pk and "GENERIC" not in raw_pk:
        return raw_pk

    # Otherwise, generic fallback: refine by description tokens
    desc = str(cand.get("product_text_normalized") or cand.get("COMMODITY_DESCRIPTION") or "")
    tokens = [t.upper() for t in re.findall(r"[A-Za-z0-9]+", desc)][:4]
    desc_str = "_".join(tokens) if tokens else "GENERIC"
    return f"{comm}|GENERIC|{desc_str}|{uom}"


def run_stratified_identity_experiment():
    t_start = time.time()
    print("=" * 80)
    print("EXPERIMENT 2: STRATIFIED PRODUCT-IDENTITY DIVERSITY RE-RANKING")
    print("Population: 10,560 Validation Queries | Constraint: Max 2 per Stratified Identity")
    print("=" * 80, flush=True)

    assert PROD_CATALOG_PATH.exists(), f"Catalog missing: {PROD_CATALOG_PATH}"
    assert PROD_MODEL_PATH.exists(), f"Model missing: {PROD_MODEL_PATH}"
    assert CACHE_DIR.exists(), f"Cache missing: {CACHE_DIR}"

    # 1. Load Dataset & Construct Deterministic Validation Partition
    print("\n[1/5] Loading procurement catalog and establishing validation partition...", flush=True)
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
    print(f"  * Validation Population: {n_val:,} queries")

    # 2. Compute PO Context for T3 Features
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

    # 3. Load Step-36 Cache Arrays
    print("\n[3/5] Loading Step-36 candidate pools and feature matrices...", flush=True)
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

    # 4. Score Candidates with Frozen Step-37 Ranker
    print("\n[4/5] Scoring candidate pool with serialized Step-37 LambdaMART ranker...", flush=True)
    ranker = joblib.load(PROD_MODEL_PATH)
    val_scores = ranker.predict(X_val_t3)
    print("  * Candidate scoring complete.", flush=True)

    # 5. Execute Stratified Product Identity Selection
    print("\n[5/5] Executing Stratified Product Identity Selection (Top-30 -> Compact Top-10)...", flush=True)
    offset = 0

    comparison_records = []

    # Counters for Baseline
    base_hits_10 = 0
    base_hits_5 = 0
    base_hits_20 = 0
    base_r1_hits_10 = 0
    base_top5_hits_10 = 0
    base_r1_apes = []

    # Counters for Diversity Top-10
    div_hits_10 = 0
    div_hits_5 = 0
    div_hits_20 = 0
    div_r1_hits_10 = 0
    div_top5_hits_10 = 0
    div_r1_apes = []

    # Transition Counters
    improved_count = 0
    degraded_count = 0
    unchanged_count = 0

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
        all_sorted_cands = [cands[idx] for idx in sort_idx]
        all_sorted_scores = q_sc[sort_idx]
        all_sorted_prices = np.array([float(c.get("target_unit_price", 0.0)) for c in all_sorted_cands])
        all_sorted_apes = np.abs(all_sorted_prices - q_p) / max(q_p, 1e-4) * 100.0

        # --- Baseline Top-10 Metrics ---
        base_top10_cands = all_sorted_cands[:min(10, k)]
        base_top10_prices = all_sorted_prices[:min(10, k)]
        base_top10_apes = all_sorted_apes[:min(10, k)]

        base_hit_10 = bool(np.any(base_top10_apes <= 10.0))
        base_hit_5 = bool(np.any(base_top10_apes <= 5.0))
        base_hit_20 = bool(np.any(base_top10_apes <= 20.0))
        base_r1_hit = bool(base_top10_apes[0] <= 10.0)
        base_top5_hit = bool(np.any(base_top10_apes[:min(5, len(base_top10_apes))] <= 10.0))
        base_r1_ape = float(base_top10_apes[0])

        if base_hit_10: base_hits_10 += 1
        if base_hit_5: base_hits_5 += 1
        if base_hit_20: base_hits_20 += 1
        if base_r1_hit: base_r1_hits_10 += 1
        if base_top5_hit: base_top5_hits_10 += 1
        base_r1_apes.append(base_r1_ape)

        # --- Stratified Product Identity Selection (Top-30 -> Compact Top-10) ---
        pool_30_limit = min(30, k)
        selected_indices: List[int] = []
        selected_pos_in_top30: List[int] = []
        strat_counts: Counter = Counter()

        # Pass 1: Select up to 2 per stratified product identity
        for pos in range(pool_30_limit):
            c_item = all_sorted_cands[pos]
            s_key = build_stratified_identity(c_item)
            if strat_counts[s_key] < 2:
                selected_indices.append(pos)
                selected_pos_in_top30.append(pos + 1)
                strat_counts[s_key] += 1
                if len(selected_indices) == 10:
                    break

        # Pass 2: Backfill if diversity constraint was too strict to fill 10 candidates
        if len(selected_indices) < min(10, k):
            for pos in range(pool_30_limit):
                if pos not in selected_indices:
                    selected_indices.append(pos)
                    selected_pos_in_top30.append(pos + 1)
                    if len(selected_indices) == min(10, k):
                        break

        div_top10_cands = [all_sorted_cands[idx] for idx in selected_indices]
        div_top10_prices = all_sorted_prices[selected_indices]
        div_top10_apes = all_sorted_apes[selected_indices]
        div_top10_scores = all_sorted_scores[selected_indices]

        div_hit_10 = bool(np.any(div_top10_apes <= 10.0))
        div_hit_5 = bool(np.any(div_top10_apes <= 5.0))
        div_hit_20 = bool(np.any(div_top10_apes <= 20.0))
        div_r1_hit = bool(div_top10_apes[0] <= 10.0)
        div_top5_hit = bool(np.any(div_top10_apes[:min(5, len(div_top10_apes))] <= 10.0))
        div_r1_ape = float(div_top10_apes[0])

        if div_hit_10: div_hits_10 += 1
        if div_hit_5: div_hits_5 += 1
        if div_hit_20: div_hits_20 += 1
        if div_r1_hit: div_r1_hits_10 += 1
        if div_top5_hit: div_top5_hits_10 += 1
        div_r1_apes.append(div_r1_ape)

        # Transition status
        if (not base_hit_10) and div_hit_10:
            status = "IMPROVED"
            improved_count += 1
        elif base_hit_10 and (not div_hit_10):
            status = "DEGRADED"
            degraded_count += 1
        else:
            status = "UNCHANGED"
            unchanged_count += 1

        # Determine best candidate in diversity selection
        best_div_subidx = int(np.argmin(div_top10_apes))
        best_div_cand = div_top10_cands[best_div_subidx]
        best_div_price = float(div_top10_prices[best_div_subidx])
        best_div_ape = float(div_top10_apes[best_div_subidx])
        best_div_score = float(div_top10_scores[best_div_subidx])
        best_div_orig_rank = int(selected_pos_in_top30[best_div_subidx])

        # Record detail
        comparison_records.append({
            "query_index": i,
            "actual_unit_price": round(q_p, 4),
            "target_description": str(q_row["COMMODITY_DESCRIPTION"])[:120],
            "target_commodity": str(q_row["COMMODITY"]),
            "target_quantity": float(q_row["quantity_numeric"]),
            "target_uom": str(q_row["uom_standardized"]),
            "baseline_top10_hit": base_hit_10,
            "diversity_top10_hit": div_hit_10,
            "transition_status": status,
            # Baseline Rank-1
            "baseline_r1_price": round(float(base_top10_prices[0]), 4),
            "baseline_r1_score": round(float(all_sorted_scores[0]), 4),
            "baseline_r1_ape": round(base_r1_ape, 2),
            "baseline_r1_po": str(base_top10_cands[0].get("PURCHASE_ORDER")),
            "baseline_r1_desc": str(base_top10_cands[0].get("COMMODITY_DESCRIPTION"))[:80],
            # Diversity Rank-1
            "diversity_r1_price": round(float(div_top10_prices[0]), 4),
            "diversity_r1_score": round(float(div_top10_scores[0]), 4),
            "diversity_r1_ape": round(div_r1_ape, 2),
            "diversity_r1_po": str(div_top10_cands[0].get("PURCHASE_ORDER")),
            "diversity_r1_orig_rank": int(selected_pos_in_top30[0]),
            # Best Diversity Candidate Surfaced
            "best_div_price": round(best_div_price, 4),
            "best_div_score": round(best_div_score, 4),
            "best_div_ape": round(best_div_ape, 2),
            "best_div_orig_rank_in_top30": best_div_orig_rank,
            "best_div_po": str(best_div_cand.get("PURCHASE_ORDER")),
            "best_div_desc": str(best_div_cand.get("COMMODITY_DESCRIPTION"))[:80],
            "best_div_strat_key": build_stratified_identity(best_div_cand),
            "best_div_raw_key": str(best_div_cand.get("product_identity_key")),
            "best_div_conf": str(best_div_cand.get("product_identity_confidence")),
            "selected_ranks_from_top30": ",".join(map(str, selected_pos_in_top30)),
        })

    # Save Comparison CSV
    print(f"\nWriting row-level comparison to {CSV_OUT_PATH}...", flush=True)
    df_cmp = pd.DataFrame(comparison_records)
    df_cmp.to_csv(CSV_OUT_PATH, index=False)

    # Compute Summary Statistics
    base_acc_10 = round(base_hits_10 / n_val * 100.0, 2)
    div_acc_10 = round(div_hits_10 / n_val * 100.0, 2)
    abs_imp = round(div_acc_10 - base_acc_10, 2)
    rel_imp = round((div_acc_10 - base_acc_10) / base_acc_10 * 100.0, 2)

    base_top5_acc = round(base_top5_hits_10 / n_val * 100.0, 2)
    div_top5_acc = round(div_top5_hits_10 / n_val * 100.0, 2)
    base_r1_acc = round(base_r1_hits_10 / n_val * 100.0, 2)
    div_r1_acc = round(div_r1_hits_10 / n_val * 100.0, 2)

    base_acc_5 = round(base_hits_5 / n_val * 100.0, 2)
    div_acc_5 = round(div_hits_5 / n_val * 100.0, 2)
    base_acc_20 = round(base_hits_20 / n_val * 100.0, 2)
    div_acc_20 = round(div_hits_20 / n_val * 100.0, 2)

    base_mean_ape = round(float(np.mean(base_r1_apes)), 2)
    base_mdape = round(float(np.median(base_r1_apes)), 2)
    div_mean_ape = round(float(np.mean(div_r1_apes)), 2)
    div_mdape = round(float(np.median(div_r1_apes)), 2)

    metrics_dict = {
        "experiment_name": "stratified_product_identity_top30_max2",
        "total_queries": n_val,
        "runtime_seconds": round(time.time() - t_start, 2),
        "constraint": "maximum_2_candidates_per_stratified_product_identity",
        "search_pool_depth": 30,
        "final_output_depth": 10,
        "primary_metric": "Oracle_Accuracy_10pct",
        "comparison": {
            "baseline_top10_oracle_coverage_10pct": base_acc_10,
            "diversity_top10_oracle_coverage_10pct": div_acc_10,
            "absolute_improvement_pp": abs_imp,
            "relative_improvement_pct": rel_imp,
            "baseline_top5_coverage_10pct": base_top5_acc,
            "diversity_top5_coverage_10pct": div_top5_acc,
            "baseline_r1_accuracy_10pct": base_r1_acc,
            "diversity_r1_accuracy_10pct": div_r1_acc,
            "baseline_accuracy_5pct": base_acc_5,
            "diversity_accuracy_5pct": div_acc_5,
            "baseline_accuracy_20pct": base_acc_20,
            "diversity_accuracy_20pct": div_acc_20,
            "baseline_r1_mean_ape": base_mean_ape,
            "baseline_r1_median_ape": base_mdape,
            "diversity_r1_mean_ape": div_mean_ape,
            "diversity_r1_median_ape": div_mdape,
        },
        "query_transitions": {
            "improved_queries_count": improved_count,
            "improved_queries_pct": round(improved_count / n_val * 100.0, 2),
            "degraded_queries_count": degraded_count,
            "degraded_queries_pct": round(degraded_count / n_val * 100.0, 2),
            "unchanged_queries_count": unchanged_count,
            "unchanged_queries_pct": round(unchanged_count / n_val * 100.0, 2),
            "net_queries_gained": improved_count - degraded_count,
        },
    }

    with open(METRICS_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    print("\n" + "=" * 80)
    print("EXPERIMENT 2 BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Baseline Top-10 Oracle Coverage @ 10%:  {base_acc_10:.2f}% ({base_hits_10:,} / {n_val:,})")
    print(f"Stratified Diversity Top-10 @ 10%:      {div_acc_10:.2f}% ({div_hits_10:,} / {n_val:,})")
    print(f"Absolute Improvement:                    {abs_imp:+.2f} percentage points")
    print(f"Relative Improvement:                    {rel_imp:+.2f}%")
    print(f"Baseline Top-5 Coverage:                 {base_top5_acc:.2f}%")
    print(f"Diversity Top-5 Coverage:                {div_top5_acc:.2f}%")
    print(f"Transitions:")
    print(f"  * Improved queries:                    {improved_count:,} ({improved_count/n_val*100:.2f}%)")
    print(f"  * Degraded queries:                    {degraded_count:,} ({degraded_count/n_val*100:.2f}%)")
    print(f"  * Unchanged queries:                   {unchanged_count:,} ({unchanged_count/n_val*100:.2f}%)")
    print(f"  * Net Gain / Loss:                     {improved_count - degraded_count:+,} queries")
    print("=" * 80, flush=True)

    # 6. Generate Markdown Findings Report
    write_findings_markdown(metrics_dict, df_cmp)


def write_findings_markdown(m: Dict[str, Any], df_cmp: pd.DataFrame):
    comp = m["comparison"]
    trans = m["query_transitions"]

    improved_df = df_cmp[df_cmp["transition_status"] == "IMPROVED"].head(5)
    degraded_df = df_cmp[df_cmp["transition_status"] == "DEGRADED"].head(5)

    md = f"""# CivicEngage Procurement Benchmark — Stratified Product-Identity Diversity Findings

This research report documents the empirical results of **Experiment 2: Stratified Product-Identity Diversity Selection** evaluated on the canonical frozen Step-37 evaluation population ($N = 10,560$ validation queries).

---

## 1. Research Objective & Experimental Architecture

- **Research Problem:** In Experiment 1, contract-level diversity (`PURCHASE_ORDER` cap at 2) was found to be slightly negative (-0.20 pp net) because multiple legitimate line items on the same multi-line municipal contract were discarded.
- **Hypothesis:** Diversity based on **Stratified Product Identity** protects valid multi-item contracts while pruning genuine duplicate product specifications (same brand+model or identical description tokens).
- **Stratified Representation Rule:**
  - **Specific / High Confidence:** `commodity_code | brand | model | uom`
  - **Generic / Low Confidence:** `commodity_code | GENERIC | clean_description_tokens[:4] | uom`
- **Selection Rule:** Sequential traversal of canonical LambdaMART ranks 1–30 with a maximum of 2 candidates per stratified product identity, producing **exactly 10 compact candidates** for each query.

```
Canonical Retrieval (10 Channels)
             ↓
LambdaMART 176-Feature Ranking (Top-30 Ranked Pool)
             ↓
Stratified Product-Identity Filter (Max 2 Candidates per Stratified Key)
             ↓
COMPACT TOP-10 CANDIDATE SET (Sent to downstream LLM)
```

---

## 2. Experimental Benchmark Results

| Metric | Canonical Baseline Top-10 | Stratified Diversity Top-10 | Absolute Delta | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Top-10 Oracle Coverage ($\\le 10\\%$ Error)** | **{comp['baseline_top10_oracle_coverage_10pct']:.2f}%** | **{comp['diversity_top10_oracle_coverage_10pct']:.2f}%** | **{comp['absolute_improvement_pp']:+.2f} pp** | **{comp['relative_improvement_pct']:+.2f}%** |
| **Top-5 Oracle Coverage ($\\le 10\\%$ Error)** | {comp['baseline_top5_coverage_10pct']:.2f}% | {comp['diversity_top5_coverage_10pct']:.2f}% | {comp['diversity_top5_coverage_10pct'] - comp['baseline_top5_coverage_10pct']:+.2f} pp | {((comp['diversity_top5_coverage_10pct'] - comp['baseline_top5_coverage_10pct']) / comp['baseline_top5_coverage_10pct'] * 100.0):+.2f}% |
| **Rank-1 Accuracy ($\\le 10\\%$ Error)** | {comp['baseline_r1_accuracy_10pct']:.2f}% | {comp['diversity_r1_accuracy_10pct']:.2f}% | +0.00 pp | +0.00% |
| **Tight Accuracy ($\\le 5\\%$ Error)** | {comp['baseline_accuracy_5pct']:.2f}% | {comp['diversity_accuracy_5pct']:.2f}% | {comp['diversity_accuracy_5pct'] - comp['baseline_accuracy_5pct']:+.2f} pp | {((comp['diversity_accuracy_5pct'] - comp['baseline_accuracy_5pct']) / comp['baseline_accuracy_5pct'] * 100.0):+.2f}% |
| **Broad Accuracy ($\\le 20\\%$ Error)** | {comp['baseline_accuracy_20pct']:.2f}% | {comp['diversity_accuracy_20pct']:.2f}% | {comp['diversity_accuracy_20pct'] - comp['baseline_accuracy_20pct']:+.2f} pp | {((comp['diversity_accuracy_20pct'] - comp['baseline_accuracy_20pct']) / comp['baseline_accuracy_20pct'] * 100.0):+.2f}% |
| **Rank-1 Mean APE** | {comp['baseline_r1_mean_ape']:.2f}% | {comp['diversity_r1_mean_ape']:.2f}% | +0.00% | — |
| **Rank-1 Median APE** | {comp['baseline_r1_median_ape']:.2f}% | {comp['diversity_r1_median_ape']:.2f}% | +0.00% | — |

---

## 3. Query Transition Analysis

Across the 10,560 validation queries:

- **Improved Queries (Baseline Failed $\\to$ Diversity Succeeded):** **{trans['improved_queries_count']:,} queries** ({trans['improved_queries_pct']:.2f}%)
- **Degraded Queries (Baseline Succeeded $\\to$ Diversity Failed):** **{trans['degraded_queries_count']:,} queries** ({trans['degraded_queries_pct']:.2f}%)
- **Unchanged Queries:** **{trans['unchanged_queries_count']:,} queries** ({trans['unchanged_queries_pct']:.2f}%)
- **Net Query Shift:** **{trans['net_queries_gained']:+,} queries** ({comp['absolute_improvement_pp']:+.2f} percentage points)

---

## 4. Case Studies: Successful Recovery vs. Degradation

### A. Representative Recovery Examples
"""
    for _, r in improved_df.iterrows():
        md += f"""
- **Query #{int(r['query_index'])}:** `{r['target_description']}`
  - Target Price: ${r['actual_unit_price']:,.2f}
  - Baseline Rank-1: ${r['baseline_r1_price']:,.2f} (Error: {r['baseline_r1_ape']:.1f}%, PO: `{r['baseline_r1_po']}`)
  - Recovered Candidate: ${r['best_div_price']:,.2f} (**Error: {r['best_div_ape']:.1f}%**, Orig Rank: #{int(r['best_div_orig_rank_in_top30'])})
  - Stratified Key: `{r['best_div_strat_key']}`
"""

    md += """
### B. Representative Degradation Examples
"""
    for _, r in degraded_df.iterrows():
        md += f"""
- **Query #{int(r['query_index'])}:** `{r['target_description']}`
  - Target Price: ${r['actual_unit_price']:,.2f}
  - Baseline Rank-1: ${r['baseline_r1_price']:,.2f} (Error: {r['baseline_r1_ape']:.1f}%, PO: `{r['baseline_r1_po']}`)
  - Best Candidate Remaining: ${r['best_div_price']:,.2f} (Error: {r['best_div_ape']:.1f}%)
  - Degradation Cause: Valid candidate at original rank shared stratified key with two earlier candidates, resulting in exclusion.
"""

    md += f"""
---

## 5. Architectural Takeaways & Recommendation

1. **Comparison to Experiment 1 (Contract Diversity):**
   - Contract diversity (max 2 per `PURCHASE_ORDER`) resulted in: **{comp['absolute_improvement_pp']:.2f} pp net**.
   - Stratified product identity evaluates the actual commodity specifications, preventing duplicate items while preserving diverse contract line items.
2. **Discrete Cutoff Limitations:**
   - Any hard discrete ceiling (whether at the contract level or stratified key level) creates an all-or-nothing threshold where a highly relevant 3rd candidate can be discarded in favor of a weak 25th candidate.
3. **Next Step:**
   - Proceed to **Maximal Marginal Relevance (MMR)**: MMR replaces hard binary truncation with a smooth penalty decay proportional to candidate similarity, ensuring that genuinely strong candidates can still survive while suppressing near-duplicates.

---

*All experiments are research-only. Production ML inference and canonical frozen evaluation datasets remain 100% unmodified.*
"""

    with open(MD_OUT_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Findings report written to {MD_OUT_PATH}", flush=True)


if __name__ == "__main__":
    run_stratified_identity_experiment()
