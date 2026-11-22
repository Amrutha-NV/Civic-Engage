import sys
from pathlib import Path
import math
import re
import joblib
import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parent.parent.parent

PROD_CATALOG_PATH = REPO_ROOT / "ML" / "procurement_benchmark_engine" / "data" / "catalog" / "procurement_catalog.parquet"
PROD_MODEL_PATH = REPO_ROOT / "ML" / "procurement_benchmark_engine" / "models" / "procurement_ranker.joblib"
CACHE_DIR = Path(r"D:\tender_generation\ML\data\cache\step36")
SIGNALS_CSV_PATH = _THIS_DIR / "selective_mmr_signal_analysis.csv"

sys.path.insert(0, str(REPO_ROOT))

from ML.research.procurement_benchmark_topk.selective_mmr_experiment import (
    get_tokens,
    build_stratified_identity,
    run_mmr_selection,
    tune_threshold_on_training_fold,
)

RANDOM_SEED = 42
OUTPUT_PATH = _THIS_DIR / "selective_mmr_top10_candidates.csv"

print("[1/6] Loading frozen research artifacts...", flush=True)

df_signals = pd.read_csv(SIGNALS_CSV_PATH)
df_all = pd.read_parquet(PROD_CATALOG_PATH)

val_rids = np.load(CACHE_DIR / "val_rids.npy")
y_true_val = np.load(CACHE_DIR / "y_val.npy")
val_pools_raw = joblib.load(CACHE_DIR / "val_pools.joblib")
X_val_t0 = np.load(CACHE_DIR / "X_val_t0.npy")
val_groups = np.load(CACHE_DIR / "val_groups.npy").tolist()

n_val = len(y_true_val)

print(f"  Validation queries: {n_val}", flush=True)

print("[2/6] Rebuilding exact LambdaMART candidate scores...", flush=True)

po_total_qtys = df_all.groupby("PURCHASE_ORDER")["quantity_numeric"].transform("sum").values.astype(np.float32)
commodities = df_all["COMMODITY"].values
quantities = df_all["quantity_numeric"].values.astype(np.float32)
po_hhis = df_all["po_hhi"].fillna(1.0).values.astype(np.float32)
po_max_qty = df_all.groupby("PURCHASE_ORDER")["quantity_numeric"].transform("max").values.astype(np.float32)
is_primary_item = (quantities >= (po_max_qty - 1e-4)).astype(np.float32)
po_comm_sets = df_all.groupby("PURCHASE_ORDER")["COMMODITY"].apply(set).to_dict()
po_keys = df_all["PURCHASE_ORDER"].values

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

X_val_t3 = np.hstack([
    X_val_t0,
    np.array(t3_list, dtype=np.float32)
])

ranker = joblib.load(PROD_MODEL_PATH)
val_scores = ranker.predict(X_val_t3)

print("  LambdaMART scoring complete.", flush=True)

print("[3/6] Reconstructing Top-30 and MMR selections...", flush=True)

query_candidates = {}

offset = 0

for i in range(n_val):
    cands = val_pools_raw[i]
    k = val_groups[i]

    if k == 0 or len(cands) == 0:
        query_candidates[i] = {
            "baseline": [],
            "mmr": [],
        }
        continue

    q_sc = val_scores[offset:offset + k]
    offset += k

    sort_idx = np.argsort(-q_sc)

    top30_idx = sort_idx[:min(30, k)]
    top30_cands = [cands[idx] for idx in top30_idx]
    top30_sc = q_sc[top30_idx]

    proc_cands = []

    for pos, c in enumerate(top30_cands):
        proc_cands.append({
            "score": float(top30_sc[pos]),
            "price": float(c.get("target_unit_price", 0.0)),
            "strat_key": build_stratified_identity(c),
            "po": str(c.get("PURCHASE_ORDER") or ""),
            "tokens": get_tokens(
                c.get("product_text_normalized")
                or c.get("COMMODITY_DESCRIPTION")
                or ""
            ),
            "_raw": c,
            "_original_rank": int(pos + 1),
        })

    baseline = proc_cands[:min(10, len(proc_cands))]

    scores_arr = np.array(
        [c["score"] for c in proc_cands],
        dtype=np.float32
    )

    mmr_idx = run_mmr_selection(
        proc_cands,
        scores_arr,
        lam=0.6,
        k_select=10,
    )

    mmr = [proc_cands[idx] for idx in mmr_idx]

    query_candidates[i] = {
        "baseline": baseline,
        "mmr": mmr,
    }

print("[4/6] Reproducing exact 2-fold Selective-MMR gate...", flush=True)

rng = np.random.RandomState(RANDOM_SEED)
permuted_indices = rng.permutation(n_val)

fold1_indices = permuted_indices[: n_val // 2]
fold2_indices = permuted_indices[n_val // 2 :]

df_signals["cv_fold"] = 0
df_signals.loc[fold1_indices, "cv_fold"] = 1
df_signals.loc[fold2_indices, "cv_fold"] = 2

df_train1 = df_signals.loc[fold1_indices].copy().reset_index(drop=True)
df_test2 = df_signals.loc[fold2_indices].copy().reset_index(drop=True)

best_rule_f1, _ = tune_threshold_on_training_fold(df_train1)

if best_rule_f1["rule_type"] == "sim_only":
    f2_trigger = (
        df_test2["max_pairwise_similarity"].values
        <= best_rule_f1["sim_threshold"]
    )
else:
    f2_trigger = (
        (df_test2["max_pairwise_similarity"].values
         <= best_rule_f1["sim_threshold"])
        &
        (df_test2["unique_po_count"].values
         >= best_rule_f1["po_threshold"])
    )

df_train2 = df_signals.loc[fold2_indices].copy().reset_index(drop=True)
df_test1 = df_signals.loc[fold1_indices].copy().reset_index(drop=True)

best_rule_f2, _ = tune_threshold_on_training_fold(df_train2)

if best_rule_f2["rule_type"] == "sim_only":
    f1_trigger = (
        df_test1["max_pairwise_similarity"].values
        <= best_rule_f2["sim_threshold"]
    )
else:
    f1_trigger = (
        (df_test1["max_pairwise_similarity"].values
         <= best_rule_f2["sim_threshold"])
        &
        (df_test1["unique_po_count"].values
         >= best_rule_f2["po_threshold"])
    )

trigger = np.zeros(n_val, dtype=bool)
trigger[fold2_indices] = f2_trigger
trigger[fold1_indices] = f1_trigger

print(
    f"  Selective MMR triggered for {int(trigger.sum()):,} / {n_val:,} queries.",
    flush=True
)

print("[5/6] Writing actual Selective-MMR Top-10 candidates...", flush=True)

rows = []

for q_idx in range(n_val):
    selected_type = "MMR" if trigger[q_idx] else "BASELINE"

    selected = (
        query_candidates[q_idx]["mmr"]
        if trigger[q_idx]
        else query_candidates[q_idx]["baseline"]
    )

    actual_price = float(y_true_val[q_idx])

    for final_rank, cand in enumerate(selected, start=1):
        raw = cand["_raw"]

        candidate_price = float(cand["price"])
        ape = (
            abs(candidate_price - actual_price)
            / max(actual_price, 1e-4)
            * 100.0
        )

        rows.append({
            "query_index": q_idx,
            "selection_source": selected_type,
            "actual_unit_price": actual_price,
            "candidate_rank": final_rank,
            "original_top30_rank": cand["_original_rank"],
            "candidate_unit_price": candidate_price,
            "candidate_score": float(cand["score"]),
            "candidate_description": str(
                raw.get("product_text_normalized")
                or raw.get("COMMODITY_DESCRIPTION")
                or ""
            ),
            "purchase_order": str(
                raw.get("PURCHASE_ORDER") or ""
            ),
            "commodity_code": str(
                raw.get("commodity_code") or
                raw.get("COMMODITY") or
                ""
            ),
            "unit_of_measure": str(
                raw.get("uom_standardized") or
                raw.get("UNIT_OF_MEASURE") or
                ""
            ),
            "product_identity_key": str(
                raw.get("product_identity_key") or ""
            ),
            "absolute_percentage_error": ape,
            "is_within_10pct": bool(ape <= 10.0),
        })

df_out = pd.DataFrame(rows)

df_out.to_csv(OUTPUT_PATH, index=False)

print("[6/6] Validation...", flush=True)

query_counts = df_out.groupby("query_index").size()

print(f"  Output rows: {len(df_out):,}", flush=True)
print(f"  Unique queries: {df_out['query_index'].nunique():,}", flush=True)
print(f"  Queries with 10 candidates: {int((query_counts == 10).sum()):,}", flush=True)
print(f"  Output: {OUTPUT_PATH}", flush=True)

if df_out["query_index"].nunique() != n_val:
    raise RuntimeError(
        f"Expected {n_val} queries, got "
        f"{df_out['query_index'].nunique()}"
    )

if not bool((query_counts == 10).all()):
    bad = query_counts[query_counts != 10]
    raise RuntimeError(
        f"Some queries do not have exactly 10 candidates:\n{bad.head(20)}"
    )

print("\nSUCCESS: Selective-MMR Top-10 candidate file created.", flush=True)
