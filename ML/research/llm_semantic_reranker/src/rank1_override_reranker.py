"""
Experiment 6 - Conservative Rank-1 Override Pipeline (fixed)

Rank-1 is the default prediction. It is overridden only when a lower-ranked
candidate has sufficiently stronger evidence. Deterministic, not trained.

actual_unit_price / absolute_percentage_error / is_within_10pct are
evaluation-only and NEVER used for candidate selection.

FIX: removed the stray sort/drop of "_original_row_id" at the end of
build_features (that column was never created there). The row-order
bookkeeping lives entirely inside calculate_price_features.
"""

from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

DATA_PATH = (
    PROJECT_ROOT / "ML" / "research" / "procurement_benchmark_topk"
    / "frozen_evaluation_dataset.parquet"
)

BASE_OUTPUT_DIR = (
    PROJECT_ROOT / "ML" / "research" / "llm_semantic_reranker"
    / "results" / "rank1_override"
)

# True = first 100 queries only, False = full frozen dataset
DEBUG_MODE = False
DEBUG_N_QUERIES = 100

OUTPUT_DIR = BASE_OUTPUT_DIR / "debug_100" if DEBUG_MODE else BASE_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = OUTPUT_DIR / "rank1_override_features_checkpoint.parquet"
CANDIDATE_CSV = OUTPUT_DIR / "rank1_override_candidate_scores.csv"
CANDIDATE_PARQUET = OUTPUT_DIR / "rank1_override_candidate_scores.parquet"
LEDGER_CSV = OUTPUT_DIR / "rank1_override_ledger.csv"
LEDGER_PARQUET = OUTPUT_DIR / "rank1_override_ledger.parquet"
METRICS_PATH = OUTPUT_DIR / "rank1_override_metrics.json"

# ============================================================
# CONFIG
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 16

OVERRIDE_MARGIN = 0.08
MIN_ALTERNATIVE_EVIDENCE = 0.55
STRONG_RANK1_IDENTITY = 0.75
STRONG_RANK1_EXTRA_MARGIN = 0.10


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 6 - Conservative Rank-1 Override Pipeline"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the feature checkpoint and skip embedding generation.",
    )
    return parser.parse_args()


# ============================================================
# TEXT / FIELD HELPERS
# ============================================================

def clean_text(text):
    """Strip explicit currency strings before embedding (prevents price leakage)."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"[$€£]\s?\d[\d,]*(?:\.\d+)?", " ", text)
    text = re.sub(
        r"\b(?:USD|INR|EUR|GBP)\s*\d[\d,]*(?:\.\d+)?\b",
        " ", text, flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def normalize_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def exact_field_match(target_value, candidate_description):
    target = normalize_value(target_value)
    candidate = normalize_value(candidate_description)
    if not target or target == "nan":
        return 0.0
    return float(target in candidate)


def numeric_tokens(text):
    text = normalize_value(text)
    tokens = re.findall(
        r"\b\d+(?:\.\d+)?(?:gb|tb|mb|inch|in|mm|cm|w|v|hz)?\b",
        text, flags=re.IGNORECASE,
    )
    alnum = re.findall(r"\b[a-z]+\d+[a-z0-9-]*\b", text, flags=re.IGNORECASE)
    return set(tokens + alnum)


def numeric_overlap(target_text, candidate_text):
    t = numeric_tokens(target_text)
    c = numeric_tokens(candidate_text)
    if not t:
        return 0.0
    return len(t & c) / len(t)


def normalize_query_scores(series):
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-9:
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


# ============================================================
# VALIDATION
# ============================================================

def validate_dataset(df):
    required = [
        "query_index", "target_description", "target_quantity",
        "target_unit_of_measure", "target_procurement_date", "target_city",
        "target_state", "target_brand", "target_model",
        "target_commodity_code", "target_commodity_family",
        "candidate_rank", "candidate_description", "candidate_unit_price",
        "candidate_score", "actual_unit_price",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    counts = df.groupby("query_index").size()
    if not (counts == 10).all():
        raise ValueError(
            f"Expected exactly 10 candidates per query. "
            f"Bad queries: {counts[counts != 10].head().to_dict()}"
        )

    expected = list(range(1, 11))
    ok = df.groupby("query_index")["candidate_rank"].apply(
        lambda x: sorted(x.tolist()) == expected
    )
    if not ok.all():
        raise ValueError(
            f"Some queries lack ranks 1-10. Examples: {ok[~ok].head().index.tolist()}"
        )

    print("[OK] Dataset validation passed.")
    print(f"[INFO] Rows: {len(df):,}")
    print(f"[INFO] Queries: {df['query_index'].nunique():,}")


# ============================================================
# PRICE CONSENSUS (vectorized)
# ============================================================

def calculate_price_features(df):
    """
    Historical price-consensus features within each Top-10 pool.
    Returns the frame in the SAME row order it was given (index reset).
    actual_unit_price is NOT used.
    """
    df = df.copy().reset_index(drop=True)
    df["_original_row_id"] = np.arange(len(df))

    df = df.sort_values(["query_index", "candidate_rank"], kind="mergesort")
    df = df.reset_index(drop=True)

    prices = df["candidate_unit_price"].astype(float).to_numpy()
    if len(prices) % 10 != 0:
        raise ValueError("Price features require exactly 10 candidates per query.")

    pm = prices.reshape(-1, 10)
    med = np.median(pm, axis=1)

    df["pool_median_price"] = np.repeat(med, 10)

    dist = np.abs(pm - med[:, None]) / np.maximum(np.abs(med[:, None]), 1e-9)
    df["price_distance_from_median"] = dist.reshape(-1)
    df["median_proximity_score"] = 1.0 / (1.0 + df["price_distance_from_median"])

    cand = pm[:, :, None]
    other = pm[:, None, :]
    rel = np.abs(cand - other) / np.maximum(np.abs(cand), 1e-9)

    df["price_support_5"] = (rel <= 0.05).sum(axis=2).astype(int).reshape(-1)
    df["price_support_10"] = (rel <= 0.10).sum(axis=2).astype(int).reshape(-1)
    df["price_support_20"] = (rel <= 0.20).sum(axis=2).astype(int).reshape(-1)
    df["price_consensus_score"] = df["price_support_10"] / 10.0

    df = df.sort_values("_original_row_id", kind="mergesort").reset_index(drop=True)
    df = df.drop(columns=["_original_row_id"])
    return df


# ============================================================
# BUILD FEATURES
# ============================================================

def build_features(df):
    print("\n[INFO] Loading semantic model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Clean positional index so array assignments and groupby align.
    df = df.copy().reset_index(drop=True)

    target_texts = df["target_description"].fillna("").apply(clean_text).tolist()
    candidate_texts = df["candidate_description"].fillna("").apply(clean_text).tolist()

    print("\n[INFO] Encoding target descriptions...")
    t_emb = model.encode(
        target_texts, batch_size=BATCH_SIZE,
        show_progress_bar=True, normalize_embeddings=True,
    )

    print("\n[INFO] Encoding candidate descriptions...")
    c_emb = model.encode(
        candidate_texts, batch_size=BATCH_SIZE,
        show_progress_bar=True, normalize_embeddings=True,
    )

    print("\n[INFO] Calculating semantic similarity...")
    df["semantic_cosine_similarity"] = np.sum(t_emb * c_emb, axis=1)
    df["semantic_score_normalized"] = df.groupby("query_index")[
        "semantic_cosine_similarity"
    ].transform(normalize_query_scores)

    print("\n[INFO] Calculating identity features...")
    df["brand_match"] = [
        exact_field_match(b, d)
        for b, d in zip(df["target_brand"], df["candidate_description"])
    ]
    df["model_match"] = [
        exact_field_match(m, d)
        for m, d in zip(df["target_model"], df["candidate_description"])
    ]
    df["commodity_code_match"] = [
        exact_field_match(c, d)
        for c, d in zip(df["target_commodity_code"], df["candidate_description"])
    ]

    print("\n[INFO] Calculating numeric/specification overlap...")
    df["numeric_overlap"] = [
        numeric_overlap(t, c)
        for t, c in zip(df["target_description"], df["candidate_description"])
    ]

    df["attribute_score"] = (
        0.40 * df["brand_match"]
        + 0.40 * df["model_match"]
        + 0.20 * df["commodity_code_match"]
    )

    print("\n[INFO] Calculating historical price consensus...")
    df = calculate_price_features(df)

    if "query_index" not in df.columns:
        raise RuntimeError("query_index disappeared after price feature calculation.")

    print("\n[INFO] Calculating retrieval evidence...")
    df["retrieval_score_normalized"] = df.groupby("query_index")[
        "candidate_score"
    ].transform(normalize_query_scores)
    df["rank_prior"] = 1.0 / df["candidate_rank"]

    df["product_identity_score"] = (
        0.50 * df["semantic_score_normalized"]
        + 0.20 * df["brand_match"]
        + 0.20 * df["model_match"]
        + 0.10 * df["numeric_overlap"]
    )

    df["historical_price_score"] = (
        0.50 * df["price_consensus_score"]
        + 0.50 * df["median_proximity_score"]
    )

    df["alternative_evidence_score"] = (
        0.55 * df["product_identity_score"]
        + 0.30 * df["historical_price_score"]
        + 0.15 * df["retrieval_score_normalized"]
    )

    # NOTE: no _original_row_id sort/drop here. That was the bug.
    return df


# ============================================================
# OVERRIDE DECISION
# ============================================================

def decide_rank1_override(group):
    group = group.sort_values("candidate_rank").copy()
    rank1 = group.iloc[0]
    alts = group.iloc[1:]

    selected_rank = int(rank1["candidate_rank"])
    if len(alts) == 0:
        return selected_rank, False, None, None

    best = alts.sort_values("alternative_evidence_score", ascending=False).iloc[0]
    best_rank = int(best["candidate_rank"])
    best_score = float(best["alternative_evidence_score"])
    rank1_score = float(rank1["alternative_evidence_score"])
    margin = best_score - rank1_score

    identity_gain = best["product_identity_score"] - rank1["product_identity_score"]
    price_gain = best["historical_price_score"] - rank1["historical_price_score"]
    semantic_gain = best["semantic_score_normalized"] - rank1["semantic_score_normalized"]

    alternative_is_strong = best_score >= MIN_ALTERNATIVE_EVIDENCE

    required_margin = OVERRIDE_MARGIN
    if rank1["product_identity_score"] >= STRONG_RANK1_IDENTITY:
        required_margin = OVERRIDE_MARGIN + STRONG_RANK1_EXTRA_MARGIN
    meaningful_advantage = margin >= required_margin

    identity_or_price_advantage = (
        identity_gain >= 0.08 or price_gain >= 0.12 or semantic_gain >= 0.08
    )

    override = bool(
        alternative_is_strong and meaningful_advantage and identity_or_price_advantage
    )
    if override:
        selected_rank = best_rank

    return selected_rank, override, best_rank, best_score


# ============================================================
# LEDGER
# ============================================================

def _ape(pred, actual):
    return abs(pred - actual) / max(abs(actual), 1e-9) * 100


def build_ledger(df):
    rows = []

    for query_index, group in df.groupby("query_index", sort=True):
        group = group.sort_values("candidate_rank")
        rank1 = group.iloc[0]

        selected_rank, override, best_rank, best_score = decide_rank1_override(group)
        selected = group[group["candidate_rank"] == selected_rank].iloc[0]

        # evaluation only
        actual = float(rank1["actual_unit_price"])
        base_price = float(rank1["candidate_unit_price"])
        sel_price = float(selected["candidate_unit_price"])
        base_ape = _ape(base_price, actual)
        sel_ape = _ape(sel_price, actual)

        rows.append({
            "query_index": int(query_index),
            "baseline_rank": int(rank1["candidate_rank"]),
            "baseline_unit_price": base_price,
            "gate_status": "OVERRIDE" if override else "KEEP_RANK_1",
            "override": bool(override),
            "best_alternative_rank": best_rank,
            "best_alternative_score": best_score,
            "selected_candidate_rank": int(selected_rank),
            "selected_unit_price": sel_price,
            "rank1_product_identity_score": float(rank1["product_identity_score"]),
            "selected_product_identity_score": float(selected["product_identity_score"]),
            "rank1_historical_price_score": float(rank1["historical_price_score"]),
            "selected_historical_price_score": float(selected["historical_price_score"]),
            "rank1_semantic_similarity": float(rank1["semantic_cosine_similarity"]),
            "selected_semantic_similarity": float(selected["semantic_cosine_similarity"]),
            "rank1_price_support_10": int(rank1["price_support_10"]),
            "selected_price_support_10": int(selected["price_support_10"]),
            "rank1_brand_match": int(rank1["brand_match"]),
            "selected_brand_match": int(selected["brand_match"]),
            "rank1_model_match": int(rank1["model_match"]),
            "selected_model_match": int(selected["model_match"]),
            "rank1_numeric_overlap": float(rank1["numeric_overlap"]),
            "selected_numeric_overlap": float(selected["numeric_overlap"]),
            "actual_unit_price": actual,
            "baseline_absolute_percentage_error": base_ape,
            "override_absolute_percentage_error": sel_ape,
            "is_baseline_within_5pct": bool(base_ape <= 5),
            "is_baseline_within_10pct": bool(base_ape <= 10),
            "is_baseline_within_20pct": bool(base_ape <= 20),
            "is_override_within_5pct": bool(sel_ape <= 5),
            "is_override_within_10pct": bool(sel_ape <= 10),
            "is_override_within_20pct": bool(sel_ape <= 20),
            "changed_from_rank_1": bool(selected_rank != 1),
        })

    return pd.DataFrame(rows)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(ledger):
    total = len(ledger)

    b5 = int(ledger["is_baseline_within_5pct"].sum())
    s5 = int(ledger["is_override_within_5pct"].sum())
    b10 = int(ledger["is_baseline_within_10pct"].sum())
    s10 = int(ledger["is_override_within_10pct"].sum())
    b20 = int(ledger["is_baseline_within_20pct"].sum())
    s20 = int(ledger["is_override_within_20pct"].sum())

    base_ok = ledger["is_baseline_within_10pct"]
    sel_ok = ledger["is_override_within_10pct"]

    recovered = int((~base_ok & sel_ok).sum())
    damaged = int((base_ok & ~sel_ok).sum())
    both_correct = int((base_ok & sel_ok).sum())
    both_wrong = int((~base_ok & ~sel_ok).sum())

    actual = ledger["actual_unit_price"]
    base_mae = float(np.mean(np.abs(ledger["baseline_unit_price"] - actual)))
    sel_mae = float(np.mean(np.abs(ledger["selected_unit_price"] - actual)))

    base_ape = ledger["baseline_absolute_percentage_error"]
    sel_ape = ledger["override_absolute_percentage_error"]

    override_count = int(ledger["changed_from_rank_1"].sum())

    return {
        "experiment": "Experiment 6 - Conservative Rank-1 Override",
        "dataset_queries": int(total),
        "baseline_accuracy_at_5pct": b5 / total * 100,
        "override_accuracy_at_5pct": s5 / total * 100,
        "baseline_accuracy_at_10pct": b10 / total * 100,
        "override_accuracy_at_10pct": s10 / total * 100,
        "baseline_accuracy_at_20pct": b20 / total * 100,
        "override_accuracy_at_20pct": s20 / total * 100,
        "accuracy_change_at_10pct_pp": (s10 - b10) / total * 100,
        "recovered": recovered,
        "damaged": damaged,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "net_recovery": recovered - damaged,
        "override_count": override_count,
        "override_rate_pct": override_count / total * 100,
        "rank1_preservation_rate_pct": (total - override_count) / total * 100,
        "baseline_mae": base_mae,
        "override_mae": sel_mae,
        "baseline_mdape": float(np.median(base_ape)),
        "override_mdape": float(np.median(sel_ape)),
        "baseline_mean_ape": float(np.mean(base_ape)),
        "override_mean_ape": float(np.mean(sel_ape)),
        "selected_rank_distribution": {
            str(int(r)): int(c)
            for r, c in ledger["selected_candidate_rank"].value_counts().sort_index().items()
        },
        "decision_breakdown": {
            "KEEP_RANK_1": int((~ledger["changed_from_rank_1"]).sum()),
            "OVERRIDE": override_count,
        },
    }


def save_metrics(metrics):
    metrics["configuration"] = {
        "debug_mode": DEBUG_MODE,
        "debug_n_queries": DEBUG_N_QUERIES,
        "embedding_model": EMBEDDING_MODEL,
        "batch_size": BATCH_SIZE,
        "override_margin": OVERRIDE_MARGIN,
        "min_alternative_evidence": MIN_ALTERNATIVE_EVIDENCE,
        "strong_rank1_identity": STRONG_RANK1_IDENTITY,
        "strong_rank1_extra_margin": STRONG_RANK1_EXTRA_MARGIN,
        "candidate_selection_uses_actual_price": False,
        "candidate_selection_uses_ape": False,
        "candidate_selection_uses_success_label": False,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


# ============================================================
# MAIN
# ============================================================

REQUIRED_FEATURE_COLUMNS = [
    "query_index", "candidate_rank", "candidate_unit_price", "candidate_score",
    "semantic_cosine_similarity", "semantic_score_normalized",
    "brand_match", "model_match", "commodity_code_match", "numeric_overlap",
    "attribute_score", "pool_median_price", "price_support_5",
    "price_support_10", "price_support_20", "price_distance_from_median",
    "median_proximity_score", "price_consensus_score",
    "retrieval_score_normalized", "rank_prior", "product_identity_score",
    "historical_price_score", "alternative_evidence_score", "actual_unit_price",
]


def print_summary(m):
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nBaseline Accuracy@5%  : {m['baseline_accuracy_at_5pct']:.2f}%")
    print(f"Override Accuracy@5%  : {m['override_accuracy_at_5pct']:.2f}%")
    print(f"\nBaseline Accuracy@10% : {m['baseline_accuracy_at_10pct']:.2f}%")
    print(f"Override Accuracy@10% : {m['override_accuracy_at_10pct']:.2f}%")
    print(f"Change                : {m['accuracy_change_at_10pct_pp']:+.2f} pp")
    print(f"\nBaseline Accuracy@20% : {m['baseline_accuracy_at_20pct']:.2f}%")
    print(f"Override Accuracy@20% : {m['override_accuracy_at_20pct']:.2f}%")
    print(f"\nRecovered             : {m['recovered']}")
    print(f"Damaged               : {m['damaged']}")
    print(f"Net recovery          : {m['net_recovery']:+d}")
    print(f"\nOverrides             : {m['override_count']}")
    print(f"Override rate         : {m['override_rate_pct']:.2f}%")
    print(f"Rank-1 preservation   : {m['rank1_preservation_rate_pct']:.2f}%")
    print(f"\nBaseline MAE          : {m['baseline_mae']:.2f}")
    print(f"Override MAE          : {m['override_mae']:.2f}")
    print(f"\nBaseline MdAPE        : {m['baseline_mdape']:.2f}%")
    print(f"Override MdAPE        : {m['override_mdape']:.2f}%")
    print(f"\nBaseline Mean APE     : {m['baseline_mean_ape']:.2f}%")
    print(f"Override Mean APE     : {m['override_mean_ape']:.2f}%")
    print("\nSelected rank distribution:")
    for rank, count in m["selected_rank_distribution"].items():
        print(f"  Rank {rank}: {count}")
    print("\nDecision breakdown:")
    print(f"  KEEP RANK-1: {m['decision_breakdown']['KEEP_RANK_1']}")
    print(f"  OVERRIDE:    {m['decision_breakdown']['OVERRIDE']}")
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


def main():
    args = parse_args()

    print("=" * 70)
    print("EXPERIMENT 6 - CONSERVATIVE RANK-1 OVERRIDE PIPELINE")
    print("=" * 70)
    print(f"\n[INFO] DEBUG_MODE = {DEBUG_MODE}")
    if DEBUG_MODE:
        print(f"[INFO] Debug queries = {DEBUG_N_QUERIES}")

    if args.resume and CHECKPOINT_PATH.exists():
        print(f"\n[RESUME] Found feature checkpoint:\n        {CHECKPOINT_PATH}")
        candidate_scores = pd.read_parquet(CHECKPOINT_PATH)
        print(f"[OK] Checkpoint loaded. Rows: {len(candidate_scores):,}")
    else:
        print("\n[INFO] Loading frozen dataset...")
        df = pd.read_parquet(DATA_PATH)
        validate_dataset(df)

        if DEBUG_MODE:
            query_ids = df["query_index"].drop_duplicates().iloc[:DEBUG_N_QUERIES]
            df = df[df["query_index"].isin(query_ids)].copy()
            print(f"\n[DEBUG MODE] Using {len(query_ids)} queries.")
            print(f"[DEBUG MODE] Candidate rows: {len(df):,}")
        else:
            print(
                f"\n[FULL MODE] Using complete dataset: "
                f"{df['query_index'].nunique():,} queries."
            )

        print("\n[INFO] Building candidate evidence features...")
        candidate_scores = build_features(df)

        print("\n[INFO] Saving feature checkpoint...")
        candidate_scores.to_parquet(CHECKPOINT_PATH, index=False)
        print(f"[OK] Feature checkpoint saved: {CHECKPOINT_PATH}")

    missing = [c for c in REQUIRED_FEATURE_COLUMNS if c not in candidate_scores.columns]
    if missing:
        raise RuntimeError(f"Checkpoint is missing required columns: {missing}")
    print("\n[OK] Feature checkpoint validation passed.")

    print("\n[INFO] Saving candidate-level results...")
    candidate_scores.to_csv(CANDIDATE_CSV, index=False)
    candidate_scores.to_parquet(CANDIDATE_PARQUET, index=False)
    print(f"[OK] {CANDIDATE_CSV}\n[OK] {CANDIDATE_PARQUET}")

    print("\n[INFO] Running Rank-1 KEEP/OVERRIDE decision...")
    ledger = build_ledger(candidate_scores)
    print(f"[OK] Ledger created. Rows: {len(ledger):,}")

    ledger.to_csv(LEDGER_CSV, index=False)
    ledger.to_parquet(LEDGER_PARQUET, index=False)
    print(f"[OK] {LEDGER_CSV}\n[OK] {LEDGER_PARQUET}")

    print("\n[INFO] Calculating metrics...")
    metrics = calculate_metrics(ledger)
    save_metrics(metrics)
    print(f"[OK] {METRICS_PATH}")

    print_summary(metrics)


if __name__ == "__main__":
    main()