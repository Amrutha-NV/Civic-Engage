"""
Experiment 5: Multi-Aspect Historical Price Reranker.

Reranks the already-frozen Top-10 candidate pool using four evidence groups:

1. Semantic similarity
2. Structured attribute/specification evidence
3. Historical price-consensus evidence
4. Original retrieval evidence

IMPORTANT:
- The Top-10 pool is NEVER modified.
- No new candidate is retrieved.
- actual_unit_price, absolute_percentage_error, and is_within_10pct
  are NEVER used during selection.
- candidate_unit_price IS allowed because historical candidate prices
  are available at inference time.
- actual_unit_price is used only AFTER selection for evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# PATHS / DEFAULTS
# ============================================================

MODULE_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = MODULE_ROOT.parent

DEFAULT_DATASET = (
    RESEARCH_ROOT
    / "procurement_benchmark_topk"
    / "frozen_evaluation_dataset.parquet"
)

DEFAULT_RESULTS = MODULE_ROOT / "results"

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


sys.path.insert(0, str(MODULE_ROOT))

from evaluation.metrics import (  # noqa: E402
    absolute_percentage_error,
    selection_metrics,
)

from src.attribute_scorer import score_attributes  # noqa: E402
from src.baseline import validate_frozen_dataset  # noqa: E402
from src.embedding_reranker import candidate_semantic_scores  # noqa: E402


# ============================================================
# NORMALISATION
# ============================================================

def min_max_by_query(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    """
    Min-max normalise a candidate-level feature independently
    inside each frozen Top-10 pool.
    """

    grouped = frame.groupby(
        "query_index",
        sort=False,
    )[column]

    minimum = grouped.transform("min")
    maximum = grouped.transform("max")

    span = maximum - minimum

    return (
        frame[column] - minimum
    ).div(
        span.where(span.ne(0), 1.0)
    )


# ============================================================
# PRICE CONSENSUS
# ============================================================

def calculate_price_consensus(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build historical price-pattern features using ONLY the
    ten candidate prices belonging to each query.

    No actual target price is used.

    Main idea:
    A historical candidate receives stronger evidence when
    several other retrieved historical candidates have similar
    prices.

    Features:
        price_support_5
        price_support_10
        price_support_20
        price_distance_from_median
        price_consensus_score
    """

    records: list[dict[str, Any]] = []

    for query_index, group in frame.groupby(
        "query_index",
        sort=False,
    ):

        group = group.sort_values("candidate_rank")

        prices = (
            group["candidate_unit_price"]
            .astype(float)
            .to_numpy()
        )

        # Historical pool median.
        median_price = float(np.median(prices))

        for row_position, (_, row) in enumerate(
            group.iterrows()
        ):

            price = float(
                row["candidate_unit_price"]
            )

            # ------------------------------------------------
            # Relative difference between this candidate price
            # and every other candidate price.
            # ------------------------------------------------

            denominator = max(
                abs(price),
                1e-9,
            )

            relative_difference = (
                np.abs(prices - price)
                / denominator
            )

            # Number of Top-10 historical prices supporting
            # this price neighbourhood.
            support_5 = int(
                np.sum(relative_difference <= 0.05)
            )

            support_10 = int(
                np.sum(relative_difference <= 0.10)
            )

            support_20 = int(
                np.sum(relative_difference <= 0.20)
            )

            # ------------------------------------------------
            # Distance from historical pool median.
            # ------------------------------------------------

            if median_price > 0 and price > 0:

                price_distance_from_median = abs(
                    np.log(price / median_price)
                )

            else:

                price_distance_from_median = 0.0

            records.append(
                {
                    "query_index":
                        query_index,

                    "candidate_rank":
                        int(row["candidate_rank"]),

                    "candidate_unit_price":
                        price,

                    "pool_median_price":
                        median_price,

                    "price_support_5":
                        support_5,

                    "price_support_10":
                        support_10,

                    "price_support_20":
                        support_20,

                    "price_distance_from_median":
                        price_distance_from_median,
                }
            )

    scored = pd.DataFrame(records)

    # --------------------------------------------------------
    # Normalize support independently inside each Top-10.
    # --------------------------------------------------------

    scored["price_support_5_normalized"] = (
        scored["price_support_5"] / 10.0
    )

    scored["price_support_10_normalized"] = (
        scored["price_support_10"] / 10.0
    )

    scored["price_support_20_normalized"] = (
        scored["price_support_20"] / 10.0
    )

    # Smaller median distance is better.
    scored["median_distance_normalized"] = (
        min_max_by_query(
            scored,
            "price_distance_from_median",
        )
    )

    scored["median_proximity_score"] = (
        1.0
        - scored["median_distance_normalized"]
    )

    # --------------------------------------------------------
    # PRICE CONSENSUS SCORE
    #
    # Strongest emphasis is placed on ±10%, because our
    # benchmark itself evaluates price accuracy at ±10%.
    #
    # IMPORTANT:
    # This does NOT compare against actual_unit_price.
    # It compares historical candidates against each other.
    # --------------------------------------------------------

    scored["price_consensus_score"] = (
        0.15
        * scored["price_support_5_normalized"]

        + 0.45
        * scored["price_support_10_normalized"]

        + 0.20
        * scored["price_support_20_normalized"]

        + 0.20
        * scored["median_proximity_score"]
    )

    return scored


# ============================================================
# ORIGINAL RETRIEVAL EVIDENCE
# ============================================================

def calculate_retrieval_evidence(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Preserve evidence from Person 1's original retrieval model.

    We do NOT alter the Top-10.

    candidate_score is normalized inside each query.
    candidate_rank becomes a simple rank prior:
        rank 1  -> 1.0
        rank 10 -> 0.1
    """

    scored = frame[
        [
            "query_index",
            "candidate_rank",
            "candidate_score",
        ]
    ].copy()

    scored["retrieval_score_normalized"] = (
        min_max_by_query(
            scored,
            "candidate_score",
        )
    )

    scored["rank_prior"] = (
        1.0
        / scored["candidate_rank"].astype(float)
    )

    return scored


# ============================================================
# MULTI-ASPECT SCORING
# ============================================================

def multi_aspect_scores(
    frame: pd.DataFrame,
    model_name: str,
    batch_size: int,
    semantic_weight: float,
    attribute_weight: float,
    price_weight: float,
    retrieval_weight: float,
) -> tuple[pd.DataFrame, int]:

    """
    Score every supplied historical candidate from four angles.

    No evaluation-only field participates in scoring.
    """

    total_weight = (
        semantic_weight
        + attribute_weight
        + price_weight
        + retrieval_weight
    )

    if not np.isclose(total_weight, 1.0):
        raise ValueError(
            "semantic_weight + attribute_weight + "
            "price_weight + retrieval_weight must equal 1.0"
        )

    # ========================================================
    # 1. SEMANTIC EVIDENCE
    # ========================================================

    print(
        "[1/4] Computing semantic similarity..."
    )

    semantic_scores, unique_descriptions = (
        candidate_semantic_scores(
            frame,
            model_name,
            batch_size,
        )
    )

    semantic_scores[
        "semantic_score_normalized"
    ] = min_max_by_query(
        semantic_scores,
        "semantic_cosine_similarity",
    )

    # ========================================================
    # 2. ATTRIBUTE / SPECIFICATION EVIDENCE
    # ========================================================

    print(
        "[2/4] Computing structured attribute evidence..."
    )

    attributes = score_attributes(frame)[
        [
            "query_index",
            "candidate_rank",
            "attribute_score",
            "brand_match",
            "model_match",
            "commodity_code_match",
        ]
    ]

    # attribute_score is already in [0,1] because the existing
    # attribute scorer uses weights 0.25 + 0.65 + 0.10.

    # ========================================================
    # 3. HISTORICAL PRICE EVIDENCE
    # ========================================================

    print(
        "[3/4] Computing historical price consensus..."
    )

    price_scores = calculate_price_consensus(
        frame
    )

    # ========================================================
    # 4. PERSON-1 RETRIEVAL EVIDENCE
    # ========================================================

    print(
        "[4/4] Computing retrieval evidence..."
    )

    retrieval_scores = (
        calculate_retrieval_evidence(frame)
    )

    # ========================================================
    # MERGE ALL FOUR ASPECTS
    # ========================================================

    scored = semantic_scores.merge(
        attributes,
        on=[
            "query_index",
            "candidate_rank",
        ],
        how="inner",
        validate="one_to_one",
    )

    scored = scored.merge(
        price_scores,
        on=[
            "query_index",
            "candidate_rank",
        ],
        how="inner",
        validate="one_to_one",
    )

    scored = scored.merge(
        retrieval_scores,
        on=[
            "query_index",
            "candidate_rank",
        ],
        how="inner",
        validate="one_to_one",
    )

    # ========================================================
    # RETRIEVAL EVIDENCE COMBINATION
    # ========================================================

    scored["retrieval_evidence_score"] = (
        0.80
        * scored["retrieval_score_normalized"]

        + 0.20
        * scored["rank_prior"]
    )

    # ========================================================
    # FINAL MULTI-ASPECT SCORE
    # ========================================================

    scored["multi_aspect_score"] = (
        semantic_weight
        * scored["semantic_score_normalized"]

        + attribute_weight
        * scored["attribute_score"]

        + price_weight
        * scored["price_consensus_score"]

        + retrieval_weight
        * scored["retrieval_evidence_score"]
    )

    return scored, unique_descriptions


# ============================================================
# SELECT BEST EXISTING CANDIDATE
# ============================================================

def select_candidates(
    scored: pd.DataFrame,
) -> pd.DataFrame:

    """
    Select exactly ONE of the supplied Top-10 candidates.

    Rank is used only as the final deterministic tie-break.
    """

    selected = (
        scored
        .sort_values(
            [
                "query_index",
                "multi_aspect_score",
                "candidate_rank",
            ],
            ascending=[
                True,
                False,
                True,
            ],
            kind="stable",
        )
        .drop_duplicates(
            "query_index",
            keep="first",
        )
    )

    selected = selected[
        [
            "query_index",
            "candidate_rank",

            "multi_aspect_score",

            "semantic_cosine_similarity",
            "semantic_score_normalized",

            "attribute_score",
            "brand_match",
            "model_match",
            "commodity_code_match",

            "candidate_unit_price",
            "pool_median_price",

            "price_support_5",
            "price_support_10",
            "price_support_20",

            "price_distance_from_median",
            "median_proximity_score",
            "price_consensus_score",

            "candidate_score",
            "retrieval_score_normalized",
            "rank_prior",
            "retrieval_evidence_score",
        ]
    ].rename(
        columns={
            "candidate_rank":
                "selected_candidate_rank",

            "candidate_unit_price":
                "selected_unit_price",
        }
    )

    if len(selected) != scored[
        "query_index"
    ].nunique():

        raise ValueError(
            "Multi-aspect reranker did not select "
            "exactly one candidate per query."
        )

    if not selected[
        "selected_candidate_rank"
    ].isin(range(1, 11)).all():

        raise ValueError(
            "Selected candidate outside frozen Top-10."
        )

    return selected


# ============================================================
# LEDGER
# ============================================================

def build_ledger(
    frame: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:

    """
    After candidate selection has finished, attach:

        baseline historical price
        selected historical price
        actual benchmark price

    Only here is actual_unit_price used.
    """

    rank_one = frame.loc[
        frame["candidate_rank"].eq(1),
        [
            "query_index",
            "candidate_unit_price",
            "actual_unit_price",
        ],
    ].copy()

    rank_one = rank_one.rename(
        columns={
            "candidate_unit_price":
                "baseline_unit_price"
        }
    )

    ledger = rank_one.merge(
        selected,
        on="query_index",
        how="inner",
        validate="one_to_one",
    )

    ledger = ledger.sort_values(
        "query_index"
    )

    if len(ledger) != frame[
        "query_index"
    ].nunique():

        raise ValueError(
            "Ledger does not contain one row "
            "for every frozen query."
        )

    ledger["baseline_rank"] = 1

    # --------------------------------------------------------
    # Evaluation happens ONLY after selection.
    # --------------------------------------------------------

    ledger[
        "baseline_absolute_percentage_error"
    ] = absolute_percentage_error(
        ledger["baseline_unit_price"],
        ledger["actual_unit_price"],
    )

    ledger[
        "multi_aspect_absolute_percentage_error"
    ] = absolute_percentage_error(
        ledger["selected_unit_price"],
        ledger["actual_unit_price"],
    )

    ledger[
        "is_multi_aspect_within_5pct"
    ] = (
        ledger[
            "multi_aspect_absolute_percentage_error"
        ]
        <= 5.0
    )

    ledger[
        "is_multi_aspect_within_10pct"
    ] = (
        ledger[
            "multi_aspect_absolute_percentage_error"
        ]
        <= 10.0
    )

    ledger[
        "is_multi_aspect_within_20pct"
    ] = (
        ledger[
            "multi_aspect_absolute_percentage_error"
        ]
        <= 20.0
    )

    ledger["changed_from_rank_1"] = (
        ledger["selected_candidate_rank"]
        != 1
    )

    return ledger


# ============================================================
# RUN EXPERIMENT
# ============================================================

def run(
    dataset_path: Path,
    results_dir: Path,
    model_name: str,
    batch_size: int,
    semantic_weight: float,
    attribute_weight: float,
    price_weight: float,
    retrieval_weight: float,
) -> dict[str, Any]:

    start = time.perf_counter()

    print("=" * 70)
    print(
        "MULTI-ASPECT HISTORICAL PRICE RERANKER"
    )
    print("=" * 70)

    print(
        f"\nDataset: {dataset_path}"
    )

    # ========================================================
    # LOAD FROZEN DATASET
    # ========================================================

    frame = pd.read_parquet(
        dataset_path
    )

    validate_frozen_dataset(frame)

    print(
        f"Rows: {len(frame)}"
    )

    print(
        "Queries:",
        frame["query_index"].nunique(),
    )

    # ========================================================
    # SCORE
    # ========================================================

    scored, unique_descriptions = (
        multi_aspect_scores(
            frame=frame,
            model_name=model_name,
            batch_size=batch_size,

            semantic_weight=semantic_weight,
            attribute_weight=attribute_weight,
            price_weight=price_weight,
            retrieval_weight=retrieval_weight,
        )
    )

    # ========================================================
    # SELECT
    # ========================================================

    selected = select_candidates(
        scored
    )

    # ========================================================
    # EVALUATE
    # ========================================================

    ledger = build_ledger(
        frame,
        selected,
    )

    baseline_metrics = selection_metrics(
        ledger["baseline_unit_price"],
        ledger["actual_unit_price"],
    )

    multi_aspect_metrics = selection_metrics(
        ledger["selected_unit_price"],
        ledger["actual_unit_price"],
    )

    # ========================================================
    # ADDITIONAL ANALYSIS
    # ========================================================

    rank_distribution = {
        str(rank):
            int(
                (
                    ledger[
                        "selected_candidate_rank"
                    ]
                    == rank
                ).sum()
            )

        for rank in range(1, 11)
    }

    changed_from_rank_1 = int(
        ledger[
            "changed_from_rank_1"
        ].sum()
    )

    # --------------------------------------------------------
    # How often changing Rank-1 helped or hurt.
    # --------------------------------------------------------

    baseline_success = (
        ledger[
            "baseline_absolute_percentage_error"
        ]
        <= 10.0
    )

    multi_success = (
        ledger[
            "multi_aspect_absolute_percentage_error"
        ]
        <= 10.0
    )

    recovered_queries = int(
        (
            (~baseline_success)
            & multi_success
        ).sum()
    )

    damaged_queries = int(
        (
            baseline_success
            & (~multi_success)
        ).sum()
    )

    unchanged_correct = int(
        (
            baseline_success
            & multi_success
        ).sum()
    )

    unchanged_wrong = int(
        (
            (~baseline_success)
            & (~multi_success)
        ).sum()
    )

    net_recovery = (
        recovered_queries
        - damaged_queries
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "experiment":
            "multi_aspect_historical_price_reranker",

        "dataset_path":
            str(dataset_path),

        "dataset_rows":
            int(len(frame)),

        "query_count":
            int(len(ledger)),

        "candidates_per_query":
            10,

        "embedding_model":
            model_name,

        "device":
            "cpu",

        "batch_size":
            batch_size,

        "unique_descriptions_encoded":
            unique_descriptions,

        "weights": {
            "semantic":
                semantic_weight,

            "attribute":
                attribute_weight,

            "historical_price_consensus":
                price_weight,

            "original_retrieval":
                retrieval_weight,
        },

        "selection_inputs": [
            "target_description",
            "candidate_description",

            "target_brand",
            "target_model",
            "target_commodity_code",

            "candidate_unit_price",

            "candidate_score",
            "candidate_rank",

            "Top-10 candidate price distribution",
        ],

        "evaluation_only_fields": [
            "actual_unit_price",
            "absolute_percentage_error",
            "is_within_10pct",
        ],

        "price_features": [
            "price_support_5",
            "price_support_10",
            "price_support_20",
            "price_distance_from_median",
            "median_proximity_score",
            "price_consensus_score",
        ],

        "selection_rule":
            "highest multi_aspect_score",

        "tie_break_rule":
            "lowest candidate_rank only after equal "
            "multi_aspect score",

        "baseline":
            baseline_metrics,

        "multi_aspect_reranker":
            multi_aspect_metrics,

        "selected_rank_distribution":
            rank_distribution,

        "rank_changes_from_baseline":
            changed_from_rank_1,

        "comparison_with_rank_1_at_10pct": {
            "recovered_queries":
                recovered_queries,

            "damaged_queries":
                damaged_queries,

            "unchanged_correct":
                unchanged_correct,

            "unchanged_wrong":
                unchanged_wrong,

            "net_recovery":
                net_recovery,
        },

        "execution_utc":
            datetime.now(UTC).isoformat(),

        "runtime_seconds":
            round(
                time.perf_counter() - start,
                3,
            ),
    }

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Full candidate-level scoring file.
    scored.to_parquet(
        results_dir
        / "multi_aspect_candidate_scores.parquet",

        index=False,
    )

    scored.to_csv(
        results_dir
        / "multi_aspect_candidate_scores.csv",

        index=False,
    )

    # Query-level selected-candidate ledger.
    ledger.to_parquet(
        results_dir
        / "multi_aspect_ledger.parquet",

        index=False,
    )

    ledger.to_csv(
        results_dir
        / "multi_aspect_ledger.csv",

        index=False,
    )

    # Metrics summary.
    metrics_path = (
        results_dir
        / "multi_aspect_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",

        encoding="utf-8",
    )

    # ========================================================
    # PRINT IMPORTANT RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        "\nBaseline Accuracy@10%:",
        baseline_metrics[
            "accuracy_at_10pct"
        ]["percentage"],
    )

    print(
        "Multi-Aspect Accuracy@10%:",
        multi_aspect_metrics[
            "accuracy_at_10pct"
        ]["percentage"],
    )

    improvement = (
        multi_aspect_metrics[
            "accuracy_at_10pct"
        ]["percentage"]

        - baseline_metrics[
            "accuracy_at_10pct"
        ]["percentage"]
    )

    print(
        "Absolute improvement:",
        round(improvement, 6),
        "percentage points",
    )

    print(
        "\nRecovered:",
        recovered_queries,
    )

    print(
        "Damaged:",
        damaged_queries,
    )

    print(
        "Net recovery:",
        net_recovery,
    )

    print(
        "\nFull metrics saved to:",
        metrics_path,
    )

    return summary


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Run multi-aspect historical-price "
            "reranking over the frozen Top-10."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS,
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    # You already established that 16 works safely on
    # your machine.
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    # --------------------------------------------------------
    # Initial experimental weights.
    #
    # These are deliberately explicit so we can later run
    # controlled weight ablations without rewriting code.
    # --------------------------------------------------------

    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=0.40,
    )

    parser.add_argument(
        "--attribute-weight",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--price-weight",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--retrieval-weight",
        type=float,
        default=0.15,
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    summary = run(
        dataset_path=args.dataset,
        results_dir=args.results_dir,
        model_name=args.model,
        batch_size=args.batch_size,

        semantic_weight=args.semantic_weight,
        attribute_weight=args.attribute_weight,
        price_weight=args.price_weight,
        retrieval_weight=args.retrieval_weight,
    )

    print(
        "\nFULL SUMMARY\n"
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )