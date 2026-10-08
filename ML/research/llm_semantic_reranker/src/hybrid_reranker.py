"""Experiment 3: combine pure semantic cosine and explicit attribute evidence."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

MODULE_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = MODULE_ROOT.parent
DEFAULT_DATASET = RESEARCH_ROOT / "procurement_benchmark_topk" / "frozen_evaluation_dataset.parquet"
DEFAULT_RESULTS = MODULE_ROOT / "results"

sys.path.insert(0, str(MODULE_ROOT))
from evaluation.metrics import absolute_percentage_error, selection_metrics  # noqa: E402
from src.attribute_scorer import score_attributes  # noqa: E402
from src.baseline import validate_frozen_dataset  # noqa: E402
from src.embedding_reranker import candidate_semantic_scores  # noqa: E402


def min_max_by_query(scores: pd.DataFrame) -> pd.Series:
    """Normalise cosine scores inside each fixed Top-10 pool before blending."""
    grouped = scores.groupby("query_index", sort=False)["semantic_cosine_similarity"]
    minimum = grouped.transform("min")
    span = grouped.transform("max") - minimum
    # A uniform score carries no within-pool preference and is assigned zero.
    return (scores["semantic_cosine_similarity"] - minimum).div(span.where(span.ne(0), 1.0))


def hybrid_selection(frame: pd.DataFrame, model_name: str, batch_size: int, semantic_weight: float) -> tuple[pd.DataFrame, int]:
    """Select candidates using a fixed blend of semantic and attribute signals."""
    if not 0.0 <= semantic_weight <= 1.0:
        raise ValueError("semantic_weight must be between 0 and 1")
    semantic_scores, unique_descriptions = candidate_semantic_scores(frame, model_name, batch_size)
    attributes = score_attributes(frame)[
        ["query_index", "candidate_rank", "attribute_score", "brand_match", "model_match", "commodity_code_match"]
    ]
    scored = semantic_scores.merge(attributes, on=["query_index", "candidate_rank"], how="inner", validate="one_to_one")
    scored["semantic_score_normalized"] = min_max_by_query(scored)
    scored["hybrid_score"] = semantic_weight * scored["semantic_score_normalized"] + (1.0 - semantic_weight) * scored["attribute_score"]
    selected = (
        scored.sort_values(["query_index", "hybrid_score", "candidate_rank"], ascending=[True, False, True], kind="stable")
        .drop_duplicates("query_index", keep="first")
        [["query_index", "candidate_rank", "hybrid_score", "semantic_score_normalized", "attribute_score", "brand_match", "model_match", "commodity_code_match"]]
        .rename(columns={"candidate_rank": "selected_candidate_rank"})
    )
    # Defensive invariant: blending must still select exactly one supplied rank.
    if not selected["selected_candidate_rank"].isin(range(1, 11)).all() or len(selected) != frame["query_index"].nunique():
        raise ValueError("Hybrid reranker selection invariant failed")
    return selected, unique_descriptions


def build_ledger(frame: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rank_one = frame.loc[frame["candidate_rank"].eq(1), ["query_index", "candidate_unit_price", "actual_unit_price"]].rename(
        columns={"candidate_unit_price": "baseline_unit_price"}
    )
    resolved = frame[["query_index", "candidate_rank", "candidate_unit_price"]].merge(
        selected, left_on=["query_index", "candidate_rank"], right_on=["query_index", "selected_candidate_rank"],
        how="inner", validate="one_to_one",
    ).rename(columns={"candidate_unit_price": "selected_unit_price"})
    ledger = rank_one.merge(resolved.drop(columns="candidate_rank"), on="query_index", how="inner", validate="one_to_one").sort_values("query_index")
    ledger["baseline_rank"] = 1
    ledger["gate_status"] = "NOT_APPLIED"
    ledger["llm_decision"] = "HISTORICAL_ACCEPTED"
    ledger["baseline_absolute_percentage_error"] = absolute_percentage_error(ledger["baseline_unit_price"], ledger["actual_unit_price"])
    ledger["llm_absolute_percentage_error"] = absolute_percentage_error(ledger["selected_unit_price"], ledger["actual_unit_price"])
    ledger["is_llm_within_10pct"] = ledger["llm_absolute_percentage_error"] <= 10.0
    return ledger


def run(dataset_path: Path, results_dir: Path, model_name: str, batch_size: int, semantic_weight: float) -> dict[str, Any]:
    start = time.perf_counter()
    frame = pd.read_parquet(dataset_path)
    validate_frozen_dataset(frame)
    selected, unique_descriptions = hybrid_selection(frame, model_name, batch_size, semantic_weight)
    ledger = build_ledger(frame, selected)
    summary = {
        "experiment": "semantic_structured_hybrid_reranker",
        "dataset_path": str(dataset_path),
        "query_count": int(len(ledger)),
        "embedding_model": model_name,
        "device": "cpu",
        "unique_descriptions_encoded": unique_descriptions,
        "semantic_weight": semantic_weight,
        "attribute_weight": 1.0 - semantic_weight,
        "selection_inputs": ["target_description", "candidate_description", "target_brand", "target_model", "target_commodity_code"],
        "excluded_from_selection": ["candidate_unit_price", "candidate_score", "actual_unit_price", "absolute_percentage_error", "is_within_10pct"],
        "normalisation": "cosine score min-max normalised within each query Top-10 before blending",
        "tie_break_rule": "lowest candidate_rank after equal hybrid score",
        "baseline": selection_metrics(ledger["baseline_unit_price"], ledger["actual_unit_price"]),
        "hybrid_reranker": selection_metrics(ledger["selected_unit_price"], ledger["actual_unit_price"]),
        "selected_rank_distribution": {str(rank): int((ledger["selected_candidate_rank"] == rank).sum()) for rank in range(1, 11)},
        "execution_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": round(time.perf_counter() - start, 3),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    ledger.to_parquet(results_dir / "hybrid_ledger.parquet", index=False)
    ledger.to_csv(results_dir / "hybrid_ledger.csv", index=False)
    (results_dir / "hybrid_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run semantic plus structured hybrid reranking.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--semantic-weight", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.dataset, args.results_dir, args.model, args.batch_size, args.semantic_weight), indent=2))
