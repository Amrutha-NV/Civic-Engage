"""Experiment 4: cross-encoder description-pair reranking for frozen Top-10s."""

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
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

sys.path.insert(0, str(MODULE_ROOT))
from evaluation.metrics import absolute_percentage_error, selection_metrics  # noqa: E402
from src.baseline import validate_frozen_dataset  # noqa: E402


def score_pairs(frame: pd.DataFrame, model_name: str, batch_size: int) -> pd.DataFrame:
    """Score each description pair; no price, rank, score, or labels are input."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as error:
        raise RuntimeError(
            "Experiment 4 requires sentence-transformers. Install module requirements before running."
        ) from error
    pairs = frame[["query_index", "candidate_rank", "target_description", "candidate_description"]].copy()
    pairs["target_description"] = pairs["target_description"].fillna("").astype(str).str.strip()
    pairs["candidate_description"] = pairs["candidate_description"].fillna("").astype(str).str.strip()
    model = CrossEncoder(model_name, device="cpu")
    pairs["cross_encoder_score"] = model.predict(
        list(zip(pairs["target_description"], pairs["candidate_description"], strict=True)),
        batch_size=batch_size,
        show_progress_bar=True,
    )
    return pairs[["query_index", "candidate_rank", "cross_encoder_score"]]


def select_candidates(scores: pd.DataFrame) -> pd.DataFrame:
    """Select highest semantic pair score with deterministic rank-only ties."""
    return (
        scores.sort_values(["query_index", "cross_encoder_score", "candidate_rank"], ascending=[True, False, True], kind="stable")
        .drop_duplicates("query_index", keep="first")
        [["query_index", "candidate_rank", "cross_encoder_score"]]
        .rename(columns={"candidate_rank": "selected_candidate_rank"})
    )


def build_ledger(frame: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rank_one = frame.loc[frame["candidate_rank"].eq(1), ["query_index", "candidate_unit_price", "actual_unit_price"]].rename(
        columns={"candidate_unit_price": "baseline_unit_price"}
    )
    resolved = frame[["query_index", "candidate_rank", "candidate_unit_price"]].merge(
        selected, left_on=["query_index", "candidate_rank"], right_on=["query_index", "selected_candidate_rank"],
        how="inner", validate="one_to_one",
    ).rename(columns={"candidate_unit_price": "selected_unit_price"})
    ledger = rank_one.merge(resolved.drop(columns="candidate_rank"), on="query_index", how="inner", validate="one_to_one").sort_values("query_index")
    if len(ledger) != frame["query_index"].nunique():
        raise ValueError("Cross-encoder did not select exactly one candidate for every query")
    ledger["baseline_rank"] = 1
    ledger["gate_status"] = "NOT_APPLIED"
    ledger["llm_decision"] = "HISTORICAL_ACCEPTED"
    ledger["baseline_absolute_percentage_error"] = absolute_percentage_error(ledger["baseline_unit_price"], ledger["actual_unit_price"])
    ledger["llm_absolute_percentage_error"] = absolute_percentage_error(ledger["selected_unit_price"], ledger["actual_unit_price"])
    ledger["is_llm_within_10pct"] = ledger["llm_absolute_percentage_error"] <= 10.0
    return ledger


def run(dataset_path: Path, results_dir: Path, model_name: str, batch_size: int) -> dict[str, Any]:
    start = time.perf_counter()
    frame = pd.read_parquet(dataset_path)
    validate_frozen_dataset(frame)
    selected = select_candidates(score_pairs(frame, model_name, batch_size))
    ledger = build_ledger(frame, selected)
    summary: dict[str, Any] = {
        "experiment": "cross_encoder_reranker",
        "dataset_path": str(dataset_path),
        "query_count": int(len(ledger)),
        "candidates_per_query": 10,
        "cross_encoder_model": model_name,
        "device": "cpu",
        "batch_size": batch_size,
        "selection_inputs": ["target_description", "candidate_description"],
        "excluded_from_selection": ["candidate_unit_price", "candidate_score", "candidate_rank", "actual_unit_price", "absolute_percentage_error", "is_within_10pct"],
        "tie_break_rule": "lowest candidate_rank after equal cross-encoder score",
        "baseline": selection_metrics(ledger["baseline_unit_price"], ledger["actual_unit_price"]),
        "cross_encoder_reranker": selection_metrics(ledger["selected_unit_price"], ledger["actual_unit_price"]),
        "selected_rank_distribution": {str(rank): int((ledger["selected_candidate_rank"] == rank).sum()) for rank in range(1, 11)},
        "execution_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": round(time.perf_counter() - start, 3),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    ledger.to_parquet(results_dir / "cross_encoder_ledger.parquet", index=False)
    ledger.to_csv(results_dir / "cross_encoder_ledger.csv", index=False)
    (results_dir / "cross_encoder_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cross-Encoder reranking over frozen Top-10 candidates.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.dataset, args.results_dir, args.model, args.batch_size), indent=2))
