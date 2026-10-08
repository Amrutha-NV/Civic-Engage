"""Experiment 1: pure semantic cosine reranking of a frozen Top-10 pool.

Candidate selection depends exclusively on target_description and
candidate_description. Candidate price, frozen rank/score, and evaluation-only
columns are excluded from embedding and selection logic.
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

MODULE_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = MODULE_ROOT.parent
DEFAULT_DATASET = RESEARCH_ROOT / "procurement_benchmark_topk" / "frozen_evaluation_dataset.parquet"
DEFAULT_RESULTS = MODULE_ROOT / "results"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

sys.path.insert(0, str(MODULE_ROOT))
from evaluation.metrics import absolute_percentage_error, selection_metrics  # noqa: E402
from src.baseline import validate_frozen_dataset  # noqa: E402


def normalise_description(values: pd.Series) -> pd.Series:
    """Preserve descriptions while making nulls safe for the embedding model."""
    return values.fillna("").astype(str).str.strip()


def candidate_semantic_scores(frame: pd.DataFrame, model_name: str, batch_size: int) -> tuple[pd.DataFrame, int]:
    """Compute candidate-level cosine scores using only the two descriptions."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Experiment 1 requires sentence-transformers. Install the module requirements first: "
            ".\\.venv\\Scripts\\python.exe -m pip install -r "
            "ML\\research\\llm_semantic_reranker\\requirements.txt"
        ) from error
    candidates = frame[["query_index", "candidate_rank", "target_description", "candidate_description"]].copy()
    candidates["target_text"] = normalise_description(candidates["target_description"])
    candidates["candidate_text"] = normalise_description(candidates["candidate_description"])

    # A shared text cache avoids embedding duplicate descriptions across queries.
    unique_texts = pd.concat([candidates["target_text"], candidates["candidate_text"]], ignore_index=True).unique().tolist()
    model = SentenceTransformer(model_name, device="cpu")
    embeddings = model.encode(
        unique_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embedding_by_text = dict(zip(unique_texts, embeddings, strict=True))
    target_embeddings = np.vstack(candidates["target_text"].map(embedding_by_text).to_numpy())
    candidate_embeddings = np.vstack(candidates["candidate_text"].map(embedding_by_text).to_numpy())
    candidates["semantic_cosine_similarity"] = np.einsum("ij,ij->i", target_embeddings, candidate_embeddings)
    return candidates[["query_index", "candidate_rank", "semantic_cosine_similarity"]], len(unique_texts)


def semantic_selection(frame: pd.DataFrame, model_name: str, batch_size: int) -> tuple[pd.DataFrame, int]:
    """Return one candidate per query using only description embedding cosine scores."""
    candidates, unique_descriptions = candidate_semantic_scores(frame, model_name, batch_size)

    # Stable ordering makes ties deterministic without treating rank as a score.
    selected = (
        candidates.sort_values(
            ["query_index", "semantic_cosine_similarity", "candidate_rank"],
            ascending=[True, False, True],
            kind="stable",
        )
        .drop_duplicates("query_index", keep="first")
        [["query_index", "candidate_rank", "semantic_cosine_similarity"]]
        .rename(columns={"candidate_rank": "selected_candidate_rank"})
    )
    return selected, unique_descriptions


def build_ledger(frame: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """Bind the selected rank to its exact historical price, then evaluate it."""
    rank_one = frame.loc[frame["candidate_rank"].eq(1), ["query_index", "candidate_unit_price", "actual_unit_price"]].copy()
    rank_one = rank_one.rename(columns={"candidate_unit_price": "baseline_unit_price"})
    resolved = frame[["query_index", "candidate_rank", "candidate_unit_price"]].merge(
        selected,
        left_on=["query_index", "candidate_rank"],
        right_on=["query_index", "selected_candidate_rank"],
        how="inner",
        validate="one_to_one",
    )
    resolved = resolved[["query_index", "selected_candidate_rank", "candidate_unit_price", "semantic_cosine_similarity"]]
    resolved = resolved.rename(columns={"candidate_unit_price": "selected_unit_price"})
    ledger = rank_one.merge(resolved, on="query_index", how="inner", validate="one_to_one").sort_values("query_index")
    if len(ledger) != frame["query_index"].nunique():
        raise ValueError("Cosine reranker did not select exactly one candidate for every query")

    ledger["baseline_rank"] = 1
    ledger["gate_status"] = "NOT_APPLIED"
    ledger["llm_decision"] = "HISTORICAL_ACCEPTED"
    ledger["baseline_absolute_percentage_error"] = absolute_percentage_error(ledger["baseline_unit_price"], ledger["actual_unit_price"])
    ledger["llm_absolute_percentage_error"] = absolute_percentage_error(ledger["selected_unit_price"], ledger["actual_unit_price"])
    ledger["is_llm_within_10pct"] = ledger["llm_absolute_percentage_error"] <= 10.0
    return ledger[
        [
            "query_index", "baseline_rank", "baseline_unit_price", "gate_status", "llm_decision",
            "selected_candidate_rank", "selected_unit_price", "semantic_cosine_similarity", "actual_unit_price",
            "baseline_absolute_percentage_error", "llm_absolute_percentage_error", "is_llm_within_10pct",
        ]
    ]


def run(dataset_path: Path, results_dir: Path, model_name: str, batch_size: int) -> dict[str, Any]:
    start = time.perf_counter()
    frame = pd.read_parquet(dataset_path)
    validate_frozen_dataset(frame)
    selected, unique_descriptions = semantic_selection(frame, model_name, batch_size)
    ledger = build_ledger(frame, selected)
    rank_distribution = {str(rank): int((ledger["selected_candidate_rank"] == rank).sum()) for rank in range(1, 11)}
    summary = {
        "experiment": "embedding_cosine_reranker",
        "dataset_path": str(dataset_path),
        "dataset_rows": int(len(frame)),
        "query_count": int(len(ledger)),
        "candidates_per_query": 10,
        "embedding_model": model_name,
        "device": "cpu",
        "batch_size": batch_size,
        "unique_descriptions_encoded": unique_descriptions,
        "selection_inputs": ["target_description", "candidate_description"],
        "excluded_from_selection": [
            "candidate_unit_price", "candidate_score", "candidate_rank",
            "actual_unit_price", "absolute_percentage_error", "is_within_10pct",
        ],
        "tie_break_rule": "lowest candidate_rank after equal cosine similarity",
        "baseline": selection_metrics(ledger["baseline_unit_price"], ledger["actual_unit_price"]),
        "cosine_reranker": selection_metrics(ledger["selected_unit_price"], ledger["actual_unit_price"]),
        "selected_rank_distribution": rank_distribution,
        "execution_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": round(time.perf_counter() - start, 3),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    ledger.to_parquet(results_dir / "embedding_cosine_ledger.parquet", index=False)
    ledger.to_csv(results_dir / "embedding_cosine_ledger.csv", index=False)
    (results_dir / "embedding_cosine_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pure semantic cosine reranking over frozen Top-10 candidates.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.dataset, args.results_dir, args.model, args.batch_size), indent=2))
