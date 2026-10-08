"""Reproduce Rank-1 and Oracle@10 metrics from the frozen candidate pool.

This module deliberately makes no reranking decision.  It selects the frozen
Rank-1 candidate for every query, then uses evaluation-only columns solely to
score that already-fixed selection.
"""

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

REQUIRED_COLUMNS = {
    "query_index",
    "candidate_rank",
    "candidate_unit_price",
    "actual_unit_price",
    "is_within_10pct",
}


def validate_frozen_dataset(frame: pd.DataFrame) -> None:
    """Verify the invariant 10-rank candidate pool without altering it."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Frozen dataset is missing required columns: {sorted(missing)}")
    counts = frame.groupby("query_index", sort=False)["candidate_rank"].agg(list)
    expected = list(range(1, 11))
    invalid = counts[counts.map(lambda ranks: sorted(ranks) != expected)]
    if not invalid.empty:
        raise ValueError(f"Expected exactly ranks 1-10 per query; invalid queries: {invalid.index[:5].tolist()}")
    if (frame["actual_unit_price"] <= 0).any():
        raise ValueError("Frozen dataset contains non-positive actual_unit_price values")


def build_baseline_ledger(frame: pd.DataFrame) -> pd.DataFrame:
    """Select frozen Rank-1 and produce one protocol-compliant row per query."""
    rank_one = frame.loc[frame["candidate_rank"].eq(1)].copy()
    rank_one = rank_one.sort_values("query_index", kind="stable")
    baseline_ape = absolute_percentage_error(rank_one["candidate_unit_price"], rank_one["actual_unit_price"])
    return pd.DataFrame(
        {
            "query_index": rank_one["query_index"].to_numpy(),
            "baseline_rank": 1,
            "baseline_unit_price": rank_one["candidate_unit_price"].to_numpy(),
            "gate_status": "NOT_APPLIED",
            "llm_decision": "BYPASSED",
            "selected_candidate_rank": pd.array([pd.NA] * len(rank_one), dtype="Int64"),
            "selected_unit_price": pd.array([pd.NA] * len(rank_one), dtype="Float64"),
            "actual_unit_price": rank_one["actual_unit_price"].to_numpy(),
            "baseline_absolute_percentage_error": baseline_ape.to_numpy(),
            "llm_absolute_percentage_error": pd.array([pd.NA] * len(rank_one), dtype="Float64"),
            "is_llm_within_10pct": pd.array([pd.NA] * len(rank_one), dtype="boolean"),
        }
    )


def coverage_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute retrieval capacity metrics; never use them for selection."""
    ordered = frame.sort_values(["query_index", "candidate_rank"], kind="stable")
    within_by_rank = ordered.pivot(index="query_index", columns="candidate_rank", values="is_within_10pct")
    total = int(len(within_by_rank))

    def coverage(k: int) -> dict[str, int | float]:
        hits = int(within_by_rank.loc[:, 1:k].any(axis="columns").sum())
        return {"hits": hits, "total": total, "percentage": round(hits / total * 100.0, 6)}

    first_valid_rank = ordered.loc[ordered["is_within_10pct"]].groupby("query_index")["candidate_rank"].min()
    mrr = float((1.0 / first_valid_rank).sum() / total)
    rank_one_failures = (~within_by_rank[1]) & within_by_rank.loc[:, 2:10].any(axis="columns")
    return {
        "top_3_coverage_at_10pct": coverage(3),
        "top_5_coverage_at_10pct": coverage(5),
        "oracle_at_10": coverage(10),
        "mrr_at_10": round(mrr, 6),
        "rank_1_failures_recoverable_in_ranks_2_to_10": {
            "count": int(rank_one_failures.sum()),
            "total": total,
            "percentage": round(float(rank_one_failures.mean() * 100.0), 6),
        },
    }


def run(dataset_path: Path, results_dir: Path) -> dict[str, Any]:
    start = time.perf_counter()
    frame = pd.read_parquet(dataset_path)
    validate_frozen_dataset(frame)
    ledger = build_baseline_ledger(frame)
    summary = {
        "experiment": "frozen_rank_1_baseline",
        "dataset_path": str(dataset_path),
        "dataset_rows": int(len(frame)),
        "query_count": int(len(ledger)),
        "candidates_per_query": 10,
        "selection_rule": "candidate_rank == 1",
        "evaluation_only_fields_used_after_selection": ["actual_unit_price", "is_within_10pct"],
        "baseline": selection_metrics(ledger["baseline_unit_price"], ledger["actual_unit_price"]),
        "candidate_pool_capacity": coverage_metrics(frame),
        "execution_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": round(time.perf_counter() - start, 3),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    ledger.to_parquet(results_dir / "baseline_ledger.parquet", index=False)
    ledger.to_csv(results_dir / "baseline_ledger.csv", index=False)
    (results_dir / "baseline_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the frozen Rank-1 procurement benchmark baseline.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.dataset, args.results_dir), indent=2))
