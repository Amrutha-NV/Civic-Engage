"""Experiment 2: explainable structured matching from verified candidate text.

The frozen artifact has structured fields only on the target. Candidate-side
brand, model and commodity-code evidence is therefore derived solely by exact
textual presence in candidate_description. No unavailable candidate metadata is
invented and no price, score, rank, or evaluation field affects the score.
"""

from __future__ import annotations

import argparse
import json
import re
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
from src.baseline import validate_frozen_dataset  # noqa: E402


def normalise(value: object) -> str:
    """Normalise text for literal attribute matching without semantic inference."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())).strip()


def contains_attribute(candidate_description: str, attribute: str) -> bool:
    """Match a non-empty whole normalised attribute sequence in candidate text."""
    return bool(attribute) and f" {attribute} " in f" {candidate_description} "


def score_attributes(frame: pd.DataFrame) -> pd.DataFrame:
    """Score brand, model, and commodity code only when they are available."""
    columns = [
        "query_index", "candidate_rank", "target_brand", "target_model", "target_commodity_code", "candidate_description",
    ]
    scored = frame[columns].copy()
    scored["candidate_text"] = scored["candidate_description"].map(normalise)
    scored["target_brand_value"] = scored["target_brand"].map(normalise)
    scored["target_model_value"] = scored["target_model"].map(normalise)
    scored["target_commodity_code_value"] = scored["target_commodity_code"].map(normalise)

    scored["brand_match"] = [
        contains_attribute(candidate, attribute)
        for candidate, attribute in zip(scored["candidate_text"], scored["target_brand_value"], strict=True)
    ]
    scored["model_match"] = [
        contains_attribute(candidate, attribute)
        for candidate, attribute in zip(scored["candidate_text"], scored["target_model_value"], strict=True)
    ]
    scored["commodity_code_match"] = [
        contains_attribute(candidate, attribute)
        for candidate, attribute in zip(scored["candidate_text"], scored["target_commodity_code_value"], strict=True)
    ]

    # Model is more specific than brand; commodity-code evidence is permitted
    # only if that literal code appears in the candidate description.
    scored["attribute_score"] = (
        scored["brand_match"].astype(float) * 0.25
        + scored["model_match"].astype(float) * 0.65
        + scored["commodity_code_match"].astype(float) * 0.10
    )
    return scored


def select_candidates(scored: pd.DataFrame) -> pd.DataFrame:
    """Choose the highest attribute score; use rank only as deterministic tie-break."""
    return (
        scored.sort_values(["query_index", "attribute_score", "candidate_rank"], ascending=[True, False, True], kind="stable")
        .drop_duplicates("query_index", keep="first")
        [["query_index", "candidate_rank", "attribute_score", "brand_match", "model_match", "commodity_code_match"]]
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
        raise ValueError("Attribute scorer did not select exactly one candidate for every query")
    ledger["baseline_rank"] = 1
    ledger["gate_status"] = "NOT_APPLIED"
    ledger["llm_decision"] = "HISTORICAL_ACCEPTED"
    ledger["baseline_absolute_percentage_error"] = absolute_percentage_error(ledger["baseline_unit_price"], ledger["actual_unit_price"])
    ledger["llm_absolute_percentage_error"] = absolute_percentage_error(ledger["selected_unit_price"], ledger["actual_unit_price"])
    ledger["is_llm_within_10pct"] = ledger["llm_absolute_percentage_error"] <= 10.0
    return ledger


def run(dataset_path: Path, results_dir: Path) -> dict[str, Any]:
    start = time.perf_counter()
    frame = pd.read_parquet(dataset_path)
    validate_frozen_dataset(frame)
    scored = score_attributes(frame)
    selected = select_candidates(scored)
    ledger = build_ledger(frame, selected)
    rank_distribution = {str(rank): int((ledger["selected_candidate_rank"] == rank).sum()) for rank in range(1, 11)}
    summary = {
        "experiment": "structured_attribute_reranker",
        "dataset_path": str(dataset_path),
        "query_count": int(len(ledger)),
        "selection_inputs": ["target_brand", "target_model", "target_commodity_code", "candidate_description"],
        "excluded_unavailable_candidate_attributes": ["quantity", "unit_of_measure", "award_date", "vendor", "location", "commodity_family"],
        "excluded_from_selection": ["candidate_unit_price", "candidate_score", "actual_unit_price", "absolute_percentage_error", "is_within_10pct"],
        "weights": {"brand_literal_match": 0.25, "model_literal_match": 0.65, "commodity_code_literal_match": 0.10},
        "tie_break_rule": "lowest candidate_rank after equal attribute score",
        "match_counts": {name: int(scored[name].sum()) for name in ["brand_match", "model_match", "commodity_code_match"]},
        "baseline": selection_metrics(ledger["baseline_unit_price"], ledger["actual_unit_price"]),
        "structured_reranker": selection_metrics(ledger["selected_unit_price"], ledger["actual_unit_price"]),
        "selected_rank_distribution": rank_distribution,
        "execution_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": round(time.perf_counter() - start, 3),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    ledger.to_parquet(results_dir / "structured_attribute_ledger.parquet", index=False)
    ledger.to_csv(results_dir / "structured_attribute_ledger.csv", index=False)
    (results_dir / "structured_attribute_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run literal structured-attribute reranking over frozen Top-10 candidates.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.dataset, args.results_dir), indent=2))
