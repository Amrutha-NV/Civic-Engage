"""Experiment 5: price-blind LLM reranker using OpenRouter.

The LLM receives only target information and the supplied Top-10 candidate
descriptions / retrieval evidence. Candidate prices are deliberately hidden
from the LLM.

The LLM may:
    - select exactly one supplied candidate rank (1-10), or
    - reject with HISTORICAL_INSUFFICIENT.

After the LLM selects a candidate rank, Python retrieves the corresponding
candidate_unit_price from the frozen evaluation dataset.

Evaluation metrics are calculated using the existing evaluation.metrics module.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODULE_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = MODULE_ROOT.parent

DEFAULT_DATASET = (
    RESEARCH_ROOT
    / "procurement_benchmark_topk"
    / "frozen_evaluation_dataset.parquet"
)

DEFAULT_RESULTS = MODULE_ROOT / "results"

# Project root contains the .env file.
PROJECT_ROOT = RESEARCH_ROOT.parent.parent

# Make evaluation and src importable.
sys.path.insert(0, str(MODULE_ROOT))

from evaluation.metrics import (  # noqa: E402
    absolute_percentage_error,
    selection_metrics,
)
from src.baseline import validate_frozen_dataset  # noqa: E402


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv(PROJECT_ROOT / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")

OPENROUTER_CHAT_COMPLETIONS_URL = (
    "https://openrouter.ai/api/v1/chat/completions"
)


# ---------------------------------------------------------------------------
# Canonical fields
# ---------------------------------------------------------------------------

TARGET_FIELDS = [
    "target_description",
    "target_quantity",
    "target_unit_of_measure",
    "target_procurement_date",
    "target_city",
    "target_state",
    "target_brand",
    "target_model",
    "target_commodity_code",
    "target_commodity_family",
]

# IMPORTANT:
# candidate_unit_price is deliberately NOT included.
#
# Candidate-side structured attributes are also not available in the frozen
# candidate pool. The LLM may only reason about information explicitly
# contained in candidate_description.
CANDIDATE_FIELDS = [
    "candidate_rank",
    "candidate_description",
    "candidate_score",
]


# ---------------------------------------------------------------------------
# LLM system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a procurement-history candidate selection system.

Your task is to select the single most comparable historical procurement
candidate from the supplied Top-10 candidates for the target procurement item.

IMPORTANT RULES:

1. You must compare ALL supplied candidates before deciding.

2. You may ONLY select one of the supplied candidate ranks 1 through 10.

3. You must NOT generate a new price.

4. You must NOT modify, average, interpolate, estimate, or combine candidate
   prices.

5. Candidate prices are NOT provided to you. Do not request them and do not
   infer price-related conclusions.

6. You must NOT use outside knowledge or external information.

7. Do NOT automatically select candidate rank 1.

8. Evaluate comparability using the information actually supplied in the
   target and candidate descriptions.

9. Consider, when explicitly available:
   - item/product identity
   - specifications
   - brand
   - model
   - commodity code
   - commodity family
   - quantity or unit of measure
   - geographic context
   - temporal context
   - description quality and specificity

10. Exact textual matches are strong evidence, but an exact generic category
    phrase is not automatically stronger than a more specific candidate that
    explicitly identifies the same product.

11. candidate_score may be used as supporting retrieval evidence, but it must
    not override strong contradictory product evidence.

12. candidate_rank may be used as supporting evidence or a tie-breaker when
    candidates are otherwise equivalent.

13. Missing information is UNKNOWN. Do not invent or assume missing
    specifications, brands, models, quantities, locations, dates, or product
    details.

14. Contradictory product information should reduce comparability.

15. IMPORTANT SELECTION POLICY:
    This is a reranking task. You should select the BEST AVAILABLE candidate
    whenever there is reasonable evidence that it is comparable to the
    target.

16. The candidate does NOT need to be a perfect or exact match.

17. If several candidates are imperfect but one is clearly more comparable
    than the others, select that candidate.

18. Do NOT reject merely because:
    - some specifications are missing,
    - the candidate description is generic,
    - the candidate does not exactly match every target detail,
    - the candidate lacks information that is unavailable in the dataset.

19. Use HISTORICAL_INSUFFICIENT only when the supplied candidate pool is
    genuinely unrelated to the target and there is no reasonably comparable
    candidate among the ten candidates.

20. When uncertain between selecting a reasonably comparable candidate and
    rejecting the entire pool, prefer selecting the best available candidate,
    provided it is not clearly contradictory or unrelated.

21. Do NOT automatically select candidate rank 1. Compare all candidates
    before making the decision.
Do not include markdown fences or any additional fields.
""".strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def json_safe(value: Any) -> Any:
    """Convert pandas / numpy values into JSON-safe Python values."""
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


def row_to_target(row: pd.Series) -> dict[str, Any]:
    """Build the target object supplied to the LLM."""
    return {
        field: json_safe(row[field])
        for field in TARGET_FIELDS
    }


def rows_to_candidates(group: pd.DataFrame) -> list[dict[str, Any]]:
    """Build the price-blind candidate list supplied to the LLM."""
    candidates: list[dict[str, Any]] = []

    for _, row in group.sort_values("candidate_rank").iterrows():
        candidate = {
            field: json_safe(row[field])
            for field in CANDIDATE_FIELDS
        }

        candidates.append(candidate)

    return candidates


def build_prompt(
    target: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> str:
    """Build the user message for one query."""
    payload = {
        "target": target,
        "candidates": candidates,
    }

    return (
        "Compare the target procurement item against ALL supplied "
        "historical candidates.\n\n"
        "Select the most comparable candidate rank, or reject the historical "
        "set if none is sufficiently comparable.\n\n"
        "Remember: candidate prices are intentionally hidden. Do not infer "
        "or discuss price.\n\n"
        "INPUT:\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}"
    )


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------

def request_decision(
    prompt: str,
    api_key: str,
    model: str,
    timeout: int = 120,
) -> dict[str, Any]:
    """Send one price-blind selection request to OpenRouter."""

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured. "
            "Add it to the project .env file."
        )

    if not model:
        raise RuntimeError(
            "OPENROUTER_MODEL is not configured. "
            "Add it to the project .env file."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "CivicEngage Procurement Research",
    }

    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "response_format": {
            "type": "json_object"
        },
    }

    response = requests.post(
        OPENROUTER_CHAT_COMPLETIONS_URL,
        headers=headers,
        json=body,
        timeout=timeout,
    )

    response.raise_for_status()

    data = response.json()

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected OpenRouter response structure: {data}"
        ) from exc

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(str(item["text"]))

        content = "".join(text_parts)

    if not isinstance(content, str):
        raise RuntimeError(
            f"Unexpected LLM content type: {type(content).__name__}"
        )

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"LLM returned invalid JSON: {content}"
        ) from exc


# ---------------------------------------------------------------------------
# Existing validator
# ---------------------------------------------------------------------------

def validate_llm_output(
    output: dict[str, Any],
) -> tuple[str, int | None, str]:
    """
    Validate the LLM's closed-set decision.

    This validator intentionally does not inspect prices.
    """

    if not isinstance(output, dict):
        raise ValueError("LLM output must be a JSON object")

    required_fields = {
        "decision",
        "selected_candidate_rank",
        "reason",
    }

    if set(output.keys()) != required_fields:
        raise ValueError(
            "LLM output must contain exactly: "
            "decision, selected_candidate_rank, reason"
        )

    decision = output["decision"]
    selected_rank = output["selected_candidate_rank"]
    reason = output["reason"]

    if decision not in {
        "HISTORICAL_ACCEPTED",
        "HISTORICAL_INSUFFICIENT",
    }:
        raise ValueError(f"Invalid decision: {decision}")

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")

    if decision == "HISTORICAL_INSUFFICIENT":
        if selected_rank is not None:
            raise ValueError(
                "HISTORICAL_INSUFFICIENT requires "
                "selected_candidate_rank=null"
            )

        return decision, None, reason.strip()

    if isinstance(selected_rank, bool):
        raise ValueError(
            "selected_candidate_rank must be an integer rank"
        )

    if not isinstance(selected_rank, int):
        raise ValueError(
            "selected_candidate_rank must be an integer"
        )

    if selected_rank not in range(1, 11):
        raise ValueError(
            "selected_candidate_rank must be between 1 and 10"
        )

    return decision, selected_rank, reason.strip()


# ---------------------------------------------------------------------------
# LLM reranking
# ---------------------------------------------------------------------------

def llm_selection(
    frame: pd.DataFrame,
    api_key: str,
    model: str,
    max_queries: int | None,
    timeout: int,
) -> pd.DataFrame:
    """Run the price-blind LLM over frozen Top-10 candidate pools."""

    records: list[dict[str, Any]] = []

    query_indices = sorted(frame["query_index"].unique())

    if max_queries is not None:
        query_indices = query_indices[:max_queries]

    total_queries = len(query_indices)

    for position, query_index in enumerate(query_indices, start=1):
        group = frame.loc[
            frame["query_index"].eq(query_index)
        ].copy()

        if len(group) != 10:
            raise ValueError(
                f"Query {query_index} does not contain exactly 10 candidates"
            )

        group = group.sort_values(
            "candidate_rank",
            kind="stable",
        )

        target = row_to_target(group.iloc[0])
        candidates = rows_to_candidates(group)

        prompt = build_prompt(
            target=target,
            candidates=candidates,
        )

        try:
            raw_output = request_decision(
                prompt=prompt,
                api_key=api_key,
                model=model,
                timeout=timeout,
            )

            decision, selected_rank, reason = validate_llm_output(
                raw_output
            )

            validation_status = "VALID"

        except Exception as exc:
            decision = "HISTORICAL_INSUFFICIENT"
            selected_rank = None
            reason = f"LLM/evaluation error: {exc}"
            validation_status = "ERROR"

        selected_unit_price = None

        if selected_rank is not None:
            selected_rows = group.loc[
                group["candidate_rank"].eq(selected_rank)
            ]

            if len(selected_rows) != 1:
                raise ValueError(
                    f"Selected candidate rank {selected_rank} "
                    f"was not uniquely found for query {query_index}"
                )

            selected_unit_price = float(
                selected_rows.iloc[0]["candidate_unit_price"]
            )

        baseline_row = group.loc[
            group["candidate_rank"].eq(1)
        ]

        if len(baseline_row) != 1:
            raise ValueError(
                f"Query {query_index} does not have exactly one rank-1 candidate"
            )

        baseline_unit_price = float(
            baseline_row.iloc[0]["candidate_unit_price"]
        )

        actual_unit_price = float(
            group.iloc[0]["actual_unit_price"]
        )

        records.append(
            {
                "query_index": int(query_index),
                "llm_decision": decision,
                "selected_candidate_rank": selected_rank,
                "selected_unit_price": selected_unit_price,
                "baseline_rank": 1,
                "baseline_unit_price": baseline_unit_price,
                "actual_unit_price": actual_unit_price,
                "reason": reason,
                "validation_status": validation_status,
            }
        )

        print(
            f"[{position}/{total_queries}] "
            f"query={query_index} "
            f"decision={decision} "
            f"selected_rank={selected_rank}"
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def add_evaluation_columns(
    ledger: pd.DataFrame,
) -> pd.DataFrame:
    """Add APE and ±10% indicators while safely handling LLM rejection."""

    ledger = ledger.copy()

    # Baseline always has a selected price.
    ledger["baseline_absolute_percentage_error"] = (
        absolute_percentage_error(
            ledger["baseline_unit_price"],
            ledger["actual_unit_price"],
        )
    )

    accepted = ledger["llm_decision"].eq(
        "HISTORICAL_ACCEPTED"
    )

    # Rejected queries deliberately have no prediction.
    ledger["llm_absolute_percentage_error"] = pd.NA
    ledger["is_llm_within_10pct"] = pd.NA

    if accepted.any():
        accepted_ape = absolute_percentage_error(
            ledger.loc[accepted, "selected_unit_price"],
            ledger.loc[accepted, "actual_unit_price"],
        )

        ledger.loc[
            accepted,
            "llm_absolute_percentage_error",
        ] = accepted_ape

        ledger.loc[
            accepted,
            "is_llm_within_10pct",
        ] = accepted_ape <= 10.0

    return ledger


def build_metrics_summary(
    ledger: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate experiment metrics using the existing evaluation module."""

    baseline_metrics = selection_metrics(
        ledger["baseline_unit_price"],
        ledger["actual_unit_price"],
    )

    accepted = ledger["llm_decision"].eq(
        "HISTORICAL_ACCEPTED"
    )

    accepted_rows = ledger.loc[accepted].copy()

    if len(accepted_rows) > 0:
        llm_selection_metric_values = selection_metrics(
            accepted_rows["selected_unit_price"],
            accepted_rows["actual_unit_price"],
        )

        accepted_hits_10 = int(
            (
                accepted_rows["llm_absolute_percentage_error"]
                <= 10.0
            ).sum()
        )
    else:
        llm_selection_metric_values = None
        accepted_hits_10 = 0

    total_queries = int(len(ledger))
    accepted_count = int(accepted.sum())
    rejected_count = int(total_queries - accepted_count)

    if total_queries > 0:
        rejection_rate = (
            rejected_count / total_queries * 100.0
        )

        # End-to-end Accuracy@10 treats rejection as failure.
        end_to_end_accuracy_10 = (
            accepted_hits_10 / total_queries * 100.0
        )
    else:
        rejection_rate = 0.0
        end_to_end_accuracy_10 = 0.0

    selected_rank_distribution = {
        str(rank): int(
            (
                ledger["selected_candidate_rank"] == rank
            ).sum()
        )
        for rank in range(1, 11)
    }

    rank_changes = int(
        (
            ledger["selected_candidate_rank"].notna()
            & ledger["selected_candidate_rank"].ne(1)
        ).sum()
    )

    return {
        "baseline": baseline_metrics,

        "llm_reranker": {
            # Metrics calculated using the existing evaluator and only
            # accepted LLM predictions.
            "selection_metrics": llm_selection_metric_values,

            # Number of queries for which the LLM actually selected a
            # candidate and therefore produced a price prediction.
            "evaluated_queries": accepted_count,

            # LLM rejections are not treated as price predictions.
            "rejected_queries": rejected_count,

            "rejection_rate_percent": round(
                rejection_rate,
                6,
            ),

            # Number of accepted predictions that were within ±10%.
            "accepted_accuracy_at_10_hits": accepted_hits_10,

            # Accuracy over ALL queries, where rejection counts as failure.
            "end_to_end_accuracy_at_10pct": round(
                end_to_end_accuracy_10,
                6,
            ),
        },

        "selected_rank_distribution": selected_rank_distribution,

        "rank_changes_from_baseline": rank_changes,

        "rank_1_retention": int(
            (
                ledger["selected_candidate_rank"] == 1
            ).sum()
        ),
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run(
    dataset_path: Path,
    results_dir: Path,
    model: str,
    max_queries: int | None,
    timeout: int,
) -> dict[str, Any]:

    start = time.perf_counter()

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    frame = pd.read_parquet(dataset_path)

    validate_frozen_dataset(frame)

    ledger = llm_selection(
        frame=frame,
        api_key=OPENROUTER_API_KEY,
        model=model,
        max_queries=max_queries,
        timeout=timeout,
    )

    ledger = add_evaluation_columns(ledger)

    metrics = build_metrics_summary(ledger)

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------------
    # Save ledger
    # -----------------------------------------------------------------------

    ledger_path = results_dir / "llm_reranker_ledger.parquet"
    ledger_csv_path = results_dir / "llm_reranker_ledger.csv"

    ledger.to_parquet(
        ledger_path,
        index=False,
    )

    ledger.to_csv(
        ledger_csv_path,
        index=False,
    )

    # -----------------------------------------------------------------------
    # Build final summary
    # -----------------------------------------------------------------------

    summary: dict[str, Any] = {
        "experiment": "llm_reranker",

        "dataset_path": str(dataset_path),

        "query_count": int(len(ledger)),

        "model": model,

        "endpoint": OPENROUTER_CHAT_COMPLETIONS_URL,

        "device": "api",

        "selection_inputs": [
            "target_description",
            "target_quantity",
            "target_unit_of_measure",
            "target_procurement_date",
            "target_city",
            "target_state",
            "target_brand",
            "target_model",
            "target_commodity_code",
            "target_commodity_family",
            "candidate_rank",
            "candidate_description",
            "candidate_score",
        ],

        "excluded_from_llm_selection": [
            "candidate_unit_price",
            "actual_unit_price",
            "absolute_percentage_error",
            "is_within_10pct",
        ],

        "decision_contract": {
            "accepted_decision": "HISTORICAL_ACCEPTED",
            "rejection_decision": "HISTORICAL_INSUFFICIENT",
            "allowed_candidate_ranks": list(range(1, 11)),
            "price_visible_to_llm": False,
            "generated_prices_allowed": False,
        },

        "evaluation": metrics,

        "execution_utc": datetime.now(UTC).isoformat(),

        "runtime_seconds": round(
            time.perf_counter() - start,
            3,
        ),
    }

    # -----------------------------------------------------------------------
    # Save metrics
    # -----------------------------------------------------------------------

    metrics_path = results_dir / "llm_reranker_metrics.json"

    metrics_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the price-blind OpenRouter LLM reranker "
            "on the frozen Top-10 procurement benchmark."
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
        default=OPENROUTER_MODEL,
    )

    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help=(
            "Maximum number of queries to evaluate. "
            "Use 10 for a smoke test or 100 for the pilot."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="OpenRouter request timeout in seconds.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    result = run(
        dataset_path=args.dataset,
        results_dir=args.results_dir,
        model=args.model,
        max_queries=args.max_queries,
        timeout=args.timeout,
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )