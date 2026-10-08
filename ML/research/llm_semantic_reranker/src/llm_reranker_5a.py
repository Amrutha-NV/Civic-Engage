"""Experiment 5A: Comparative OpenRouter LLM selection over a closed Top-10
historical candidate set.

The LLM must compare all supplied Top-10 candidates before selecting one.
The experiment remains strictly closed-set: the model may only select one
of the supplied candidate ranks or reject the historical set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

MODULE_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = MODULE_ROOT.parent
PROJECT_ROOT = RESEARCH_ROOT.parent.parent

DEFAULT_DATASET = (
    RESEARCH_ROOT
    / "procurement_benchmark_topk"
    / "frozen_evaluation_dataset.parquet"
)

DEFAULT_RESULTS = MODULE_ROOT / "results"

OPENROUTER_CHAT_COMPLETIONS_URL = (
    "https://openrouter.ai/api/v1/chat/completions"
)


sys.path.insert(0, str(MODULE_ROOT))

from src.baseline import validate_frozen_dataset  # noqa: E402
from src.validator import ValidatedDecision, validate_llm_output  # noqa: E402


# ---------------------------------------------------------------------------
# Fields exposed to the LLM
# ---------------------------------------------------------------------------

TARGET_FIELDS = [
    "query_index",
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

CANDIDATE_FIELDS = [
    "candidate_rank",
    "candidate_description",
    "candidate_unit_price",
    "candidate_score",
]


# ---------------------------------------------------------------------------
# Experiment 5A system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert procurement historical-candidate reranker.

Your task is to select the SINGLE most comparable historical procurement
candidate from the supplied Top-10 candidate set.

You MUST compare ALL supplied candidates before making your final decision.

IMPORTANT RULES:

1. CLOSED-SET DECISION

You may ONLY select one of the supplied candidate ranks.

You must never create a new candidate, modify a candidate, combine
candidates, or construct a new price.

If the supplied candidates are ranks 1 through 10, the selected rank must
be one of those supplied ranks.

--------------------------------------------------

2. NO PRICE GENERATION

You must never:

- generate a price
- estimate a price
- calculate a price
- interpolate a price
- average prices
- modify a price
- infer a price
- output a new price

If you select a candidate, the final predicted price will be exactly the
candidate_unit_price belonging to that supplied candidate.

Your task is candidate comparability and selection, NOT price estimation.

--------------------------------------------------

3. NO OUTSIDE KNOWLEDGE

Use ONLY the information supplied in:

- the target procurement record
- the ten historical candidates

Do not use external knowledge, web knowledge, assumed market prices,
assumed product specifications, or unstated relationships.

--------------------------------------------------

4. DO NOT AUTOMATICALLY SELECT RANK 1

candidate_rank represents the ORIGINAL retrieval ranking.

It is NOT the final answer.

A candidate ranked 2, 3, 4, or lower may be a better historical match
than rank 1 if its available evidence provides stronger comparability.

You must independently evaluate the candidates.

--------------------------------------------------

5. COMPARE ALL SUPPLIED CANDIDATES

Before selecting a candidate, inspect and compare the complete supplied
candidate set.

Do not stop after finding a plausible candidate.

The purpose of this experiment is to determine whether an LLM can improve
upon the original Top-10 ranking through comparative reasoning.

--------------------------------------------------

6. EVIDENCE HIERARCHY

When comparing candidates, consider the following evidence approximately
in this order:

A. Exact or near-exact product/item identity

Does the candidate represent the same actual item or procurement need?

B. Model and specification compatibility

Does the candidate contain the same or highly compatible model,
technical specification, equipment type, or product variant?

C. Brand compatibility

Does the candidate match the target brand when brand information is
available?

D. Commodity code and commodity family compatibility

Does the candidate belong to the same commodity classification?

E. Quantity and unit-of-measure compatibility

When the available information supports the comparison, determine whether
the candidate's described procurement context is compatible with the
target's quantity or unit of measure.

F. Geographic/contextual compatibility

When location information is available, consider whether the historical
candidate is contextually relevant to the target procurement location.

G. Procurement-date/temporal relevance

When date information is available, consider whether the historical
record is reasonably relevant to the target procurement period.

H. Description completeness and specificity

Prefer candidates whose description provides stronger evidence that the
historical record represents the same procurement item.

I. Candidate semantic similarity score

Use candidate_score as supporting evidence.

Do NOT blindly select the candidate with the highest candidate_score.

J. Original candidate rank

Use candidate_rank only as supporting evidence or tie-breaking evidence.

A lower numerical rank does not automatically mean a better final match.

--------------------------------------------------

7. SPECIFICATION MATCHING

Prefer candidates that describe the same actual procurement item rather
than candidates that merely share broad words.

For example:

A candidate with an exact or near-exact model/specification match should
generally be preferred over a candidate that only matches the broad
commodity category.

Similarly, a candidate describing a highly specific equipment variant
should not automatically be considered equivalent to a generic candidate
merely because both contain similar words.

--------------------------------------------------

8. EXACT TEXT MATCHING IS NOT SUFFICIENT BY ITSELF

An exact description match is strong evidence, but still consider the
other available target fields.

Do not assume two records are identical solely because one description
contains or repeats the target description.

Look for supporting evidence in the available fields.

--------------------------------------------------

9. MISSING INFORMATION

Do not invent missing attributes.

If a candidate does not contain enough information to establish a
particular match, treat that evidence as UNKNOWN.

Do not assume that a missing brand, model, location, quantity, or
specification matches the target.

--------------------------------------------------

10. CONTRADICTIONS

Penalize candidates whose available information conflicts with the target.

For example, if a candidate clearly describes a different product,
commodity, equipment type, model, or application, it should generally be
ranked below candidates with stronger compatibility even if its semantic
similarity score is high.

--------------------------------------------------

11. CANDIDATE SCORE

candidate_score is retrieval evidence.

It is NOT ground truth.

Do not simply choose:

"the candidate with the highest candidate_score."

Instead, use candidate_score together with the available semantic and
specification evidence.

--------------------------------------------------

12. CANDIDATE PRICE

candidate_unit_price is part of the historical candidate record.

Do NOT use the numerical attractiveness of the price as evidence that a
candidate is comparable.

For example, do not reason:

"This price looks close to what the target should cost, therefore this
candidate is better."

The experiment is testing historical candidate comparability, not price
guessing.

--------------------------------------------------

13. HISTORICAL INSUFFICIENCY

If none of the supplied candidates provides sufficient evidence of being
a comparable historical procurement, return:

HISTORICAL_INSUFFICIENT

In that case:

selected_candidate_rank must be null.

Do not force a selection simply because ten candidates were supplied.

--------------------------------------------------

14. FINAL COMPARATIVE DECISION

If at least one candidate is sufficiently comparable:

1. Compare all ten candidates.
2. Identify the strongest candidate(s).
3. Consider the important differences between the strongest candidates.
4. Select exactly ONE supplied candidate rank.
5. Explain why that candidate is preferable to the strongest alternatives.

The reason should describe the actual comparative evidence.

Avoid generic explanations such as:

"Candidate 1 is the best match."

Instead explain what makes the selected candidate stronger.

--------------------------------------------------

15. OUTPUT FORMAT

Return JSON only.

Do not return markdown.
Do not return bullet points.
Do not return analysis outside the JSON object.
Do not return additional fields.

Return exactly:

{
  "decision": "HISTORICAL_ACCEPTED" or "HISTORICAL_INSUFFICIENT",
  "selected_candidate_rank": integer or null,
  "reason": "concise comparative explanation"
}

If decision is HISTORICAL_ACCEPTED:

- selected_candidate_rank MUST be one of the supplied candidate ranks.

If decision is HISTORICAL_INSUFFICIENT:

- selected_candidate_rank MUST be null.

The reason must explain the comparative decision using only evidence
available in the supplied target and candidates.
"""


# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------

def load_project_env() -> None:
    """Load local .env entries without overwriting process configuration."""

    env_path = PROJECT_ROOT / ".env"

    print(f"Loading environment from: {env_path}")

    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)

        os.environ.setdefault(
            key.strip(),
            value.strip().strip('"').strip("'"),
        )


# ---------------------------------------------------------------------------
# JSON-safe conversion
# ---------------------------------------------------------------------------

def json_safe(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-safe Python values."""

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(query_rows: pd.DataFrame) -> str:
    """
    Construct the LLM input exclusively from permitted frozen-dataset fields.

    All Top-10 candidates are supplied together so the model can perform
    comparative reranking.
    """

    first = query_rows.iloc[0]

    query = {
        field: json_safe(first[field])
        for field in TARGET_FIELDS
    }

    candidates = [
        {
            field: json_safe(row[field])
            for field in CANDIDATE_FIELDS
        }
        for _, row in query_rows.iterrows()
    ]

    payload = {
        "query": query,
        "candidates": candidates,
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# OpenRouter API request
# ---------------------------------------------------------------------------

def request_decision(
    prompt: str,
    api_key: str,
    model: str,
    timeout_seconds: int,
) -> str:
    """
    Send the comparative Top-10 selection request to OpenRouter.

    The OpenRouter API is OpenAI-compatible and returns a standard
    chat-completion response.
    """

    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {
            "type": "json_object"
        },
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
    }

    request = urllib.request.Request(
        OPENROUTER_CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",

            # Optional OpenRouter attribution headers.
            "HTTP-Referer": "http://localhost",
            "X-Title": "CivicEngage Procurement Research",
        },
        method="POST",
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=timeout_seconds,
        ) as response:

            body = json.loads(
                response.read().decode("utf-8")
            )

    except urllib.error.HTTPError as error:

        error_body = ""

        try:
            error_body = error.read().decode("utf-8")
        except Exception:
            pass

        raise RuntimeError(
            f"OpenRouter request failed with HTTP {error.code}: "
            f"{error_body[:1000]}"
        ) from error

    except urllib.error.URLError as error:

        raise RuntimeError(
            f"OpenRouter request could not be completed: "
            f"{error.reason}"
        ) from error

    try:

        return body["choices"][0]["message"]["content"]

    except (KeyError, IndexError, TypeError) as error:

        raise RuntimeError(
            "OpenRouter response did not contain a valid chat completion"
        ) from error


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------

def run(
    dataset_path: Path,
    results_dir: Path,
    max_queries: int,
    timeout_seconds: int,
) -> pd.DataFrame:
    """
    Run a bounded Experiment 5A.

    Each query generates exactly one OpenRouter API call.
    """

    if max_queries <= 0:

        raise ValueError(
            "--max-queries must be positive; "
            "this prevents accidental full paid API evaluations"
        )

    # ---------------------------------------------------------
    # Load environment
    # ---------------------------------------------------------

    load_project_env()

    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL")

    if not api_key:

        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured in the project "
            ".env or process environment"
        )

    if not model:

        raise RuntimeError(
            "OPENROUTER_MODEL is not configured in the project "
            ".env or process environment"
        )

    print(f"OpenRouter model: {model}")
    print(f"Maximum queries: {max_queries}")
    print()

    # ---------------------------------------------------------
    # Load frozen evaluation dataset
    # ---------------------------------------------------------

    print(f"Loading dataset: {dataset_path}")

    frame = pd.read_parquet(dataset_path)

    validate_frozen_dataset(frame)

    print(
        f"Loaded {len(frame):,} candidate rows "
        f"across {frame['query_index'].nunique():,} queries."
    )

    # ---------------------------------------------------------
    # Select bounded query set
    # ---------------------------------------------------------

    query_ids = (
        frame["query_index"]
        .drop_duplicates()
        .head(max_queries)
        .tolist()
    )

    records: list[dict[str, Any]] = []

    # ---------------------------------------------------------
    # Process each query
    # ---------------------------------------------------------

    for position, query_index in enumerate(
        query_ids,
        start=1,
    ):

        print(
            f"[{position}/{len(query_ids)}] "
            f"Processing query {query_index}..."
        )

        rows = (
            frame.loc[
                frame["query_index"].eq(query_index)
            ]
            .sort_values("candidate_rank")
        )

        # -----------------------------------------------------
        # Verify Top-10 candidate set
        # -----------------------------------------------------

        supplied_ranks = set(
            rows["candidate_rank"].tolist()
        )

        print(
            f"  Candidates supplied: "
            f"{sorted(supplied_ranks)}"
        )

        try:

            # -------------------------------------------------
            # Build comparative Top-10 prompt
            # -------------------------------------------------

            prompt = build_prompt(rows)

            # -------------------------------------------------
            # Request LLM decision
            # -------------------------------------------------

            raw_response = request_decision(
                prompt=prompt,
                api_key=api_key,
                model=model,
                timeout_seconds=timeout_seconds,
            )

            # -------------------------------------------------
            # Validate LLM response using EXISTING validator
            # -------------------------------------------------

            decision = validate_llm_output(
                raw_response,
                supplied_ranks,
            )

            validation_status = "VALID"

            print(
                f"  Decision: {decision.decision}"
            )

            print(
                f"  Selected rank: "
                f"{decision.selected_candidate_rank}"
            )

        except (RuntimeError, ValueError) as error:

            decision = ValidatedDecision(
                "HISTORICAL_INSUFFICIENT",
                None,
                str(error),
            )

            validation_status = "INVALID_OR_API_FAILURE"

            print(
                f"  ERROR: {error}"
            )

        # -----------------------------------------------------
        # Retrieve selected historical price
        # -----------------------------------------------------

        selected_price = None

        if decision.selected_candidate_rank is not None:

            matching_rows = rows.loc[
                rows["candidate_rank"].eq(
                    decision.selected_candidate_rank
                )
            ]

            if not matching_rows.empty:

                selected_price = float(
                    matching_rows[
                        "candidate_unit_price"
                    ].iloc[0]
                )

        # -----------------------------------------------------
        # Store ledger record
        # -----------------------------------------------------

        records.append(
            {
                "query_index": query_index,
                "llm_decision": decision.decision,
                "selected_candidate_rank": (
                    decision.selected_candidate_rank
                ),
                "selected_unit_price": selected_price,
                "reason": decision.reason,
                "validation_status": validation_status,
            }
        )

        print()

    # ---------------------------------------------------------
    # Build result ledger
    # ---------------------------------------------------------

    ledger = pd.DataFrame(records)

    # ---------------------------------------------------------
    # Save Experiment 5A separately
    # ---------------------------------------------------------

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    parquet_path = (
        results_dir
        / "llm_reranker_5a_ledger.parquet"
    )

    csv_path = (
        results_dir
        / "llm_reranker_5a_ledger.csv"
    )

    ledger.to_parquet(
        parquet_path,
        index=False,
    )

    ledger.to_csv(
        csv_path,
        index=False,
    )

    print(
        f"Saved {len(ledger)} Experiment 5A decisions."
    )

    print(
        f"Parquet: {parquet_path}"
    )

    print(
        f"CSV:     {csv_path}"
    )

    return ledger


# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Run bounded Experiment 5A: "
            "comparative OpenRouter LLM reranking "
            "over a closed Top-10 historical candidate set."
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
        "--max-queries",
        type=int,
        required=True,
        help=(
            "Explicit API-call limit. "
            "Required to prevent accidental full paid evaluation."
        ),
    )

    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    args = parse_args()

    output = run(
        dataset_path=args.dataset,
        results_dir=args.results_dir,
        max_queries=args.max_queries,
        timeout_seconds=args.timeout_seconds,
    )

    print(
        f"Completed Experiment 5A with "
        f"{len(output)} queries."
    )