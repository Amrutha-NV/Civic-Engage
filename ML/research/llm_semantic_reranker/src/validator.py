"""Strict validator for the closed-context historical-candidate contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

ALLOWED_DECISIONS = {"HISTORICAL_ACCEPTED", "HISTORICAL_INSUFFICIENT"}
ALLOWED_KEYS = {"decision", "selected_candidate_rank", "reason"}


@dataclass(frozen=True)
class ValidatedDecision:
    decision: str
    selected_candidate_rank: int | None
    reason: str


def validate_llm_output(raw_output: str, available_ranks: set[int]) -> ValidatedDecision:
    """Parse a decision and reject prices, extra fields, invalid ranks, and bad JSON."""
    try:
        payload: Any = json.loads(raw_output)
    except json.JSONDecodeError as error:
        raise ValueError("LLM response is not valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != ALLOWED_KEYS:
        raise ValueError("LLM response must contain exactly decision, selected_candidate_rank, and reason")
    decision, rank, reason = payload["decision"], payload["selected_candidate_rank"], payload["reason"]
    if decision not in ALLOWED_DECISIONS:
        raise ValueError("LLM decision is not permitted")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("LLM reason must be a non-empty string")
    if decision == "HISTORICAL_ACCEPTED":
        if not isinstance(rank, int) or isinstance(rank, bool) or rank not in available_ranks:
            raise ValueError("Accepted response must select an existing candidate rank")
    elif rank is not None:
        raise ValueError("Insufficient-history response must use null selected_candidate_rank")
    return ValidatedDecision(decision=decision, selected_candidate_rank=rank, reason=reason.strip())
