
"""Asynchronous Top-K inference API plus current-market evidence lookup."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from fastapi.middleware.cors import CORSMiddleware

from ML.research.llm_semantic_reranker.src import learned_reranker as pipeline
from ML.research.current_market_evidence.src.market_search import search_market


ROOT = Path(__file__).resolve().parents[4]
DATASET = (
    ROOT
    / "ML"
    / "research"
    / "procurement_benchmark_topk"
    / "frozen_evaluation_dataset.parquet"
)

K = pipeline.K

app = FastAPI(
    title="Learned Semantic Reranker and Market Evidence API",
    version="1.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------
# EXISTING LEARNED RERANKER REQUEST MODELS
# -------------------------------------------------------------------

class Query(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(min_length=1, max_length=128)
    target_description: str
    target_quantity: float | None = None
    target_unit_of_measure: str | None = None
    target_brand: str | None = None
    target_model: str | None = None
    target_commodity_code: str | None = None
    target_commodity_family: str | None = None


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_rank: int = Field(ge=1, le=K)
    candidate_description: str
    candidate_unit_price: float = Field(ge=0)
    candidate_score: float


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: Query
    candidates: list[Candidate] = Field(min_length=K, max_length=K)


# -------------------------------------------------------------------
# NEW CURRENT-MARKET REQUEST MODEL
# -------------------------------------------------------------------

class MarketBenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_name: str = Field(min_length=3, max_length=500)
    currency: str = Field(default="INR", min_length=3, max_length=3)
    unit: str = Field(default="EA", min_length=1, max_length=30)


# -------------------------------------------------------------------
# EXISTING RERANKER STATE
# -------------------------------------------------------------------

_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()
_model = None
_embedder = None


def _as_frame(request: RerankRequest) -> pd.DataFrame:
    query = request.query.model_dump(exclude={"query_id"})
    rows = []

    for candidate in request.candidates:
        rows.append({**query, **candidate.model_dump()})

    return (
        pd.DataFrame(rows)
        .sort_values("candidate_rank", kind="mergesort")
        .reset_index(drop=True)
    )


def _load_serving_model():
    """Fit once using the existing frozen training data."""
    global _model

    if _model is not None:
        return _model

    with _lock:
        if _model is not None:
            return _model

        if not DATASET.exists():
            raise FileNotFoundError(
                f"Training dataset not found: {DATASET}"
            )

        train = pipeline.load_dataset(DATASET, 0)

        target = [
            pipeline.clean_text(x)
            for x in train["target_description"].to_numpy()[::K]
        ]

        candidate = [
            pipeline.clean_text(x)
            for x in train["candidate_description"].to_numpy()
        ]

        cache = (
            ROOT
            / "ML"
            / "research"
            / "llm_semantic_reranker"
            / "results"
            / "learned_reranker_exp7"
            / "full"
            / "semantic_cache.npz"
        )

        sem_tc, sem_cc = pipeline.compute_semantic(
            train, target, candidate, "minilm", cache, True, 42
        )

        features, _ = pipeline.build_features(
            train, sem_tc, sem_cc, target, candidate
        )

        price = pd.to_numeric(
            train["candidate_unit_price"], errors="coerce"
        ).to_numpy(float).reshape(-1, K)

        actual = pd.to_numeric(
            train["actual_unit_price"], errors="coerce"
        ).to_numpy(float).reshape(-1, K)[:, 0]

        ape = (
            np.abs(price - actual[:, None])
            / np.maximum(np.abs(actual[:, None]), 1e-9)
            * 100
        )

        labels = (ape <= 10).astype(np.uint8).reshape(-1)

        fitted = pipeline.make_binary(42)
        fitted.fit(features.to_numpy(dtype=np.float32), labels)

        _model = fitted

    return _model


def _score(request: RerankRequest) -> dict[str, Any]:
    model = _load_serving_model()
    frame = _as_frame(request)

    target = [
        pipeline.clean_text(frame["target_description"].iloc[0])
    ]

    descriptions = [
        pipeline.clean_text(x)
        for x in frame["candidate_description"].tolist()
    ]

    global _embedder

    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(pipeline.EMBEDDING_MODEL)

    vectors = _embedder.encode(
        target + descriptions,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    sem_tc = np.asarray(
        vectors[:1] @ vectors[1:].T,
        dtype=np.float32,
    )

    sem_cc = np.asarray(
        vectors[1:] @ vectors[1:].T,
        dtype=np.float32,
    ).reshape(1, K, K)

    tgt_clean = [target[0]] * K
    cand_clean = descriptions

    features, _ = pipeline.build_features(
        frame, sem_tc, sem_cc, tgt_clean, cand_clean
    )

    probabilities = model.predict_proba(
        features.to_numpy(dtype=np.float32)
    )[:, 1]

    # Existing Experiment 7 selection policy.
    best_alt = int(np.argmax(probabilities[1:]) + 1)
    gap = float(probabilities[best_alt] - probabilities[0])
    selected = best_alt if gap > 0.03 else 0

    candidate_row = frame.iloc[selected]

    return {
        "query_id": request.query.query_id,
        "decision": "HISTORICAL_ACCEPTED",
        "selected_candidate_rank": int(
            candidate_row["candidate_rank"]
        ),
        "selected_candidate_unit_price": float(
            candidate_row["candidate_unit_price"]
        ),
        "candidate_scores": [
            {
                "candidate_rank": i + 1,
                "relevance_probability": round(float(p), 6),
            }
            for i, p in enumerate(probabilities)
        ],
        "reason": (
            "Selected from the supplied candidates using "
            "the learned rank-1-aware policy."
        ),
    }


def _process(job_id: str, request: RerankRequest) -> None:
    with _lock:
        _jobs[job_id]["status"] = "processing"
        _jobs[job_id]["updated_at"] = (
            datetime.now(timezone.utc).isoformat()
        )

    try:
        result = _score(request)

        with _lock:
            _jobs[job_id].update(
                status="completed",
                result=result,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )

    except Exception as exc:
        with _lock:
            _jobs[job_id].update(
                status="failed",
                error=str(exc),
                updated_at=datetime.now(timezone.utc).isoformat(),
            )


# -------------------------------------------------------------------
# NEW CURRENT-MARKET EVIDENCE ENDPOINT
# -------------------------------------------------------------------

@app.post("/v1/market-benchmark")
def market_benchmark(
    request: MarketBenchmarkRequest,
) -> dict[str, Any]:
    """
    Find market evidence matching the item, currency, and unit.

    This endpoint does not alter historical candidate ranking.
    A match is not automatically proof of a verified market price.
    """

    currency = request.currency.strip().upper()
    unit = request.unit.strip()

    if not currency.isalpha():
        raise HTTPException(
            status_code=422,
            detail="Currency must be a three-letter currency code.",
        )

    if not request.item_name.strip():
        raise HTTPException(
            status_code=422,
            detail="Item name cannot be empty.",
        )

    try:
        matches = search_market(
            item_name=request.item_name.strip(),
            max_results=50,
            currency=currency,
            unit=unit,
            min_score=0.35,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Market evidence catalog is unavailable.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Market evidence lookup failed: {exc}",
        ) from exc

    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                "No compatible market evidence found for the "
                "requested item, currency, and unit."
            ),
        )

    # Generic token matches are not enough for an automated benchmark.
    exact = matches[
        matches["match_confidence"] == "review_exact_model"
    ].copy()

    if exact.empty:
        raise HTTPException(
            status_code=422,
            detail=(
                "Only generic product matches were found. "
                "Manual verification is required."
            ),
        )

    exact["observed_price"] = pd.to_numeric(
        exact["observed_price"],
        errors="coerce",
    )

    exact = exact.dropna(
        subset=["observed_price", "source_name"]
    )

    exact = exact[
        (exact["observed_price"] > 0)
        & (exact["currency"].astype(str).str.upper() == currency)
    ]

    if exact.empty:
        raise HTTPException(
            status_code=422,
            detail="No valid positive prices in the requested currency.",
        )

    # Deduplicate repeated records from the same named source.
    exact["source_name"] = exact["source_name"].astype(str).str.strip()
    exact["source_url"] = exact["source_url"].fillna("").astype(str).str.strip()

    exact = exact[
        (exact["source_name"] != "")
        & (exact["source_name"].str.lower() != "nan")
    ]

    independent = exact.drop_duplicates(
        subset=["source_name"],
        keep="first",
    )

    # Conservative starting threshold; this is not a guarantee
    # that the prices or sources have been independently verified.
    if independent["source_name"].nunique() < 3:
        raise HTTPException(
            status_code=422,
            detail=(
                "Insufficient independent named sources to create "
                "a market benchmark. Manual review is required."
            ),
        )

    prices = independent["observed_price"].astype(float)

    return {
        "status": "MARKET_BENCHMARK_AVAILABLE",
        "benchmarkUnitPrice": float(prices.median()),
        "expectedRange": {
            "lower": float(prices.min()),
            "upper": float(prices.max()),
        },
        "reliability": (
            "Provisional market evidence; exact-model identifier match; "
            "at least 3 distinct named sources"
        ),
        "currency": currency,
        "unit": unit,
        "evidenceCount": int(len(independent)),
        "sources": independent[
            [
                "item_name",
                "observed_price",
                "currency",
                "unit",
                "source_name",
                "source_url",
                "date_collected",
                "match_score",
                "match_reason",
            ]
        ].to_dict(orient="records"),
    }


# -------------------------------------------------------------------
# EXISTING API ROUTES
# -------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "online",
        "service": "learned-semantic-reranker",
    }


@app.post("/v1/rerank", status_code=202)
def submit_rerank(
    request: RerankRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    ranks = sorted(
        candidate.candidate_rank
        for candidate in request.candidates
    )

    if ranks != list(range(1, K + 1)):
        raise HTTPException(
            status_code=422,
            detail=f"Candidates must contain each rank 1..{K} exactly once",
        )

    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "query_id": request.query.query_id,
            "status": "pending",
            "created_at": now,
            "updated_at": now,
        }

    background_tasks.add_task(_process, job_id, request)

    return {
        "job_id": job_id,
        "query_id": request.query.query_id,
        "status": "pending",
        "result_url": f"/v1/rerank/{job_id}",
    }


@app.get("/v1/rerank/{job_id}")
def get_rerank_result(job_id: str) -> dict[str, Any]:
    with _lock:
        job = _jobs.get(job_id)

        if job is None:
            raise HTTPException(
                status_code=404,
                detail="Unknown job_id",
            )

        return dict(job)
