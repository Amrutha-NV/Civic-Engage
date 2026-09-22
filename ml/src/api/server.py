"""
CivicEngage Procurement Benchmark ML -- REST API Server

FastAPI service wrapping the frozen T3 Co-Occurrence/Bundle Ranker with:
- Conformal Prediction price range
- Transparent reliability classification (HIGH / MEDIUM / LOW)
- USD -> INR currency presentation layer (configurable via environment)
- Strict causal anti-leakage invariants
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
#  Bootstrap paths for ML imports                                              #
# --------------------------------------------------------------------------- #
_THIS_FILE = Path(__file__).resolve()
ML_ROOT = _THIS_FILE.parent.parent.parent  # D:\tender_generation\ML
sys.path.insert(0, str(ML_ROOT))
sys.path.insert(0, str(ML_ROOT / "src" / "inference"))

from src.inference.predict import ProcurementPredictor  # noqa: E402

# --------------------------------------------------------------------------- #
#  Currency Configuration (USD -> INR Presentation Layer)                      #
# --------------------------------------------------------------------------- #
# The underlying historical procurement dataset from Austin is in USD.
# For India-facing CivicEngage deployment, the API converts output prices to INR.
#
# Configuration:
# - USD_TO_INR_RATE environment variable can override this value at deployment time.
# - Default development fallback: 83.50 (median RBI reference rate during the
#   2023-2024 procurement evaluation baseline).
# - This is strictly an API presentation-layer configuration, NOT a model constant.
DEFAULT_USD_TO_INR_RATE = 83.50


def get_usd_to_inr_rate() -> float:
    """Read exchange rate from environment variable or return default development rate."""
    env_val = os.getenv("USD_TO_INR_RATE")
    if env_val:
        try:
            rate = float(env_val)
            if rate > 0:
                return rate
        except ValueError:
            pass
    return DEFAULT_USD_TO_INR_RATE


# --------------------------------------------------------------------------- #
#  Global Predictor Instance (Initialized ONCE on startup)                     #
# --------------------------------------------------------------------------- #
predictor: Optional[ProcurementPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize ProcurementPredictor once when the server starts."""
    global predictor
    print("Initializing ProcurementPredictor for API service...", flush=True)
    predictor = ProcurementPredictor(verbose=True)
    print(f"[OK] ProcurementPredictor ready. USD->INR rate: {get_usd_to_inr_rate():.2f}", flush=True)
    yield
    print("Shutting down Procurement ML Service...", flush=True)
    predictor = None


# --------------------------------------------------------------------------- #
#  FastAPI Application                                                         #
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="CivicEngage Procurement Benchmark ML Service",
    description="Inference API for the frozen T3 LightGBM ranker with conformal prediction and reliability scoring.",
    version="1.0.0",
    lifespan=lifespan,
)

# Optional CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
#  Request & Response Models (Pydantic)                                        #
# --------------------------------------------------------------------------- #
class ProcurementQueryRequest(BaseModel):
    """
    Basic request validation for procurement queries.
    Preserves all extra line-item fields without redesigning predictor schema.
    Exchange rates cannot be supplied by clients; only server-side configuration is used.
    """
    ITEM_DESCRIPTION: Optional[str] = None
    COMMODITY_DESCRIPTION: Optional[str] = None
    EXTENDED_DESCRIPTION: Optional[str] = None
    product_text_normalized: Optional[str] = None
    COMMODITY: Optional[str] = None
    commodity_code: Optional[str] = None
    commodity_family: Optional[str] = None
    UNIT_OF_MEASURE: Optional[str] = "EA"
    uom_standardized: Optional[str] = "EA"
    quantity_numeric: Optional[float] = 1.0
    award_date_parsed: Optional[str] = "2024-06-15"
    PURCHASE_ORDER: Optional[str] = "PO-API-REQ"
    VENDOR_CODE: Optional[str] = "VENDOR-UNKNOWN"
    BRAND_NAME: Optional[str] = None
    MODEL_NUMBER: Optional[str] = None
    PART_NUMBER: Optional[str] = None
    MASTER_AGREEMENT: Optional[str] = None
    PRODUCT_TYPE: Optional[str] = None

    model_config = {"extra": "allow"}


# --------------------------------------------------------------------------- #
#  Endpoints                                                                   #
# --------------------------------------------------------------------------- #
@app.get("/health")
def health_check():
    """Service health check endpoint."""
    return {
        "status": "ok",
        "service": "procurement-benchmark-ml"
    }


@app.post("/predict")
def predict_benchmark(request: ProcurementQueryRequest):
    """
    Predict procurement benchmark unit price, conformal expected range,
    and reliability classification with explicit USD -> INR presentation conversion.
    """
    if predictor is None:
        raise HTTPException(
            status_code=500,
            detail="Predictor model is not initialized or still warming up."
        )

    query_dict = request.model_dump()

    # Ensure descriptions are cross-populated if one is provided
    if not query_dict.get("COMMODITY_DESCRIPTION") and query_dict.get("ITEM_DESCRIPTION"):
        query_dict["COMMODITY_DESCRIPTION"] = query_dict["ITEM_DESCRIPTION"]
    if not query_dict.get("ITEM_DESCRIPTION") and query_dict.get("COMMODITY_DESCRIPTION"):
        query_dict["ITEM_DESCRIPTION"] = query_dict["COMMODITY_DESCRIPTION"]

    try:
        result = predictor.predict(query_dict)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    # ----------------------------------------------------------------------- #
    #  USD -> INR Presentation Conversion (API Boundary Only)                 #
    # ----------------------------------------------------------------------- #
    rate = get_usd_to_inr_rate()

    usd_price = result.get("benchmarkUnitPrice")
    usd_lower = result.get("lowerBound")
    usd_upper = result.get("upperBound")

    inr_price = round(float(usd_price * rate), 2) if usd_price is not None else None
    inr_lower = round(float(usd_lower * rate), 2) if usd_lower is not None else None
    inr_upper = round(float(usd_upper * rate), 2) if usd_upper is not None else None

    if result.get("expectedRange") is not None:
        expected_range_inr = {
            "lower": inr_lower,
            "upper": inr_upper,
            "coverageLevel": float(result["expectedRange"].get("coverageLevel", 0.80)),
            "method": str(result["expectedRange"].get("method", "multiplicative_log_ratio_split_conformal")),
        }
    else:
        expected_range_inr = None

    raw_usd = {
        "benchmarkUnitPrice": usd_price,
        "lowerBound": usd_lower,
        "upperBound": usd_upper,
    }

    return {
        "currency": "INR",
        "exchangeRate": rate,
        "benchmarkUnitPrice": inr_price,
        "expectedRange": expected_range_inr,
        "lowerBound": inr_lower,
        "upperBound": inr_upper,
        "reliability": result.get("reliability"),
        "rawUSD": raw_usd,
        "n_candidates": result.get("n_candidates"),
        "rank1_score": result.get("rank1_score"),
        "retrieval_time_s": result.get("retrieval_time_s"),
        "feature_time_s": result.get("feature_time_s"),
        "ranking_time_s": result.get("ranking_time_s"),
        "total_time_s": result.get("total_time_s"),
        "inference_time_s": result.get("total_time_s"),
    }


# --------------------------------------------------------------------------- #
#  CLI entry point for running server directly                                #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
