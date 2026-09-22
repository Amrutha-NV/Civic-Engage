# CivicEngage Procurement Benchmark ML

## Purpose
Historical procurement benchmark unit-price estimation using retrieval and a frozen LightGBM LambdaRank ranker.

## Architecture & Pipeline
The prediction pipeline processes incoming procurement line item queries through the following stages:

```
Input Line Item
  → Candidate Retrieval (multi-view matching across catalog)
  → Feature Extraction (temporal, scope, commercial units, price state)
  → LambdaRank Ranking (frozen LightGBM ranker)
  → Historical UNIT_PRICE Benchmark Selection
  → Conformal Calibration (expected range calculation)
  → Reliability Classification
  → FastAPI Response Formatting
```

## API Endpoints
- `GET /health`: Health check endpoint reporting service status, model version, and artifact availability.
- `POST /predict`: Benchmark unit-price estimation endpoint for procurement line items.

## Currency Handling
The underlying machine learning model and historical training data operate strictly in USD. Currency conversion to INR is performed entirely at the presentation layer using the configured exchange rate (`USD_TO_INR_RATE`). The underlying model weights, ranking, and benchmark logic remain unchanged.

## Empirical Performance & Validation
- **Target Performance**: >65% within ±10% of ground truth
- **Final Development Validation**: 37.57% within ±10%
- **Final Frozen Test**: 36.58% within ±10% (N = 6,755)

**Explicit Evaluation Statement**:
The target performance threshold of >65% within ±10% was **not achieved**. The system operates as a historical benchmark reference and should not be treated as meeting the high-precision target without further structural data enrichment.
