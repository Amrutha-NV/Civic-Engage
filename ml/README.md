# CivicEngage — Procurement Benchmark ML Service
## Overview
The Procurement Benchmark ML system is the machine-learning component of CivicEngage responsible for estimating a procurement benc
The system:
1. Receives a procurement requirement.
2. Normalizes the commercial unit and procurement description.
3. Enriches the requirement with product/specification information.
4. Retrieves comparable historical procurement candidates.
5. Builds ranking and contextual features.
6. Ranks candidates using a LightGBM LambdaRank model.
7. Estimates the benchmark unit price.
8. Produces an uncertainty range using split conformal calibration.
9. Assigns a deterministic reliability level.
10. Exposes the result through a FastAPI service.
The final deployment package is self-contained under `ml/`.
---
## What the ML Service Outputs
The prediction response provides:
- **Benchmark Unit Price** — estimated procurement benchmark price per unit.
- **Expected Range** — uncertainty interval around the benchmark.
- **Reliability** — deterministic `HIGH`, `MEDIUM`, or `LOW` evidence-quality level.
- **Candidate Count** — number of historical candidates retrieved.
- **Ranking Score** — score of the top-ranked candidate.
- **Inference Timings** — retrieval, feature, ranking, and total inference time.
- **Currency Information** — INR presentation and exchange-rate configuration.
- **Raw USD Values** — underlying model-side price values.
---
## Architecture
```text
Procurement Request
|
v
Normalization / Enrichment
|
v
Multi-view Candidate Retrieval
|
v
Feature Engineering
|
v
LightGBM LambdaRank
|
v
Benchmark Estimation
|
v
Split Conformal Calibration
|
v
Reliability Gate
|
v
FastAPI Response
```
---
## Final ML Algorithm
### Model
The final model is a **LightGBM LambdaRank learning-to-rank model**.
### Final Configuration
| Parameter | Value |
|---|---:|
| Features | 176 |
| Trees | 300 |
| Objective | `lambdarank` |
| Metric | `ndcg` |
| Learning rate | 0.04 |
| Number of leaves | 31 |
| Random seed | 42 |
| Training query groups | 7,997 |
| Candidate pairs | 304,621 |
| Hard-negative setting | HN5 graded relevance |
---
## Candidate Retrieval and Feature Engineering
### Commercial Unit Normalization
Procurement descriptions are parsed and normalized so that unit-price comparisons are made on comparable commercial units.
### Specification Enrichment
Product identity and specification signals are extracted from procurement descriptions.
### Entity Resolution
Historical procurement records are mapped to normalized product/entity identities where supported by the stored entity matcher.
### Multi-view Candidate Retrieval
Multiple retrieval views are used to construct the historical candidate pool instead of depending only on exact-text matching.
### Quantity and Temporal Context
Quantity and procurement date information contribute to candidate relevance and price-state features.
### Scope and Context
Scope and configuration information helps distinguish materially different procurement requirements.
### Price-state Features
Historical price behavior and temporal state features provide additional ranking context.
---
## Benchmark Estimation
At inference time, the model scores the retrieved historical candidate pool.
The selected high-ranked evidence is used by the inference pipeline to derive the benchmark unit price.
Candidate support information is retained for the reliability layer.
---
## Uncertainty — Split Conformal Calibration
The system uses **multiplicative log-ratio split conformal calibration**.
### Configuration
- `q_hat = 1.5404`
- Nominal coverage level = `0.80`
The interval is calculated as:
```text
lower = prediction × exp(-q_hat)
upper = prediction × exp(q_hat)
```
This produces an uncertainty range around the predicted benchmark.
---
## Reliability
Reliability is a **deterministic heuristic**, not a calibrated probability.
### HIGH
```text
Pool support >= 5
AND structured match / consensus
AND sigma <= 0.35
AND nonnegative margin
AND recency <= 1095 days
```
### LOW
```text
N < 3
OR sigma > 0.85
OR coarse fallback only
```
### MEDIUM
Cases that do not satisfy either the HIGH or LOW conditions.
---
## Final Frozen Evaluation
The final frozen test set was accessed once and then treated as closed.
The original project target was greater than 65% accuracy within ±10%. The final frozen result did not achieve that target. The re
| Metric | Final Frozen Result |
|---|---:|
| Test N | 6,755 |
| Accuracy within ±10% | 36.58% |
| Accuracy within ±5% | 29.95% |
| Accuracy within ±20% | 44.47% |
| Top-3 | 48.73% |
| Top-5 | 54.39% |
| Top-10 | 63.97% |
| MAE | $5,732.05 |
| Median APE | 30.00% |
| Log RMSE | 1.4422 |
| High-dispersion ±10% | 29.68% (N=5,287) |
| Candidate-oracle ±10% | 85.73% |
| Mean candidates | 56.16 |
| Failures | 0 |
---
## Model and Data Artifacts
The final runtime package contains:
```text
models/
nnn procurement_ranker.joblib
nnn procurement_ranker_metadata.joblib
nnn conformal_calibration.joblib
nnn entity_matcher.joblib
data/catalog/
nnn procurement_catalog.parquet
nnn normalized_identity_cache.parquet
nnn parsed_records_cache.joblib
```
These files are required by the final inference pipeline.
---
## Project Structure
```text
ml/
nnn README.md
nnn requirements.txt
nnn .gitignore
nnn ML_FINAL_FREEZE.md
n
nnn models/
n nnn procurement_ranker.joblib
n nnn procurement_ranker_metadata.joblib
n nnn conformal_calibration.joblib
n nnn entity_matcher.joblib
n
nnn data/
n nnn catalog/
n nnn procurement_catalog.parquet
n nnn normalized_identity_cache.parquet
n nnn parsed_records_cache.joblib
n
nnn src/
n nnn api/
n n nnn server.py
n nnn inference/
n n nnn predict.py
n nnn calibration/
n n nnn build_conformal_calibration.py
n nnn ranking/
n nnn normalization/
n nnn price_state/
n nnn retrieval/
n nnn entity_resolution/
n nnn spec_enrichment/
n nnn confidence/
n
nnn docs/
nnn final_frozen_test_report.md
nnn final_frozen_test_results.json
```
---
## Local Setup
### Requirements
- Python 3.11
- Windows / Linux / macOS
- The files in the final `ml/` package
Open a terminal inside:
```text
CivicEngage/ml/
```
### Create Virtual Environment
```powershell
py -3.11 -m venv .venv
```
### Activate Virtual Environment
Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```
### Upgrade pip
```powershell
python -m pip install --upgrade pip
```
### Install Dependencies
```powershell
pip install -r requirements.txt
```
---
## Run the Predictor Directly
Example:
```powershell
python -c "from src.inference.predict import predict; print(predict({'description':'Dell Latitude laptop, Intel Core i5, 16GB RAM,
```
---
## Run FastAPI
Start the API using:
```powershell
python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000
```
The service will be available locally at:
```text
http://127.0.0.1:8000
```
### Health Check
```text
GET /health
```
### Interactive API Documentation
```text
http://127.0.0.1:8000/docs
```
### Prediction Endpoint
```text
POST /predict
```
---
## API Request
The prediction request requires at least:
- `description`
- `commodity_code`
Additional fields can also be provided.
Example:
```json
{
"description": "Dell Latitude laptop, Intel Core i5, 16GB RAM, 512GB SSD",
"commodity_code": "OTHER_GOODS",
"uom": "EA",
"quantity": 10,
"award_date": "2024-06-15"
}
```
### Default Values
If optional values are omitted, the API uses:
```text
UOM = EA
quantity = 1
award_date = 2024-06-15
PO number = PO-API-REQ
vendor = unknown
```
---
## API Response
The response contains fields such as:
```json
{
"currency": "INR",
"exchangeRate": 83.5,
"benchmarkUnitPrice": 71625.46,
"expectedRange": {
"lower": 15348.97,
"upper": 334237.14,
"coverageLevel": 0.8,
"method": "multiplicative_log_ratio_split_conformal"
},
"lowerBound": 15348.97,
"upperBound": 334237.14,
"reliability": "LOW",
"rawUSD": {
"benchmarkUnitPrice": 857.79,
"lowerBound": 183.82,
"upperBound": 4002.84
},
"n_candidates": 60
}
```
The exact values depend on the input request.
---
## Date Guard
The deployed API rejects prediction dates on or after:
```text
2025-01-01
```
This guard exists because the final package is aligned with the frozen evaluation/training boundary.
Do not remove this guard without explicitly reviewing the model's temporal assumptions and retraining/evaluation process.
---
## INR Presentation
The API presents INR values using:
```text
USD_TO_INR_RATE
```
The default configuration value is:
```text
83.50
```
This is a deployment/presentation configuration value. It is not claimed to be an official or scientific exchange-rate source.
The value can be configured through an environment variable.
Example:
```powershell
$env:USD_TO_INR_RATE="84.00"
```
---
## Backend Integration
The ML service exposes a FastAPI contract.
The Node.js backend can call:
```text
POST /predict
```
and use the returned:
- benchmark unit price
- expected range
- lower bound
- upper bound
- reliability
- candidate count
- supporting metadata
The ML component does **not** own:
- Node.js backend
- React frontend
- authentication
- tender generation
- tender PDF generation
- NGO dashboard
- application-level workflow
Those belong to the application/backend side.
---
## Research Workspace vs Deployment Package
The research workspace contains:
- raw datasets
- experiments
- notebooks
- training matrices
- historical model versions
- analysis scripts
- temporary caches
- evaluation harnesses
These are intentionally excluded from the CivicEngage deployment package.
The `ml/` directory contains only the final runtime artifacts and documentation required for deployment and review.
---
## Verification Status
The final CivicEngage ML package was verified locally.
Verified:
- Fresh `ml/.venv` creation
- Dependency installation
- Direct predictor execution
- FastAPI import
- `GET /health` → HTTP 200
- `POST /predict` → HTTP 200
- Package-local artifact paths
- Self-contained runtime
- No raw dataset included
- No training matrices included
- No frozen-test harness included
Example successful local prediction:
```text
Benchmark: n71,625.46
Range: n15,348.97 – n334,237.14
Reliability: LOW
Candidates: 60
Inference time: ~0.127 seconds
```
---
## Ownership
### ML Component
Responsible for:
- Procurement benchmark model
- Candidate retrieval
- Feature engineering
- LightGBM ranking
- Benchmark estimation
- Conformal calibration
- Reliability logic
- FastAPI endpoints
- INR presentation layer
### Application / Backend
Responsible for:
- Node.js backend
- API integration
- React frontend
- Tender generation
- Tender display
- PDF generation
- Authentication
- Application workflow
- NGO dashboard
---
## Important Notes
1. The final frozen evaluation must not be rerun for tuning.
2. The frozen test set was accessed once.
3. The reported metrics are the actual final results.
4. The >65% ±10% target was not achieved.
5. No metric manipulation or synthetic evaluation evidence is used.
6. The final deployment package is separate from the research workspace.
7. Model artifacts and catalog files are intentionally committed because they are required for runtime inference.
8. `.venv`, caches, raw datasets, logs, and temporary files should remain excluded through `.gitignore`.