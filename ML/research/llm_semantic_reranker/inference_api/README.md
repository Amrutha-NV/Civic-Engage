# Learned reranker inference API

This API accepts a query and the upstream model's Top-10 candidates, applies the Experiment 7 learned reranker, and provides the selected candidate through a job result endpoint. The selected unit price is copied exactly from that candidate's `candidate_unit_price`.

## Start the app

Open PowerShell at the repository root (`E:\MajorProject\CivicEngage-final`) and run:

```powershell
.\.venv\Scripts\python.exe -m uvicorn ML.research.llm_semantic_reranker.inference_api.app:app --host 127.0.0.1 --port 8011
```

Interactive API docs are at `http://127.0.0.1:8011/docs`. The model and frozen labeled dataset must be available in the repository. The first submitted request fits and caches the serving classifier and may take several minutes. Install dependencies if needed:

```powershell
.\.venv\Scripts\python.exe -m pip install -r ML\research\llm_semantic_reranker\inference_api\requirements.txt
```

Run one Uvicorn worker: jobs and results currently live in process memory and are lost when the app restarts.

## Endpoints

| Method | URL | Purpose |
|---|---|---|
| `GET` | `http://127.0.0.1:8011/health` | Check that the API is online. |
| `POST` | `http://127.0.0.1:8011/v1/rerank` | Submit one query and exactly ten candidates. Returns HTTP `202` with a job ID. |
| `GET` | `http://127.0.0.1:8011/v1/rerank/{job_id}` | Poll the submitted job and retrieve its result. |

## Submit a Top-10 request

Send `Content-Type: application/json` to `POST http://127.0.0.1:8011/v1/rerank`. This complete example includes all ten required candidates:

```json
{
  "query": {
    "query_id": "upstream-req-8421",
    "target_description": "Dell Latitude 5530 laptop, 16 GB RAM, 512 GB SSD",
    "target_quantity": 25,
    "target_unit_of_measure": "EA",
    "target_brand": "Dell",
    "target_model": "Latitude 5530",
    "target_commodity_code": "20453",
    "target_commodity_family": "COMPUTER_HARDWARE"
  },
  "candidates": [
    {"candidate_rank": 1, "candidate_description": "Dell Latitude laptop, 16 GB RAM, 512 GB SSD", "candidate_unit_price": 1189.5, "candidate_score": 1.482019},
    {"candidate_rank": 2, "candidate_description": "Dell Latitude laptop, 8 GB RAM, 256 GB SSD", "candidate_unit_price": 999.0, "candidate_score": 1.321},
    {"candidate_rank": 3, "candidate_description": "Dell Latitude laptop, 16 GB RAM, 256 GB SSD", "candidate_unit_price": 1099.0, "candidate_score": 1.217},
    {"candidate_rank": 4, "candidate_description": "Dell Inspiron laptop, 16 GB RAM, 512 GB SSD", "candidate_unit_price": 899.0, "candidate_score": 1.106},
    {"candidate_rank": 5, "candidate_description": "Dell Latitude laptop, 32 GB RAM, 1 TB SSD", "candidate_unit_price": 1399.0, "candidate_score": 1.044},
    {"candidate_rank": 6, "candidate_description": "Dell Latitude laptop, 16 GB RAM, 1 TB SSD", "candidate_unit_price": 1299.0, "candidate_score": 0.982},
    {"candidate_rank": 7, "candidate_description": "Dell Latitude laptop, 8 GB RAM, 512 GB SSD", "candidate_unit_price": 949.0, "candidate_score": 0.901},
    {"candidate_rank": 8, "candidate_description": "Dell Precision laptop, 16 GB RAM, 512 GB SSD", "candidate_unit_price": 1599.0, "candidate_score": 0.842},
    {"candidate_rank": 9, "candidate_description": "Dell Latitude laptop, 16 GB RAM, 256 GB SSD", "candidate_unit_price": 1049.0, "candidate_score": 0.791},
    {"candidate_rank": 10, "candidate_description": "Dell Latitude laptop, 8 GB RAM, 128 GB SSD", "candidate_unit_price": 799.0, "candidate_score": 0.702}
  ]
}
```

`query_id` and `target_description` are required. The other shown target fields may be omitted or set to `null`. Every candidate must include all four fields shown, and ranks 1 through 10 must each appear exactly once. `candidate_unit_price` must be nonnegative; `candidate_score` must be numeric. Extra fields are rejected. Send the upstream values unchanged.

PowerShell example (save the JSON above as `request.json` in the current folder):

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8011/v1/rerank -ContentType 'application/json' -InFile .\request.json
```

The immediate response contains the `job_id` to use in the result URL:

```json
{
  "job_id": "e353d010-c295-4c29-8d3a-c90e3ff4057a",
  "query_id": "upstream-req-8421",
  "status": "pending",
  "result_url": "/v1/rerank/e353d010-c295-4c29-8d3a-c90e3ff4057a"
}
```

## Retrieve the result

Poll `GET http://127.0.0.1:8011/v1/rerank/{job_id}` until `status` is `completed` or `failed`. For example:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8011/v1/rerank/e353d010-c295-4c29-8d3a-c90e3ff4057a
```

While work is running, the response has `status: "pending"` or `"processing"`. A completed response has this shape; `candidate_scores` always contains ten rank-ordered entries:

```json
{
  "job_id": "e353d010-c295-4c29-8d3a-c90e3ff4057a",
  "query_id": "upstream-req-8421",
  "status": "completed",
  "result": {
    "query_id": "upstream-req-8421",
    "decision": "HISTORICAL_ACCEPTED",
    "selected_candidate_rank": 2,
    "selected_candidate_unit_price": 999.0,
    "candidate_scores": [
      {"candidate_rank": 1, "relevance_probability": 0.51},
      {"candidate_rank": 2, "relevance_probability": 0.62},
      {"candidate_rank": 3, "relevance_probability": 0.48},
      {"candidate_rank": 4, "relevance_probability": 0.42},
      {"candidate_rank": 5, "relevance_probability": 0.39},
      {"candidate_rank": 6, "relevance_probability": 0.36},
      {"candidate_rank": 7, "relevance_probability": 0.34},
      {"candidate_rank": 8, "relevance_probability": 0.31},
      {"candidate_rank": 9, "relevance_probability": 0.29},
      {"candidate_rank": 10, "relevance_probability": 0.25}
    ],
    "reason": "Selected from the supplied candidates using the learned rank-1-aware policy."
  }
}
```

The probabilities above illustrate the response shape. Actual values are computed for the submitted request. The returned selected price is the exact input price associated with `selected_candidate_rank`; the model does not generate or adjust prices.

An unknown job ID returns HTTP `404`. Invalid JSON or candidate ranks return HTTP `422`. Processing failures return `status: "failed"` and an `error` string. Check `/docs` for the interactive schema and endpoint details.
