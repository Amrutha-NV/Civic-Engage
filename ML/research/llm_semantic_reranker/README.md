# CivicEngage Semantic Reranker Research

This isolated research module evaluates a second-stage selector over the frozen
Top-10 procurement candidates. It never modifies the production ranker, the
retrieval pool, or candidate prices.

## Implemented: Experiment 0

Run the Rank-1 baseline and Oracle@10 analysis from the repository root:

```powershell
.\.venv\Scripts\python.exe ML\research\llm_semantic_reranker\src\baseline.py
```

The command reads the immutable source artifact in
`ML/research/procurement_benchmark_topk/` and writes these reproducible outputs
to `results/`:

- `baseline_metrics.json`
- `baseline_ledger.parquet`
- `baseline_ledger.csv`

Evaluation-only fields are read only after Rank-1 is fixed; no reranking input
or selection depends on actual prices or hit indicators.

## Implemented: Experiment 1

The cosine reranker embeds only `target_description` and
`candidate_description` with the CPU-compatible
`sentence-transformers/all-MiniLM-L6-v2` model. It caches duplicate text within
the run, selects the highest cosine-similarity candidate, and resolves that
candidate's exact historical price only after selection.

```powershell
.\.venv\Scripts\python.exe ML\research\llm_semantic_reranker\src\embedding_reranker.py
```

It writes `embedding_cosine_metrics.json` plus CSV and Parquet evaluation
ledgers to `results/`.

## Implemented: Experiment 2

The structured attribute scorer uses only literal target brand, model, and
commodity-code matches found in `candidate_description`. Candidate-side fields
that do not exist in the frozen artifact—such as quantity, UOM, location,
vendor, and date—are deliberately excluded.

```powershell
.\.venv\Scripts\python.exe ML\research\llm_semantic_reranker\src\attribute_scorer.py
```

## Implemented: Experiment 3

The hybrid reranker min-max normalises cosine similarity inside each Top-10
pool and blends it with literal structured-attribute evidence. Its default is
an even 0.5/0.5 blend; use `--semantic-weight` to run explicit ablations.

```powershell
.\.venv\Scripts\python.exe ML\research\llm_semantic_reranker\src\hybrid_reranker.py --semantic-weight 0.5
```

## Implemented: Experiment 4

The Cross-Encoder evaluates each target/candidate description pair directly
with the CPU-compatible `cross-encoder/ms-marco-MiniLM-L-6-v2` model. It does
not receive a candidate price, frozen score, rank, or an evaluation label.

```powershell
.\.venv\Scripts\python.exe ML\research\llm_semantic_reranker\src\cross_encoder.py
```

## Implemented: Experiment 5

The LLM reranker uses the existing root `.env` configuration: `GROQ_API_KEY`
and `GROQ_MODEL`. It sends only allowed target/candidate fields and validates
the model response against the exact three-field no-price contract before
resolving a selected historical price. API execution must name an explicit,
cost-controlled query limit:

```powershell
.\.venv\Scripts\python.exe ML\research\llm_semantic_reranker\src\llm_reranker.py --max-queries 10
```

## Research handoff API

Run the research-only FastAPI service from the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn ML.research.llm_semantic_reranker.api:app --host 127.0.0.1 --port 8010
```

- `GET /handoff/{query_index}?method=structured` returns one selected frozen
  historical candidate and its exact unit price for the next stage.
- `GET /research/summary` returns measured findings for completed experiments
  and marks unmeasured experiments as pending.
