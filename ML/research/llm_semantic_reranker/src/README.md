# Semantic reranker experiments

This folder contains the implementations for the seven reranking experiments. Experiment 0 is the Rank-1/oracle baseline used as a comparison. All experiments operate on the supplied frozen Top-10 pool; they do not change retrieval or generate candidate prices.

Run commands below from the repository root using the project virtual environment.

## Experiment 0 — Rank-1 baseline and Oracle@10

Measures the original Rank-1 selection and the best possible selection available in the Top-10 pool. The oracle is an evaluation upper bound, not a runtime selector.

```powershell
.\.venv\Scripts\python.exe ML\research\llm_semantic_reranker\src\baseline.py
```

## Experiment 1 — Embedding cosine

Embeds target and candidate descriptions with `sentence-transformers/all-MiniLM-L6-v2`; selects the candidate with the highest cosine similarity.

```powershell
.\.venv\Scripts\python.exe ML\research\llm_semantic_reranker\src\embedding_reranker.py
```

## Experiment 2 — Structured attribute matching

Scores literal target brand, model, and commodity-code matches in candidate descriptions. It does not fabricate candidate metadata that is absent from the source data.

```powershell
.\.venv\Scripts\python.exe ML\research\llm_semantic_reranker\src\attribute_scorer.py
```

## Experiment 3 — Hybrid semantic and attribute reranker

Combines normalized embedding cosine similarity and structured attribute evidence. The default semantic weight is 0.5; `--semantic-weight` supports controlled ablations.

```powershell
.\.venv\Scripts\python.exe ML\research\llm_semantic_reranker\src\hybrid_reranker.py --semantic-weight 0.5
```

## Experiment 4 — Cross-encoder

Scores each target/candidate description pair with `cross-encoder/ms-marco-MiniLM-L-6-v2`.

```powershell
.\.venv\Scripts\python.exe ML\research\llm_semantic_reranker\src\cross_encoder.py
```

## Experiment 5 — LLM and multi-aspect variants

The closed-context LLM selector validates a rank-only response and resolves the selected price from the supplied historical candidate. API calls are bounded with `--max-queries` and require the configured OpenRouter credentials/model in `.env`.

```powershell
.\.venv\Scripts\python.exe ML\research\llm_semantic_reranker\src\llm_reranker.py --max-queries 10
```

The related multi-aspect implementation and comparative 5A implementation are available as `multi_aspect_reranker.py` and `llm_reranker_5a.py`.

## Experiment 6 — Conservative Rank-1 override

Combines semantic, structured, retrieval, and historical price-pool evidence with a conservative override gate. The selected price remains the exact price of the selected candidate.

```powershell
.\.venv\Scripts\python.exe ML\research\llm_semantic_reranker\src\rank1_override_reranker.py
```

## Experiment 7 — Learned Rank-1-aware reranker (selected pipeline)

Builds 77 query/candidate features and evaluates LightGBM binary and ranking models with grouped cross-validation. Its primary policy is the binary classifier with a nested-CV tuned gate. Evaluation labels are used for training/evaluation only, not as features.

```powershell
.\.venv\Scripts\python.exe ML\research\llm_semantic_reranker\src\learned_reranker.py
```

Useful options include `--debug 300`, `--resume`, `--embedder tfidf`, `--folds`, and `--no-gate`. Outputs are written under `results/learned_reranker_exp7/`; see `../inference_api/README.md` for online Top-10 inference.

## Shared implementation details

- `validator.py` contains validation helpers used by the research pipelines.
- `evaluation/` contains metric helpers.
- Results and experiment summaries are stored under the sibling `results/` directory.
- No reranker may invent, modify, average, or interpolate a candidate's unit price.
