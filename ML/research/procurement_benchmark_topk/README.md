# CivicEngage Procurement Benchmark — Top-K Analysis Package

This directory contains the frozen research artifacts, empirical findings, and interface specifications for the Top-K candidate analysis based on the frozen **Step-37 / T3 Co-Occurrence Model** (`N = 10,560` validation queries).

---

## Relationship Between Production and Research

### Production package

[`ML/procurement_benchmark_engine/`](../../procurement_benchmark_engine/)

The production package provides runtime inference through the frozen Procurement Benchmark engine/API.

### Research package

`ML/research/procurement_benchmark_topk/`

The research package evaluates the frozen Step-37/T3 ranking behavior and provides the frozen evaluation artifact for downstream LLM semantic reranking, current-market evidence experiments, and research evaluation.

> **Decoupling:** The research package is intentionally decoupled from the production package. The research scripts do not import the production package.

---

## Directory Contents

| File | Description |
|---|---|
| **[`frozen_evaluation_dataset.parquet`](./frozen_evaluation_dataset.parquet)** | **Frozen downstream evaluation artifact.** Contains 105,600 rows across 10,560 validation queries, with exactly 10 ranked candidates per query. Downstream researchers can use this artifact without the original 3.4 GB training dataset or Step-36 cache files. |
| **[`analyze_top_k.py`](./analyze_top_k.py)** | Offline research reproduction script for the frozen Step-37/T3 Top-K analysis. Reproducing the original ranking run requires the original development model/data/cache artifacts. Downstream experiments should use `frozen_evaluation_dataset.parquet`. |
| **[`top_10_candidates.csv`](./top_10_candidates.csv)** | Frozen Top-10 candidate ranking results for all 10,560 validation queries, including candidate descriptions, prices, ML scores, percentage errors, and ±10% indicators. |
| **[`top_k_metrics.json`](./top_k_metrics.json)** | Machine-readable benchmark summary containing Rank-1, Top-3, Top-5, and Top-10 accuracy at the ±10% threshold and related recovery statistics. |
| **[`recovery_rank_distribution.json`](./recovery_rank_distribution.json)** | Empirical distribution of the first valid ±10% candidate position across Ranks 2–10 for Rank-1 failure cases. |
| **[`evidence_gate_analysis.py`](./evidence_gate_analysis.py)** | Research analysis script for Step 1 Historical Evidence Gate. Evaluates inference-safe score strength and price dispersion signals against ground-truth recovery outcomes. |
| **[`evidence_gate_metrics.json`](./evidence_gate_metrics.json)** | Machine-readable empirical findings and distribution metrics for the Historical Evidence Gate, including ROC-AUC scores, quintile hit rates, and policy simulation results. |
| **[`evidence_gate_threshold_validation.json`](./evidence_gate_threshold_validation.json)** | Validation review metrics and sensitivity analysis around the balanced threshold 0.0 across the neighborhood [-0.2, +0.2]. |
| **[`EVIDENCE_GATE_FINDINGS.md`](./EVIDENCE_GATE_FINDINGS.md)** | Research report documenting signal taxonomy, empirical findings, and routing logic between Person 2 (LLM semantic reranker) and Person 3 (market evidence). |
| **[`HISTORICAL_EVIDENCE_GATE_CONTRACT.md`](./HISTORICAL_EVIDENCE_GATE_CONTRACT.md)** | Step 2 formal interface contract between the ML pipeline and Person 2 (LLM reranker), defining gate inputs, decision routing, output schemas, invalid output protocols, and failovers. |
| **[`TOP_K_FINDINGS.md`](./TOP_K_FINDINGS.md)** | Research summary containing the core Top-K evaluation results, failure recovery analysis, and justification for downstream semantic reranking. |
| **[`LLM_RERANKING_INPUT_SPEC.md`](./LLM_RERANKING_INPUT_SPEC.md)** | Interface contract for the downstream LLM semantic reranking experiment. Defines the available input fields, expected output, no-price-invention constraint, and Top-K evaluation setup. |

---

## Frozen Evaluation Artifact Details

The primary downstream deliverable is:

`frozen_evaluation_dataset.parquet`

- **Total rows:** 105,600
- **Validation queries:** 10,560
- **Validation period:** 2023–2024 temporal split
- **Candidates per query:** exactly 10
- **Candidate ranks:** 1 through 10

The artifact is designed so downstream researchers can evaluate their algorithms against the same frozen benchmark without requiring the original training dataset or Step-36 cache files.

### Available target and candidate information

The frozen artifact contains:

- Query identifier
- Target procurement description
- Target quantity
- Target unit of measure
- Target procurement date
- Target city and state
- Target brand and model where available
- Target commodity code and family
- Candidate rank
- Candidate description
- Candidate historical unit price
- Candidate ML score
- Actual ground-truth unit price
- Absolute percentage error
- ±10% evaluation indicator

### Candidate metadata not included

The current frozen artifact does **not** contain verified candidate-level:

- quantity
- unit of measure
- award date
- vendor name
- vendor location
- contract type
- extracted structured candidate specifications

These fields must not be fabricated or inferred as if they were present in the frozen artifact.

---

## Key Benchmark Highlights

- **Validation queries:** 10,560
- **Temporal split:** 2023–2024
- **Rank-1 accuracy (±10%):** 37.57% (3,967 queries)
- **Top-3 accuracy (±10%):** 48.49%
- **Top-5 accuracy (±10%):** 54.14% (5,717 queries)
- **Top-10 accuracy (±10%):** 62.64% (6,615 queries)
- **Rank-1 failure recovery within Top-10:** 2,648 queries

The Top-K analysis therefore provides a frozen candidate set for downstream semantic reranking and independent experimental evaluation.

---

## Important Research Constraint

The downstream LLM reranker must **select from the supplied historical candidates**.

It must not invent, estimate, modify, or hallucinate a historical candidate price.

If candidate `X` is selected, the resulting historical price must exactly equal:

`candidate_unit_price` of candidate `X`.

The frozen `actual_unit_price` is used only as ground truth during evaluation and must not be provided to the production reranking process.
