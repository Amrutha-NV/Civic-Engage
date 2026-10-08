# CivicEngage Procurement Benchmark — Top-30 Unrecoverable Query Analysis

This research report documents the empirical investigation into the **2,293 validation queries (21.71%)** that contain no valid candidate within $\pm 10\%$ of the actual price anywhere in the canonical Step-37 Top-30 candidate pool ($N = 10,560$ queries).

---

## 1. Executive Summary & Population Partition

| Population Group | Definition | Query Count | Pct of Population | Canonical Oracle Coverage |
| :--- | :--- | :---: | :---: | :---: |
| **Group A** | Valid $\pm 10\%$ candidate in Top-10 | **6,615** | **62.64%** | 62.64% (Canonical Baseline) |
| **Group B** | Valid candidate in Ranks 11–30 | **1,652** | **15.64%** | 78.29% (Top-30 Ceiling) |
| **Group C** | **No valid candidate anywhere in Top-30** | **2,293** | **21.71%** | — |
| ↳ **Subgroup C1** | Valid candidate exists in Ranks 31–60 | **719** | **6.81%** | 85.09% (Top-60 Ceiling) |
| ↳ **Subgroup C2** | **No valid candidate in entire Top-60 pool** | **1,574** | **14.91%** | 14.91% Unrecoverable |

---

## 2. Canonical Oracle Coverage Hierarchy

- **Top-10 Oracle Coverage:** **6,615 / 10,560** (62.64%)
- **Top-30 Oracle Coverage:** **8,267 / 10,560** (78.29%)
- **Top-60 Oracle Coverage:** **8,986 / 10,560** (85.09%)
- **Top-30 Unrecoverable (Group C):** **2,293 queries** (21.71%)
- **Top-60 Unrecoverable (Subgroup C2):** **1,574 queries** (14.91%)

---

## 3. Deep Dive: Subgroup C1 (Valid Candidate in Ranks 31–60, $N = 719$)

These **719 queries** represent candidates that the multi-channel retrieval system **successfully found**, but the LambdaMART ranker suppressed below Rank 30.

### A. Rank Distribution of the First Valid Candidate
- **Ranks 31–40:** **380 queries** (52.85%)
- **Ranks 41–50:** **207 queries** (28.79%)
- **Ranks 51–60:** **132 queries** (18.36%)

### B. Retrieval Channel Origins
Which channels were responsible for unearthing these valid candidates?
- **`E_WORD_TFIDF`**: 223 queries (31.02%)
- **`I_COMM_UOM`**: 145 queries (20.17%)
- **`H_SEMANTIC_EMB`**: 130 queries (18.08%)
- **`F_CHAR_NGRAMS`**: 85 queries (11.82%)
- **`J_QUANTITY_AWARE`**: 58 queries (8.07%)
- **`D_NUMERIC_SPECS`**: 39 queries (5.42%)
- **`G_LSI_SVD`**: 27 queries (3.76%)
- **`C_MODEL_FAMILY`**: 7 queries (0.97%)

### C. Why Were These Candidates Pushed Below Rank 30?
1. **Score Deficit vs Rank 30:**
   - Mean score deficit vs Rank 30: **0.3207** (Median: **0.1799**).
   - Mean score deficit vs Rank 10: **0.7662**.
2. **Channel Consensus Deficit:**
   - In Subgroup C1, valid candidates were surfaced by an average of only **1.8 channels** (vs 4.2 channels for Top-10 distractors). When multi-channel consensus is low, LambdaMART channel-count features heavily penalize the candidate.
3. **Lexical vs Price Mismatch:**
   - Mean token Jaccard similarity between query and candidate was **0.4089**. High-ranking distractors had verbatim lexical matches for parts/accessories, whereas the price-compatible whole-unit candidate had lower word overlap.

---

## 4. Deep Dive: Subgroup C2 (No Valid Candidate in Top-60, $N = 1574$)

For these **1,574 queries**, the retrieval engine completely failed to surface any valid candidate within 60 slots. A search of the full historical catalog (strictly pre-dating the query award date) reveals the root cause:

### A. Historical Catalog Evidence Search
- **Evidence Found Elsewhere in Catalog:** **430 queries (27.32%)**
  - A historical procurement record matching this commodity at a compatible price ($\le 10\%$ error) **existed in the database**, but the 10 retrieval channels failed to pull it into the Top-60 pool!
- **No Evidence Found in Entire Catalog:** **1,144 queries (72.68%)**
  - **Literally zero comparable historical purchases** exist in the entire database at this price point. This is an intrinsic **Historical Data Coverage Limitation** (first-time purchases, custom engineered items, or radical inflationary price shifts).

### B. Failure Taxonomy Breakdown for Subgroup C2

| Failure Category | Classification Description | C2 Queries | Pct of C2 | Pct of All 10,560 Queries |
| :--- | :--- | :---: | :---: | :---: |
| **`HISTORICAL_DATA_COVERAGE_LIMITATION`** | Evidence category | **1,144** | **72.68%** | **10.83%** |
| **`RETRIEVAL_CHANNEL_MISS`** | Evidence category | **335** | **21.28%** | **3.17%** |
| **`TERMINOLOGY_VARIATION`** | Evidence category | **61** | **3.88%** | **0.58%** |
| **`QUANTITY_CONTEXT_FAILURE`** | Evidence category | **21** | **1.33%** | **0.20%** |
| **`UOM_FAILURE`** | Evidence category | **13** | **0.83%** | **0.12%** |

---

## 5. Architectural Findings & Strategic Answers

1. **Why do 2,293 queries fail in Top-30?**
   - **31.35% (719 queries, Subgroup C1)** are **Ranking Failures**: The retrieval engine found the candidate, but LambdaMART ranked it between positions 31 and 60.
   - **68.65% (1,574 queries, Subgroup C2)** are **Upstream Retrieval or Data Coverage Failures**: No candidate exists anywhere in the Top-60 pool.
2. **What dominates Subgroup C2 (Top-60 Unrecoverable)?**
   - The majority of C2 queries represent a **Historical Data Coverage Limitation**: the procurement catalog simply does not contain an equivalent item at that unit price prior to the query date.
   - However, for the queries where evidence *does* exist in the catalog, the dominant retrieval gap is **Terminology Variation & Channel Capacity**: lexical channels (BM25/TF-IDF) failed to bridge synonym differences, and channel quotas (e.g. top-10 per channel) truncated the candidates before pooling.
3. **Implications for the Next Retrieval Experiment:**
   - Re-ranking cannot solve Group C2.
   - To capture Subgroup C1 without expanding prompt tokens, we need **Stage-2 Pool Filtering** or **Targeted Reranking** that explicitly scores channel diversity.
   - To capture the recoverable portion of Subgroup C2, we need **Dual-Channel Semantic Query Expansion** or **Bi-Encoder Dense Retrieval** that searches the full catalog beyond keyword overlap.

---

*All experiments are research-only. Production ML inference and canonical frozen evaluation datasets remain 100% unmodified.*
