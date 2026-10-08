# CivicEngage Procurement Benchmark — Stratified Product-Identity Diversity Findings

This research report documents the empirical results of **Experiment 2: Stratified Product-Identity Diversity Selection** evaluated on the canonical frozen Step-37 evaluation population ($N = 10,560$ validation queries).

---

## 1. Research Objective & Experimental Architecture

- **Research Problem:** In Experiment 1, contract-level diversity (`PURCHASE_ORDER` cap at 2) was found to be slightly negative (-0.20 pp net) because multiple legitimate line items on the same multi-line municipal contract were discarded.
- **Hypothesis:** Diversity based on **Stratified Product Identity** protects valid multi-item contracts while pruning genuine duplicate product specifications (same brand+model or identical description tokens).
- **Stratified Representation Rule:**
  - **Specific / High Confidence:** `commodity_code | brand | model | uom`
  - **Generic / Low Confidence:** `commodity_code | GENERIC | clean_description_tokens[:4] | uom`
- **Selection Rule:** Sequential traversal of canonical LambdaMART ranks 1–30 with a maximum of 2 candidates per stratified product identity, producing **exactly 10 compact candidates** for each query.

```
Canonical Retrieval (10 Channels)
             ↓
LambdaMART 176-Feature Ranking (Top-30 Ranked Pool)
             ↓
Stratified Product-Identity Filter (Max 2 Candidates per Stratified Key)
             ↓
COMPACT TOP-10 CANDIDATE SET (Sent to downstream LLM)
```

---

## 2. Experimental Benchmark Results

| Metric | Canonical Baseline Top-10 | Stratified Diversity Top-10 | Absolute Delta | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Top-10 Oracle Coverage ($\le 10\%$ Error)** | **62.64%** | **62.65%** | **+0.01 pp** | **+0.02%** |
| **Top-5 Oracle Coverage ($\le 10\%$ Error)** | 54.14% | 54.06% | -0.08 pp | -0.15% |
| **Rank-1 Accuracy ($\le 10\%$ Error)** | 37.57% | 37.57% | +0.00 pp | +0.00% |
| **Tight Accuracy ($\le 5\%$ Error)** | 53.29% | 52.77% | -0.52 pp | -0.98% |
| **Broad Accuracy ($\le 20\%$ Error)** | 74.29% | 74.33% | +0.04 pp | +0.05% |
| **Rank-1 Mean APE** | 1206.51% | 1206.51% | +0.00% | — |
| **Rank-1 Median APE** | 26.95% | 26.95% | +0.00% | — |

---

## 3. Query Transition Analysis

Across the 10,560 validation queries:

- **Improved Queries (Baseline Failed $\to$ Diversity Succeeded):** **216 queries** (2.05%)
- **Degraded Queries (Baseline Succeeded $\to$ Diversity Failed):** **215 queries** (2.04%)
- **Unchanged Queries:** **10,129 queries** (95.92%)
- **Net Query Shift:** **+1 queries** (+0.01 percentage points)

---

## 4. Case Studies: Successful Recovery vs. Degradation

### A. Representative Recovery Examples

- **Query #62:** `COMPUTER ACCESSORIES AND SUPPLIES`
  - Target Price: $76.94
  - Baseline Rank-1: $119.69 (Error: 55.6%, PO: `DO560022122103778`)
  - Recovered Candidate: $79.19 (**Error: 2.9%**, Orig Rank: #11)
  - Stratified Key: `20710|GENERIC|COMPUTER_ACCESSORIES_AND_SUPPLIES|EA`

- **Query #111:** `Unmanned Aerial Vehicles (UAV), Drones`
  - Target Price: $864.00
  - Baseline Rank-1: $10,643.55 (Error: 1131.9%, PO: `CTM1100MAX83649`)
  - Recovered Candidate: $854.25 (**Error: 1.1%**, Orig Rank: #20)
  - Stratified Key: `2809521|GENERIC|CABLE_PATCH_CAT_6|EA`

- **Query #160:** `Monitors, Color and Monochrome, Environmentally Certified`
  - Target Price: $664.99
  - Baseline Rank-1: $532.99 (Error: 19.9%, PO: `DOM1100MAX125636`)
  - Recovered Candidate: $650.00 (**Error: 2.2%**, Orig Rank: #12)
  - Stratified Key: `20560|DELL|U3419W|EA`

- **Query #173:** `PARTS AND ACCESSORIES, CONVEYOR BELT`
  - Target Price: $34.43
  - Baseline Rank-1: $175.16 (Error: 408.7%, PO: `DO810022080810740`)
  - Recovered Candidate: $34.45 (**Error: 0.1%**, Orig Rank: #15)
  - Stratified Key: `1102255|GENERIC|PARTS_AND_ACCESSORIES_CONVEYOR|BOX`

- **Query #202:** `Valves, Butterfly, All Kinds`
  - Target Price: $199.00
  - Baseline Rank-1: $740.00 (Error: 271.9%, PO: `PO220022081002339`)
  - Recovered Candidate: $206.85 (**Error: 3.9%**, Orig Rank: #26)
  - Stratified Key: `68010|GENERIC|BADGE_CASES_POLICE_ALL|EA`

### B. Representative Degradation Examples

- **Query #120:** `CABLE, SIGNAL, TWISTED SHIELDED PAIR IMSA-50-2`
  - Target Price: $0.23
  - Baseline Rank-1: $0.14 (Error: 39.6%, PO: `DO240020062510476`)
  - Best Candidate Remaining: $0.29 (Error: 27.8%)
  - Degradation Cause: Valid candidate at original rank shared stratified key with two earlier candidates, resulting in exclusion.

- **Query #152:** `Air Conditioning and Heating: Central Units, and P`
  - Target Price: $5,327.00
  - Baseline Rank-1: $3,240.00 (Error: 39.2%, PO: `CTM1100MAX94762`)
  - Best Candidate Remaining: $3,240.00 (Error: 39.2%)
  - Degradation Cause: Valid candidate at original rank shared stratified key with two earlier candidates, resulting in exclusion.

- **Query #153:** `Air Conditioning and Heating: Central Units, and P`
  - Target Price: $5,327.00
  - Baseline Rank-1: $3,240.00 (Error: 39.2%, PO: `CTM1100MAX94762`)
  - Best Candidate Remaining: $3,240.00 (Error: 39.2%)
  - Degradation Cause: Valid candidate at original rank shared stratified key with two earlier candidates, resulting in exclusion.

- **Query #154:** `Air Conditioning and Heating: Central Units, and P`
  - Target Price: $5,327.00
  - Baseline Rank-1: $3,240.00 (Error: 39.2%, PO: `CTM1100MAX94762`)
  - Best Candidate Remaining: $4,695.00 (Error: 11.9%)
  - Degradation Cause: Valid candidate at original rank shared stratified key with two earlier candidates, resulting in exclusion.

- **Query #181:** `Meter Boxes, Meter Vaults, and Valve Boxes (See 21`
  - Target Price: $3,797.00
  - Baseline Rank-1: $7,094.00 (Error: 86.8%, PO: `CT2200AW220330033`)
  - Best Candidate Remaining: $2,864.50 (Error: 24.6%)
  - Degradation Cause: Valid candidate at original rank shared stratified key with two earlier candidates, resulting in exclusion.

---

## 5. Architectural Takeaways & Recommendation

1. **Comparison to Experiment 1 (Contract Diversity):**
   - Contract diversity (max 2 per `PURCHASE_ORDER`) resulted in: **-0.20 pp net** (-21 queries).
   - Stratified product identity diversity resulted in: **+0.01 pp net** (+1 query: +216 recoveries vs. -215 degradations).
   - Stratified product identity successfully avoided the net negative degradation of contract capping, but was essentially neutral overall (+0.01 pp).
2. **Discrete Cutoff Limitations:**
   - Any hard discrete ceiling (whether at the contract level or stratified key level) creates an all-or-nothing threshold where a highly relevant 3rd candidate can be discarded in favor of a weak 25th candidate.
3. **Next Step:**
   - Proceed to **Maximal Marginal Relevance (MMR)**: MMR replaces hard binary truncation with a smooth penalty decay proportional to candidate similarity, ensuring that genuinely strong candidates can still survive while suppressing near-duplicates.

---

*All experiments are research-only. Production ML inference and canonical frozen evaluation datasets remain 100% unmodified.*
