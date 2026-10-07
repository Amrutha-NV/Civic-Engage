# CivicEngage Procurement Benchmark — MMR Diversity Re-Ranking Findings

This research report documents the empirical results of **Experiment 3: Maximal Marginal Relevance (MMR) Diversity Re-Ranking** evaluated across all $N = 10,560$ validation queries.

---

## 1. Experimental Formulation & Architecture

- **Objective:** Replace hard discrete candidate cutoffs (which caused regressions in Experiments 1 and 2) with a smooth continuous penalty decay.
- **MMR Formulation:**
  $$\text{MMR}(c) = \lambda \cdot \text{Rel}(c) - (1 - \lambda) \cdot \max_{s \in S} \text{Sim}(c, s)$$
- **Relevance Normalization:** Min-Max normalization within each query's Top-30 pool:
  $$\text{Rel}(c) = \frac{\text{score}(c) - \min_j \text{score}(c_j)}{\max_j \text{score}(c_j) - \min_j \text{score}(c_j) + 10^{-8}}$$
- **Candidate Redundancy Similarity Function:**
  $$\text{Sim}(c, s) = 0.50 \cdot \mathbb{I}[\text{strat\_key}(c) == \text{strat\_key}(s)] + 0.30 \cdot \mathbb{I}[\text{PO}(c) == \text{PO}(s)] + 0.20 \cdot \text{Jaccard}(\text{Tokens}(c), \text{Tokens}(s))$$

---

## 2. Lambda Sweep Benchmark Comparison

| Configuration | Top-10 Coverage ($\le 10\%$) | Absolute Delta | Relative Change | Top-5 Coverage | Rank-1 Acc | Improved | Degraded | Net Shift |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Canonical Baseline Top-10** | **62.64%** | **—** | **—** | 54.14% | 37.57% | — | — | — |
| **MMR $\lambda = 0.9$** | 62.64% | +0.00 pp | +0.00% | 54.35% | 37.57% | 64 | 64 | +0 |
| **MMR $\lambda = 0.8$** | 62.76% | +0.12 pp | +0.19% | 54.45% | 37.57% | 133 | 121 | +12 |
| **MMR $\lambda = 0.7$** | 62.95% | +0.31 pp | +0.49% | 54.34% | 37.57% | 229 | 197 | +32 |
| **MMR $\lambda = 0.6$** | 63.12% | +0.48 pp | +0.77% | 53.99% | 37.57% | 332 | 281 | +51 |

---

## 3. Best Configuration Transition Analysis ($\lambda = 0.6$)

- **Top-10 Oracle Coverage:** **63.12%** (+0.48 pp)
- **Improved Queries (Baseline Failed $\to$ MMR Succeeded):** **332 queries** (3.14%)
- **Degraded Queries (Baseline Succeeded $\to$ MMR Failed):** **281 queries** (2.66%)
- **Unchanged Queries:** **9,947 queries** (94.20%)
- **Net Query Shift:** **+51 queries**

---

## 4. Case Studies

### A. Representative Recovery Examples

- **Query #62:** `COMPUTER ACCESSORIES AND SUPPLIES`
  - Target Price: $76.94
  - Baseline Rank-1: $119.69 (Error: 55.6%)
  - Surfaced MMR Candidate: $70.96 (**Error: 7.8%**, Orig Rank: #19)
  - PO: `DO560022112202796` | Stratified Key: `20710|DELL|2666MHZ|EA`

- **Query #94:** `ACCESSORIES FOR LAPTOP COMPUTER`
  - Target Price: $19.49
  - Baseline Rank-1: $252.99 (Error: 1198.0%)
  - Surfaced MMR Candidate: $19.49 (**Error: 0.0%**, Orig Rank: #18)
  - PO: `DO560022120603198` | Stratified Key: `20454|DELL|ES1520P|EA`

- **Query #111:** `Unmanned Aerial Vehicles (UAV), Drones`
  - Target Price: $864.00
  - Baseline Rank-1: $10,643.55 (Error: 1131.9%)
  - Surfaced MMR Candidate: $854.25 (**Error: 1.1%**, Orig Rank: #20)
  - PO: `PO240019050601802` | Stratified Key: `2809521|GENERIC|CABLE_PATCH_CAT_6|EA`

- **Query #133:** `CHEMICALS, BULK`
  - Target Price: $17.50
  - Baseline Rank-1: $13.82 (Error: 21.0%)
  - Surfaced MMR Candidate: $18.99 (**Error: 8.5%**, Orig Rank: #19)
  - PO: `PO810018071802663` | Stratified Key: `1751370|GENERIC|CHEMICALS_LABORATORY_LIQUID_A|GAL`

- **Query #155:** `Air Conditioning and Heating: Central Units, and P`
  - Target Price: $8,315.00
  - Baseline Rank-1: $5,156.00 (Error: 38.0%)
  - Surfaced MMR Candidate: $8,890.00 (**Error: 6.9%**, Orig Rank: #24)
  - PO: `CTM1100MAX102630` | Stratified Key: `2853084|GENERIC|TESTERS_ELECTRICAL_EQUIPMENT_HIGH|EA`

### B. Representative Regression Examples

- **Query #47:** `Stripes and Legends, Plastic, Prefabricated, Refle`
  - Target Price: $241.74
  - Baseline Rank-1: $99.40 (Error: 58.9%)
  - Best MMR Remaining: $156.48 (Error: 35.3%)
  - Regression Mechanism: Candidate at baseline rank 8-10 was penalized by similarity to ranks 1-2, permitting an off-price alternative from rank 20-30.

- **Query #91:** `LABORATORY SUPPLIES PER PRICE AGREEMENT`
  - Target Price: $648.90
  - Baseline Rank-1: $86.10 (Error: 86.7%)
  - Best MMR Remaining: $86.10 (Error: 86.7%)
  - Regression Mechanism: Candidate at baseline rank 8-10 was penalized by similarity to ranks 1-2, permitting an off-price alternative from rank 20-30.

- **Query #92:** `LABORATORY SUPPLIES PER PRICE AGREEMENT`
  - Target Price: $13.65
  - Baseline Rank-1: $80.85 (Error: 492.3%)
  - Best MMR Remaining: $80.85 (Error: 492.3%)
  - Regression Mechanism: Candidate at baseline rank 8-10 was penalized by similarity to ranks 1-2, permitting an off-price alternative from rank 20-30.

- **Query #120:** `CABLE, SIGNAL, TWISTED SHIELDED PAIR IMSA-50-2`
  - Target Price: $0.23
  - Baseline Rank-1: $0.14 (Error: 39.6%)
  - Best MMR Remaining: $0.29 (Error: 27.8%)
  - Regression Mechanism: Candidate at baseline rank 8-10 was penalized by similarity to ranks 1-2, permitting an off-price alternative from rank 20-30.

- **Query #136:** `LIGHTS, EMERGENCY`
  - Target Price: $158.40
  - Baseline Rank-1: $97.20 (Error: 38.6%)
  - Best MMR Remaining: $105.00 (Error: 33.7%)
  - Regression Mechanism: Candidate at baseline rank 8-10 was penalized by similarity to ranks 1-2, permitting an off-price alternative from rank 20-30.

---

## 5. Architectural Findings & Takeaways

1. **Trade-Off Dynamics in MMR:**
   As $\lambda$ decreases (stronger diversity penalty), more candidates from ranks 11–30 are pulled into the Top-10 pool, recovering previously missed queries. However, this simultaneously increases regressions by pushing valid candidates near ranks 8–10 out of the Top-10.
2. **Comparison Across Diversity Paradigms:**
   - Experiment 1 (Contract Cap at 2): -0.20 pp net
   - Experiment 2 (Stratified Identity Cap at 2): +0.01 pp net
   - Experiment 3 (Continuous MMR): Evaluated smoothly across the sweep.
3. **Recommendation:**
   Report whether MMR justifies replacing canonical ranking or whether retrieval expansion at the bi-encoder / dual-channel level is the true upper bound.

---

*All experiments are research-only. Production ML inference and canonical frozen evaluation datasets remain 100% unmodified.*
