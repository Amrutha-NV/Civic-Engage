"""
Procurement Intelligence System — Confidence Engine & Selective Prediction Gate (Step 17C).

Implements multi-signal confidence scoring and selective prediction gating:
1. Low-dispersion score (weight 0.30)
2. Rank-margin score (weight 0.20)
3. Identity confidence score (weight 0.15)
4. Specification similarity score (weight 0.10)
5. Candidate recency score (weight 0.10)
6. Candidate support score (weight 0.10)
7. Match tier priority score (weight 0.05)

Strict Anti-Leakage Invariants:
- Query UNIT_PRICE / target_unit_price / APE is NEVER accessed.
- Strictly uses pre-target metadata and candidate ranking signals.
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


class ConfidenceGate:
    """Computes transparent normalized confidence scores in [0, 1] and manages selective gating."""

    TIER_WEIGHTS = {
        "TIER_1_EXACT_MPN": 1.00,
        "TIER_2_EXACT_BRAND_MODEL": 0.85,
        "TIER_3_SUPERVISED_MATCH": 0.70,
        "TIER_4_SPEC_SIMILARITY": 0.40,
        "TIER_5_COMM_UOM": 0.15,
        "FALLBACK": 0.00,
        "UNKNOWN": 0.20
    }

    def __init__(
        self,
        w_disp: float = 0.30,
        w_margin: float = 0.20,
        w_ident: float = 0.15,
        w_spec: float = 0.10,
        w_recency: float = 0.10,
        w_support: float = 0.10,
        w_tier: float = 0.05,
        disp_decay: float = 2.5,
        recency_decay_days: float = 730.0
    ):
        self.w_disp = w_disp
        self.w_margin = w_margin
        self.w_ident = w_ident
        self.w_spec = w_spec
        self.w_recency = w_recency
        self.w_support = w_support
        self.w_tier = w_tier

        self.disp_decay = disp_decay
        self.recency_decay_days = recency_decay_days

        total_w = w_disp + w_margin + w_ident + w_spec + w_recency + w_support + w_tier
        assert abs(total_w - 1.0) < 1e-4, f"Weights must sum to 1.0 (got {total_w})"

    def compute_confidence(
        self,
        q_dict: Dict[str, Any],
        sorted_candidates: List[Dict[str, Any]],
        sorted_scores: np.ndarray
    ) -> Dict[str, float]:
        """
        Computes all 7 sub-scores and the composite confidence score for a query.
        Returns a dict containing sub-scores, log_std, and composite_score.
        """
        n_cands = len(sorted_candidates)

        if n_cands == 0:
            return {
                "score_disp": 0.0,
                "score_margin": 0.0,
                "score_ident": 0.0,
                "score_spec": 0.0,
                "score_recency": 0.0,
                "score_support": 0.0,
                "score_tier": 0.0,
                "log_std": np.nan,
                "dispersion_score": 0.0,
                "composite_score": 0.0
            }

        top_cand = sorted_candidates[0]

        # 1. Low-Dispersion Score (weight 0.30)
        c_prices = [float(c["target_unit_price"]) for c in sorted_candidates[:min(5, n_cands)]]
        if len(c_prices) >= 2:
            log_std = float(np.std(np.log(np.maximum(c_prices, 1e-4))))
            score_disp = float(math.exp(-self.disp_decay * log_std))
        else:
            log_std = 0.0
            score_disp = 0.50  # Neutral for single candidate

        # 2. Rank Margin Score (weight 0.20)
        if len(sorted_scores) >= 2:
            margin = float(sorted_scores[0] - sorted_scores[1])
            score_margin = float(math.tanh(max(0.0, margin) / 0.50))
        else:
            score_margin = 0.80  # Strong if unambiguous single match

        # 3. Identity Confidence Score (weight 0.15)
        tier_val = top_cand.get("tier", "UNKNOWN")
        q_ident_conf = str(q_dict.get("product_identity_confidence", "") or "").strip().lower()
        if tier_val == "TIER_1_EXACT_MPN":
            score_ident = 1.00
        elif tier_val == "TIER_2_EXACT_BRAND_MODEL":
            score_ident = 0.85
        elif q_ident_conf == "high":
            score_ident = 0.70
        elif q_ident_conf == "medium":
            score_ident = 0.50
        else:
            score_ident = 0.20

        # 4. Specification Similarity Score (weight 0.10)
        spec_cos = float(top_cand.get("spec_cosine_similarity", 0.50))
        score_spec = float(max(0.0, min(1.0, spec_cos)))

        # 5. Candidate Recency Score (weight 0.10)
        days_el = float(top_cand.get("days_elapsed", 365.0))
        score_recency = float(math.exp(-days_el / self.recency_decay_days))

        # 6. Candidate Support Score (weight 0.10)
        score_support = float(min(1.0, n_cands / 5.0))

        # 7. Match Tier Priority Score (weight 0.05)
        score_tier = float(self.TIER_WEIGHTS.get(tier_val, 0.20))

        # Composite Confidence
        composite = (
            self.w_disp * score_disp +
            self.w_margin * score_margin +
            self.w_ident * score_ident +
            self.w_spec * score_spec +
            self.w_recency * score_recency +
            self.w_support * score_support +
            self.w_tier * score_tier
        )
        composite = float(max(0.0, min(1.0, composite)))

        return {
            "score_disp": score_disp,
            "score_margin": score_margin,
            "score_ident": score_ident,
            "score_spec": score_spec,
            "score_recency": score_recency,
            "score_support": score_support,
            "score_tier": score_tier,
            "log_std": log_std,
            "dispersion_score": score_disp,
            "composite_score": composite
        }
