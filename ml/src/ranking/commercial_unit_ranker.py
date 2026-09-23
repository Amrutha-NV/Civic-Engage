"""
Procurement Intelligence System — Step 28 Commercial Unit & Pack-Size Aware Ranker.

Implements:
1. 151-dimensional Rich Feature Extractor (G0 Base + G1 Provenance + G2 Multi-View + G3 State +
   G4 Spec + G5 Distribution + G6 Hard-Case + G7 Price-Tier + G8 Commercial Unit / Pack-Size).
2. Commercial-Aware HN5 Relevance Gain Objectives for LambdaMART.
3. High-Dispersion Specialized LambdaMART Ranking Engines.
"""

import os
import sys
import math
from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    from step27_two_stage_ranker import Step27FeatureExtractor
except ImportError:
    from src.ranking.step27_two_stage_ranker import Step27FeatureExtractor

try:
    from commercial_unit_parser import CommercialUnitParser
except ImportError:
    from src.normalization.commercial_unit_parser import CommercialUnitParser

try:
    from commercial_unit_normalizer import CommercialUnitNormalizer
except ImportError:
    from src.normalization.commercial_unit_normalizer import CommercialUnitNormalizer


class Step28FeatureExtractor(Step27FeatureExtractor):
    """Expanded feature extractor incorporating commercial packaging, UOM, and price-tier normalization."""

    G8_COMMERCIAL = [
        "uom_exact_match", "uom_base_unit_match", "commercial_unit_match", "pack_size_exact_match",
        "same_pack_size", "same_base_unit", "pack_size_ratio", "pack_size_log_ratio",
        "abs_pack_size_log_diff", "pack_size_difference", "is_cross_packaging", "pack_parse_confidence_min",
        "severe_pack_mismatch", "tier_consistency_score", "norm_price_improvement", "cand_norm_base_price",
        "log_cand_norm_base_price", "z_norm_to_state", "is_pkg_q", "is_pkg_c",
        "is_single_q", "is_single_c", "is_bulk_q", "is_bulk_c"
    ]

    ALL_FEATURE_NAMES_STEP28 = Step27FeatureExtractor.ALL_FEATURE_NAMES + G8_COMMERCIAL

    def extract_pair_features_step28(
        self,
        q_row: Dict[str, Any],
        q_id: Dict[str, Any],
        q_spec: Dict[str, Any],
        q_state: Dict[str, Any],
        q_comm: Dict[str, Any],
        cand: Dict[str, Any],
        c_id: Dict[str, Any],
        c_spec: Dict[str, Any],
        c_comm: Dict[str, Any],
        pool_dist: Dict[str, float],
        cand_idx_in_pool: int
    ) -> List[float]:
        """Extract complete 151-feature vector for a query-candidate pair."""
        # Base 127 Step 27 features
        base_127 = self.extract_pair_features(
            q_row=q_row,
            q_id=q_id,
            q_spec=q_spec,
            q_state=q_state,
            cand=cand,
            c_id=c_id,
            c_spec=c_spec,
            pool_dist=pool_dist,
            cand_idx_in_pool=cand_idx_in_pool,
            feature_set="ALL"
        )

        # G8 Commercial Unit & Pack Size features (24 features)
        comm_feats = CommercialUnitNormalizer.extract_pair_commercial_features(q_comm, c_comm, q_state)
        g8_vals = [comm_feats[k] for k in self.G8_COMMERCIAL]

        return base_127 + g8_vals


class CommercialUnitAwareRanker:
    """LambdaMART ranker with commercial unit awareness and dispersion specialization."""

    def __init__(self,
                 feature_names: List[str],
                 commercial_aware_relevance: bool = False,
                 n_estimators: int = 300,
                 learning_rate: float = 0.04,
                 num_leaves: int = 31,
                 min_child_samples: int = 20,
                 subsample: float = 0.85,
                 colsample_bytree: float = 0.85):
        self.feature_names = feature_names
        self.commercial_aware_relevance = commercial_aware_relevance
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            ndcg_eval_at=[1, 3, 5, 10],
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=42,
            n_jobs=-1,
            importance_type="gain"
        )

    @classmethod
    def compute_hn5_gain(cls, target_price: float, cand_price: float,
                         is_commercial_aware: bool = False,
                         same_pack: float = 1.0) -> int:
        """Compute graded relevance gain."""
        ape = abs(cand_price - target_price) / max(target_price, 1e-4) * 100.0

        if ape <= 5.0:
            base_gain = 4
        elif ape <= 10.0:
            base_gain = 3
        elif ape <= 20.0:
            base_gain = 2
        elif ape <= 35.0:
            base_gain = 1
        else:
            base_gain = 0

        if is_commercial_aware and same_pack > 0.5 and base_gain >= 3:
            return base_gain + 1  # Bonus for exact price-tier & pack match
        return base_gain

    def fit(self, X: np.ndarray, y_gains: np.ndarray, group_lens: List[int],
            eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            eval_group: Optional[List[int]] = None):
        """Fit LambdaMART ranker."""
        eval_groups = [eval_group] if eval_group is not None else None
        self.model.fit(
            X,
            y_gains,
            group=group_lens,
            eval_set=eval_set,
            eval_group=eval_groups,
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)] if eval_set else None
        )
        return self

    def predict_rank_scores(self, X_candidates: np.ndarray) -> np.ndarray:
        """Predict ranking scores for candidate set."""
        if len(X_candidates) == 0:
            return np.array([])
        return self.model.predict(X_candidates)

    def select_best_candidate(self, candidates: List[Dict[str, Any]], scores: np.ndarray) -> Tuple[Dict[str, Any], int]:
        """Return the Rank #1 candidate."""
        if len(candidates) == 0:
            return {}, -1
        best_idx = int(np.argmax(scores))
        return candidates[best_idx], best_idx
