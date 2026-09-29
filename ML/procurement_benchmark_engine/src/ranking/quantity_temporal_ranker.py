"""
Procurement Intelligence System — Step 30 Quantity Scale & Temporal Regime Ranker.

Implements:
1. 207-dimensional Rich Feature Extractor:
   - Base Step 29 Features (177f)
   - G10 Quantity Scale & Elasticity (14f)
   - G11 Temporal Regime & Interactions (16f)
2. Quantity/Temporal-Conditioned Graded Relevance & LambdaMART Ranker.
3. High-Dispersion Specialization with Quantity & Temporal Awareness.
"""

import os
import sys
import math
import re
from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np
import pandas as pd
import lightgbm as lgb

try:
    from commercial_unit_ranker import Step28FeatureExtractor, CommercialUnitAwareRanker
    from scope_configuration_parser import ScopeConfigurationParser
    from scope_context_ranker import Step29FeatureExtractor, ScopeContextRanker
    from quantity_temporal_modeler import CausalQuantityTemporalModeler, get_quantity_tier
except ImportError:
    from src.ranking.commercial_unit_ranker import Step28FeatureExtractor, CommercialUnitAwareRanker
    from src.normalization.scope_configuration_parser import ScopeConfigurationParser
    from src.ranking.scope_context_ranker import Step29FeatureExtractor, ScopeContextRanker
    from src.normalization.quantity_temporal_modeler import CausalQuantityTemporalModeler, get_quantity_tier


class Step30FeatureExtractor(Step29FeatureExtractor):
    """Expanded feature extractor incorporating non-linear quantity elasticity and temporal regimes."""

    G10_QUANTITY_SCALE = [
        "qty_ratio_scaled", "log_qty_ratio_val", "qty_similarity_exp",
        "q_qty_tier", "c_qty_tier", "qty_tier_match",
        "identity_quantity_beta", "commodity_quantity_beta", "quantity_beta_confidence",
        "expected_log_price_change_from_qty", "quantity_adjusted_cand_price_gap",
        "quantity_x_price_proximity", "quantity_x_dispersion", "quantity_support_count"
    ]

    G11_TEMPORAL_REGIME = [
        "candidate_age_days", "candidate_age_years", "recency_bucket_idx",
        "identity_recent_price_slope", "identity_price_regime_distance", "candidate_vs_recent_IQR",
        "recent_support_count", "recency_confidence_exp", "temporal_state_gap",
        "recency_x_identity", "recency_x_price_state", "recency_x_quantity",
        "recency_x_dispersion", "recency_x_scope", "recency_x_spec_similarity",
        "quantity_temporal_interaction"
    ]

    ALL_FEATURE_NAMES_STEP30 = Step29FeatureExtractor.ALL_FEATURE_NAMES_STEP29 + G10_QUANTITY_SCALE + G11_TEMPORAL_REGIME

    def extract_pair_quantity_temporal_features(
        self,
        q_row: Dict[str, Any],
        cand: Dict[str, Any],
        q_state: Optional[Dict[str, Any]] = None,
        q_scope: Optional[Dict[str, Any]] = None,
        c_scope: Optional[Dict[str, Any]] = None,
        causal_stats: Optional[Dict[str, Any]] = None,
        base_features_dict: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Extract 30 pairwise quantity and temporal regime features."""
        # Query & Candidate Quantities
        q_qty = float(q_row.get("quantity_numeric", 1.0))
        c_qty = float(cand.get("quantity_numeric", 1.0)) if "quantity_numeric" in cand else 1.0
        if c_qty <= 0: c_qty = 1.0
        if q_qty <= 0: q_qty = 1.0

        lq_q = math.log(max(q_qty, 1e-3))
        lq_c = math.log(max(c_qty, 1e-3))
        abs_lq_diff = abs(lq_q - lq_c)

        ratio_scaled = min(q_qty / c_qty, c_qty / q_qty)
        qty_sim_exp = math.exp(-0.5 * abs_lq_diff)

        q_tier = get_quantity_tier(q_qty)
        c_tier = get_quantity_tier(c_qty)
        tier_match = 1.0 if abs(q_tier - c_tier) < 1e-3 else 0.0

        # Causal Elasticity Stats
        if causal_stats:
            id_beta = float(causal_stats.get("identity_beta", -0.05))
            comm_beta = float(causal_stats.get("commodity_beta", -0.05))
            shrunk_b = float(causal_stats.get("shrunk_beta", -0.05))
            beta_conf = float(causal_stats.get("beta_confidence", 0.0))
            supp_cnt = float(causal_stats.get("support_count", 0))
            recent_cnt = float(causal_stats.get("recent_support_count", 0))
            rec_med = causal_stats.get("recent_median_price")
            rec_iqr = float(causal_stats.get("recent_iqr_price", 0.0))
            slope = float(causal_stats.get("annualized_slope", 0.0))
        else:
            id_beta = -0.05
            comm_beta = -0.05
            shrunk_b = -0.05
            beta_conf = 0.0
            supp_cnt = 0.0
            recent_cnt = 0.0
            rec_med = None
            rec_iqr = 0.0
            slope = 0.0

        expected_lp_change = shrunk_b * (lq_q - lq_c)

        # Price State interaction
        st_price = float(q_state.get("posterior_mean_price", 100.0)) if q_state else 100.0
        log_st_price = math.log(max(st_price, 1e-4))
        c_price = float(cand.get("target_unit_price", 100.0))
        log_c_price = math.log(max(c_price, 1e-4))

        qty_adj_gap = abs(log_c_price + expected_lp_change - log_st_price)

        cand_z = float(base_features_dict.get("cand_z_to_state", 0.0)) if base_features_dict else 0.0
        disp = float(base_features_dict.get("cand_pool_dispersion", 0.8)) if base_features_dict else 0.8
        spec_sim = float(base_features_dict.get("spec_composite_score", 0.0)) if base_features_dict else 0.0
        is_id_match = float(base_features_dict.get("is_exact_product_match", 0.0)) if base_features_dict else 0.0
        scope_compat = float(base_features_dict.get("scope_compatibility_score", 0.5)) if base_features_dict else 0.5

        qty_x_state = qty_sim_exp * (1.0 / (1.0 + abs(cand_z)))
        qty_x_disp = qty_sim_exp * disp

        # Temporal Age & Recency
        q_dt = pd.to_datetime(q_row["award_date_parsed"])
        c_dt = pd.to_datetime(cand["award_date_parsed"]) if "award_date_parsed" in cand else q_dt
        age_days = max(0.0, float((q_dt - c_dt).days))
        age_years = age_days / 365.25

        if age_days <= 180:
            rec_bucket = 0.0
        elif age_days <= 365:
            rec_bucket = 1.0
        elif age_days <= 730:
            rec_bucket = 2.0
        elif age_days <= 1095:
            rec_bucket = 3.0
        elif age_days <= 1825:
            rec_bucket = 4.0
        else:
            rec_bucket = 5.0

        rec_conf_exp = math.exp(-age_days / 365.25)

        if rec_med is not None and rec_med > 0:
            id_regime_dist = abs(log_c_price - math.log(max(rec_med, 1e-4)))
            cand_vs_iqr = (log_c_price - math.log(max(rec_med, 1e-4))) / max(rec_iqr, 0.1)
        else:
            id_regime_dist = abs(log_c_price - log_st_price)
            cand_vs_iqr = 0.0

        temp_state_gap = abs(log_c_price - (log_st_price + slope * age_years))

        rec_x_id = rec_conf_exp * is_id_match
        rec_x_state = rec_conf_exp * (1.0 / (1.0 + abs(cand_z)))
        rec_x_qty = rec_conf_exp * qty_sim_exp
        rec_x_disp = rec_conf_exp * disp
        rec_x_scope = rec_conf_exp * scope_compat
        rec_x_spec = rec_conf_exp * spec_sim
        qty_temp_inter = qty_sim_exp * rec_conf_exp * (1.0 / (1.0 + abs(cand_z)))

        return {
            "qty_ratio_scaled": float(np.clip(ratio_scaled, 0.0, 1.0)),
            "log_qty_ratio_val": float(np.clip(abs_lq_diff, 0.0, 10.0)),
            "qty_similarity_exp": float(np.clip(qty_sim_exp, 0.0, 1.0)),
            "q_qty_tier": q_tier,
            "c_qty_tier": c_tier,
            "qty_tier_match": tier_match,
            "identity_quantity_beta": float(np.clip(id_beta, -1.0, 1.0)),
            "commodity_quantity_beta": float(np.clip(comm_beta, -1.0, 1.0)),
            "quantity_beta_confidence": float(np.clip(beta_conf, 0.0, 1.0)),
            "expected_log_price_change_from_qty": float(np.clip(expected_lp_change, -5.0, 5.0)),
            "quantity_adjusted_cand_price_gap": float(np.clip(qty_adj_gap, 0.0, 10.0)),
            "quantity_x_price_proximity": float(np.clip(qty_x_state, 0.0, 5.0)),
            "quantity_x_dispersion": float(np.clip(qty_x_disp, 0.0, 5.0)),
            "quantity_support_count": float(np.clip(supp_cnt, 0.0, 500.0)),
            "candidate_age_days": float(np.clip(age_days, 0.0, 10000.0)),
            "candidate_age_years": float(np.clip(age_years, 0.0, 30.0)),
            "recency_bucket_idx": rec_bucket,
            "identity_recent_price_slope": float(np.clip(slope, -2.0, 2.0)),
            "identity_price_regime_distance": float(np.clip(id_regime_dist, 0.0, 10.0)),
            "candidate_vs_recent_IQR": float(np.clip(cand_vs_iqr, -10.0, 10.0)),
            "recent_support_count": float(np.clip(recent_cnt, 0.0, 100.0)),
            "recency_confidence_exp": float(np.clip(rec_conf_exp, 0.0, 1.0)),
            "temporal_state_gap": float(np.clip(temp_state_gap, 0.0, 10.0)),
            "recency_x_identity": float(np.clip(rec_x_id, 0.0, 1.0)),
            "recency_x_price_state": float(np.clip(rec_x_state, 0.0, 5.0)),
            "recency_x_quantity": float(np.clip(rec_x_qty, 0.0, 1.0)),
            "recency_x_dispersion": float(np.clip(rec_x_disp, 0.0, 5.0)),
            "recency_x_scope": float(np.clip(rec_x_scope, 0.0, 1.0)),
            "recency_x_spec_similarity": float(np.clip(rec_x_spec, 0.0, 1.0)),
            "quantity_temporal_interaction": float(np.clip(qty_temp_inter, 0.0, 5.0))
        }

    def extract_pair_features_step30(
        self,
        q_row: Dict[str, Any],
        q_id: Dict[str, Any],
        q_spec: Dict[str, Any],
        q_state: Dict[str, Any],
        q_comm: Dict[str, Any],
        q_scope: Dict[str, Any],
        cand: Dict[str, Any],
        c_id: Dict[str, Any],
        c_spec: Dict[str, Any],
        c_comm: Dict[str, Any],
        c_scope: Dict[str, Any],
        causal_stats: Optional[Dict[str, Any]] = None,
        pool_dist: Optional[Dict[str, float]] = None,
        cand_idx_in_pool: int = 0
    ) -> List[float]:
        """Extract full 207-dimensional Step 30 feature vector."""
        # Extract Step 29 177 features
        feats_177 = self.extract_pair_features_step29(
            q_row=q_row, q_id=q_id, q_spec=q_spec, q_state=q_state, q_comm=q_comm, q_scope=q_scope,
            cand=cand, c_id=c_id, c_spec=c_spec, c_comm=c_comm, c_scope=c_scope,
            pool_dist=pool_dist, cand_idx_in_pool=cand_idx_in_pool
        )
        base_dict = dict(zip(self.ALL_FEATURE_NAMES_STEP29, feats_177))

        # Extract 30 Quantity + Temporal features
        qt_dict = self.extract_pair_quantity_temporal_features(
            q_row=q_row, cand=cand, q_state=q_state, q_scope=q_scope, c_scope=c_scope,
            causal_stats=causal_stats, base_features_dict=base_dict
        )

        feats_30 = [qt_dict[k] for k in self.G10_QUANTITY_SCALE + self.G11_TEMPORAL_REGIME]
        return feats_177 + feats_30


class QuantityTemporalAwareRanker(ScopeContextRanker):
    """LambdaMART Ranker with quantity and temporal regime features."""
    pass
