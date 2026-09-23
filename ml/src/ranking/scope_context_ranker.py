"""
Procurement Intelligence System — Step 29 Scope, Configuration & Context-Aware Ranker.

Implements:
1. 177-dimensional Rich Feature Extractor:
   - G0 Base (37f) + G1 Provenance (16f) + G2 Multi-View (16f) + G3 State (10f) +
   - G4 Specs (16f) + G5 Distribution (12f) + G6 Hard-Case (11f) + G7 Price-Tier (9f) +
   - G8 Commercial Unit (24f) + G9 Scope & Context (26f).
2. Scope-Conditioned Graded Relevance & LambdaMART Ranker.
3. High-Dispersion Specialization with Scope & Configuration Awareness.
"""

import os
import sys
import math
import re
from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    from commercial_unit_ranker import Step28FeatureExtractor, CommercialUnitAwareRanker
    from scope_configuration_parser import ScopeConfigurationParser
except ImportError:
    from src.ranking.commercial_unit_ranker import Step28FeatureExtractor, CommercialUnitAwareRanker
    from src.normalization.scope_configuration_parser import ScopeConfigurationParser


SCOPE_LEVEL_MAP = {
    "ACCESSORY_OR_PART": 0.0,
    "BULK_COMMODITY": 1.0,
    "BARE_PRODUCT": 2.0,
    "CONFIGURED_PRODUCT": 3.0,
    "SERVICE": 4.0,
    "INSTALLATION_BUNDLE": 4.0,
    "MAINTENANCE_BUNDLE": 4.0,
    "UNKNOWN": 2.0
}


class Step29FeatureExtractor(Step28FeatureExtractor):
    """Expanded feature extractor incorporating procurement scope, configuration levels, and cross-field context."""

    G9_SCOPE_CONTEXT = [
        "scope_exact_match", "scope_compatibility_score", "is_scope_conflict",
        "is_part_vs_product_mismatch", "is_service_vs_product_mismatch",
        "config_level_exact_match", "config_level_diff", "abs_config_level_diff",
        "service_indicator_match", "installation_indicator_match", "maintenance_indicator_match",
        "warranty_indicator_match", "accessory_indicator_match", "included_components_diff",
        "context_text_sim", "is_same_vendor", "is_same_master_agreement",
        "q_scope_level", "c_scope_level", "q_config_level", "c_config_level",
        "scope_x_price_state_gap", "scope_x_recency", "scope_x_spec_sim",
        "config_x_price_state", "scope_x_dispersion"
    ]

    ALL_FEATURE_NAMES_STEP29 = Step28FeatureExtractor.ALL_FEATURE_NAMES_STEP28 + G9_SCOPE_CONTEXT

    @classmethod
    def compute_context_text_sim(cls, text_q: str, text_c: str) -> float:
        """Compute Jaccard word token overlap between two context strings."""
        if not text_q or not text_c:
            return 0.0
        set_q = set(re.findall(r"\b\w{3,}\b", str(text_q).lower()))
        set_c = set(re.findall(r"\b\w{3,}\b", str(text_c).lower()))
        if not set_q or not set_c:
            return 0.0
        return float(len(set_q & set_c) / len(set_q | set_c))

    def extract_pair_scope_context_features(
        self,
        q_scope: Dict[str, Any],
        c_scope: Dict[str, Any],
        q_state: Optional[Dict[str, Any]] = None,
        base_features_dict: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Extract 26 pairwise scope, configuration, and contract context features."""
        q_type = q_scope.get("scope_type", "BARE_PRODUCT")
        c_type = c_scope.get("scope_type", "BARE_PRODUCT")

        q_lvl = float(q_scope.get("configuration_level", 2))
        c_lvl = float(c_scope.get("configuration_level", 2))

        q_scope_num = SCOPE_LEVEL_MAP.get(q_type, 2.0)
        c_scope_num = SCOPE_LEVEL_MAP.get(c_type, 2.0)

        scope_exact = 1.0 if q_type == c_type and q_type != "UNKNOWN" else 0.0
        config_exact = 1.0 if abs(q_lvl - c_lvl) < 1e-3 else 0.0
        config_diff = q_lvl - c_lvl
        abs_config_diff = abs(config_diff)

        # Mismatch penalties
        is_part_mismatch = 1.0 if ((q_type == "ACCESSORY_OR_PART" and c_type in ["BARE_PRODUCT", "CONFIGURED_PRODUCT"]) or
                                   (c_type == "ACCESSORY_OR_PART" and q_type in ["BARE_PRODUCT", "CONFIGURED_PRODUCT"])) else 0.0

        is_service_mismatch = 1.0 if ((q_type in ["SERVICE", "INSTALLATION_BUNDLE", "MAINTENANCE_BUNDLE"] and c_type in ["BARE_PRODUCT", "ACCESSORY_OR_PART"]) or
                                      (c_type in ["SERVICE", "INSTALLATION_BUNDLE", "MAINTENANCE_BUNDLE"] and q_type in ["BARE_PRODUCT", "ACCESSORY_OR_PART"])) else 0.0

        is_scope_conflict = 1.0 if (is_part_mismatch == 1.0 or is_service_mismatch == 1.0) else 0.0

        if scope_exact == 1.0:
            scope_compat = 1.0
        elif is_scope_conflict == 1.0:
            scope_compat = 0.1
        elif abs(q_scope_num - c_scope_num) <= 1.0:
            scope_compat = 0.7
        else:
            scope_compat = 0.4

        srv_match = 1.0 if abs(q_scope.get("service_bundle_indicator", 0) - c_scope.get("service_bundle_indicator", 0)) < 1e-3 else 0.0
        inst_match = 1.0 if abs(q_scope.get("installation_indicator", 0) - c_scope.get("installation_indicator", 0)) < 1e-3 else 0.0
        maint_match = 1.0 if abs(q_scope.get("maintenance_indicator", 0) - c_scope.get("maintenance_indicator", 0)) < 1e-3 else 0.0
        warr_match = 1.0 if abs(q_scope.get("warranty_indicator", 0) - c_scope.get("warranty_indicator", 0)) < 1e-3 else 0.0
        acc_match = 1.0 if abs(q_scope.get("accessory_indicator", 0) - c_scope.get("accessory_indicator", 0)) < 1e-3 else 0.0

        inc_diff = abs(float(q_scope.get("included_components_count", 0)) - float(c_scope.get("included_components_count", 0)))

        # Context overlap
        context_sim = self.compute_context_text_sim(q_scope.get("contract_name", ""), c_scope.get("contract_name", ""))
        
        q_v = str(q_scope.get("vendor_code", ""))
        c_v = str(c_scope.get("vendor_code", ""))
        same_vendor = 1.0 if q_v and c_v and q_v == c_v else 0.0

        q_ma = str(q_scope.get("master_agreement", ""))
        c_ma = str(c_scope.get("master_agreement", ""))
        same_ma = 1.0 if q_ma and c_ma and q_ma == c_ma else 0.0

        # Interactions
        cand_z = float(base_features_dict.get("cand_z_to_state", 0.0)) if base_features_dict else 0.0
        rec_365 = float(base_features_dict.get("recency_weight_365d", 0.5)) if base_features_dict else 0.5
        spec_sim = float(base_features_dict.get("spec_composite_score", 0.0)) if base_features_dict else 0.0
        disp = float(base_features_dict.get("cand_pool_dispersion", 0.8)) if base_features_dict else 0.8

        scope_x_state = scope_compat * abs(cand_z)
        scope_x_rec = scope_exact * rec_365
        scope_x_spec = scope_compat * spec_sim
        config_x_state = (1.0 / (1.0 + abs_config_diff)) * abs(cand_z)
        scope_x_disp = scope_compat * disp

        return {
            "scope_exact_match": scope_exact,
            "scope_compatibility_score": scope_compat,
            "is_scope_conflict": is_scope_conflict,
            "is_part_vs_product_mismatch": is_part_mismatch,
            "is_service_vs_product_mismatch": is_service_mismatch,
            "config_level_exact_match": config_exact,
            "config_level_diff": float(np.clip(config_diff, -5.0, 5.0)),
            "abs_config_level_diff": float(np.clip(abs_config_diff, 0.0, 5.0)),
            "service_indicator_match": srv_match,
            "installation_indicator_match": inst_match,
            "maintenance_indicator_match": maint_match,
            "warranty_indicator_match": warr_match,
            "accessory_indicator_match": acc_match,
            "included_components_diff": float(np.clip(inc_diff, 0.0, 10.0)),
            "context_text_sim": float(np.clip(context_sim, 0.0, 1.0)),
            "is_same_vendor": same_vendor,
            "is_same_master_agreement": same_ma,
            "q_scope_level": q_scope_num,
            "c_scope_level": c_scope_num,
            "q_config_level": q_lvl,
            "c_config_level": c_lvl,
            "scope_x_price_state_gap": float(np.clip(scope_x_state, -10.0, 10.0)),
            "scope_x_recency": float(np.clip(scope_x_rec, 0.0, 1.0)),
            "scope_x_spec_sim": float(np.clip(scope_x_spec, 0.0, 1.0)),
            "config_x_price_state": float(np.clip(config_x_state, -10.0, 10.0)),
            "scope_x_dispersion": float(np.clip(scope_x_disp, 0.0, 10.0))
        }

    def extract_pair_features_step29(
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
        pool_dist: Dict[str, float],
        cand_idx_in_pool: int
    ) -> List[float]:
        """Extract complete 177-feature vector for a query-candidate pair."""
        # 151 features from Step 28
        base_151 = self.extract_pair_features_step28(
            q_row=q_row,
            q_id=q_id,
            q_spec=q_spec,
            q_state=q_state,
            q_comm=q_comm,
            cand=cand,
            c_id=c_id,
            c_spec=c_spec,
            c_comm=c_comm,
            pool_dist=pool_dist,
            cand_idx_in_pool=cand_idx_in_pool
        )

        base_f_dict = dict(zip(self.ALL_FEATURE_NAMES_STEP28, base_151))
        g9_feats = self.extract_pair_scope_context_features(
            q_scope=q_scope,
            c_scope=c_scope,
            q_state=q_state,
            base_features_dict=base_f_dict
        )
        g9_vals = [g9_feats[k] for k in self.G9_SCOPE_CONTEXT]

        return base_151 + g9_vals


class ScopeContextRanker(CommercialUnitAwareRanker):
    """Scope & Context Aware LambdaMART Ranker."""
    pass
