"""
Procurement Intelligence System — Hierarchical State Feature Extractor (Step 22).

Enriches base LTR candidate features with causal hierarchical price-state estimates,
state-candidate gaps, empirical-Bayes uncertainties, and multi-level dispersion signals.
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class StateFeatureExtractor:
    """
    Extracts hierarchical price-state features and state-candidate interaction features.
    """

    QUERY_STATE_FEATURE_NAMES = [
        "identity_state",                   # 0
        "model_state",                      # 1
        "brand_state",                      # 2
        "family_state",                     # 3
        "commodity_state",                  # 4
        "global_state",                     # 5
        "identity_vs_model_gap",            # 6
        "model_vs_family_gap",              # 7
        "family_vs_commodity_gap",          # 8
        "identity_support",                 # 9
        "model_support",                    # 10
        "family_support",                   # 11
        "commodity_support",                # 12
        "identity_dispersion",              # 13
        "model_dispersion",                 # 14
        "family_dispersion",                # 15
        "identity_recency",                 # 16
        "model_recency",                    # 17
        "family_recency",                   # 18
        "posterior_price_state",            # 19
        "posterior_uncertainty"             # 20
    ]

    CANDIDATE_STATE_INTERACTION_NAMES = [
        "cand_vs_identity_state",           # 21
        "cand_vs_model_state",              # 22
        "cand_vs_brand_state",              # 23
        "cand_vs_family_state",             # 24
        "cand_vs_commodity_state",          # 25
        "cand_vs_posterior_state",          # 26
        "abs_log_gap_to_state",             # 27
        "cand_recency_days",                # 28
        "cand_quantity_numeric",            # 29
        "query_quantity_numeric",           # 30
        "log_quantity_ratio_state",         # 31
        "cand_dispersion_proxy",            # 32
        "historical_support_count"          # 33
    ]

    ALL_STATE_FEATURE_NAMES = QUERY_STATE_FEATURE_NAMES + CANDIDATE_STATE_INTERACTION_NAMES

    def extract_state_features(
        self,
        q_row: Dict[str, Any],
        cand_dict: Dict[str, Any],
        state_dict: Dict[str, Any],
        q_date: Any
    ) -> List[float]:
        """
        Extracts 34 state and state-interaction features for a (query, candidate) pair.
        Strictly causal, 0 target price leakage.
        """
        # Query State Features
        id_st = float(state_dict.get("identity_state", 4.0))
        mod_st = float(state_dict.get("model_state", 4.0))
        br_st = float(state_dict.get("brand_state", 4.0))
        fam_st = float(state_dict.get("family_state", 4.0))
        comm_st = float(state_dict.get("commodity_state", 4.0))
        glob_st = float(state_dict.get("global_state", 4.0))
        post_st = float(state_dict.get("posterior_price_state", 4.0))
        post_unc = float(state_dict.get("posterior_uncertainty", 1.0))

        id_vs_mod = float(state_dict.get("identity_vs_model_gap", 0.0))
        mod_vs_fam = float(state_dict.get("model_vs_family_gap", 0.0))
        fam_vs_comm = float(state_dict.get("family_vs_commodity_gap", 0.0))

        id_supp = float(state_dict.get("identity_support", 0.0))
        mod_supp = float(state_dict.get("model_support", 0.0))
        fam_supp = float(state_dict.get("family_support", 0.0))
        comm_supp = float(state_dict.get("commodity_support", 0.0))

        id_disp = float(state_dict.get("identity_dispersion", 0.5))
        mod_disp = float(state_dict.get("model_dispersion", 0.5))
        fam_disp = float(state_dict.get("family_dispersion", 0.5))

        id_rec = float(state_dict.get("identity_recency", 9999.0))
        mod_rec = float(state_dict.get("model_recency", 9999.0))
        fam_rec = float(state_dict.get("family_recency", 9999.0))

        # Candidate interactions
        cand_p = max(float(cand_dict.get("target_unit_price", 0.0)), 1e-4)
        log_cand_p = math.log(cand_p)

        c_vs_id = log_cand_p - id_st
        c_vs_mod = log_cand_p - mod_st
        c_vs_br = log_cand_p - br_st
        c_vs_fam = log_cand_p - fam_st
        c_vs_comm = log_cand_p - comm_st
        c_vs_post = log_cand_p - post_st
        abs_gap_post = abs(c_vs_post)

        # Quantities and Recency
        cand_date = cand_dict.get("award_date_parsed")
        dt_cand = 0.0
        if cand_date is not None and q_date is not None:
            try:
                dt_cand = max(float((cand_date - q_date).days), 1.0) if hasattr(cand_date, "days") else max(float((cand_date - q_date) / np.timedelta64(1, "D")), 1.0)
            except Exception:
                dt_cand = 100.0

        cand_qty = max(float(cand_dict.get("quantity_numeric", 1.0)), 1.0)
        q_qty = max(float(q_row.get("quantity_numeric", 1.0)), 1.0)
        log_qty_ratio = math.log(q_qty / cand_qty)

        cand_disp_proxy = float(state_dict.get("identity_dispersion", 0.5))
        hist_supp_cnt = float(state_dict.get("identity_count", 0))

        feats = [
            id_st, mod_st, br_st, fam_st, comm_st, glob_st,
            id_vs_mod, mod_vs_fam, fam_vs_comm,
            id_supp, mod_supp, fam_supp, comm_supp,
            id_disp, mod_disp, fam_disp,
            id_rec, mod_rec, fam_rec,
            post_st, post_unc,
            c_vs_id, c_vs_mod, c_vs_br, c_vs_fam, c_vs_comm, c_vs_post,
            abs_gap_post,
            dt_cand,
            cand_qty, q_qty, log_qty_ratio,
            cand_disp_proxy,
            hist_supp_cnt
        ]
        return feats
