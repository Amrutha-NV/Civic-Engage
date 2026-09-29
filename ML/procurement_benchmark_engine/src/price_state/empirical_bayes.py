"""
Procurement Intelligence System — Hierarchical Empirical Bayes Shrinkage Engine (Step 22).

Implements multi-level partial pooling across:
- Level 0: Exact Product / Normalized Identity (identity_key)
- Level 1: Model / Part (commodity_code + normalized_model)
- Level 2: Brand (commodity_family + normalized_brand)
- Level 3: Commodity Family + UOM (commodity_family + uom_standardized)
- Level 4: Commodity Code + UOM (commodity_code + uom_standardized)
- Level 5: Global Prior (Dataset-level baseline)

Provides empirical Bayes shrinkage weighting, posterior uncertainty propagation,
and inter-level disagreement / gap signals.
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class HierarchicalShrinkageEngine:
    """
    Multi-level empirical-Bayes partial pooling engine for price state estimation.
    """

    DEFAULT_KAPPA = {
        "identity": 2.0,      # Identity needs ~2 effective historical observations for 50% weight
        "model": 3.0,         # Model level needs ~3 observations
        "brand": 5.0,         # Brand level needs ~5 observations
        "commodity_uom": 4.0, # Commodity+UOM needs ~4 observations
        "family_uom": 8.0     # Family+UOM needs ~8 observations
    }

    def __init__(self, kappa_dict: Optional[Dict[str, float]] = None):
        self.kappa = dict(self.DEFAULT_KAPPA)
        if kappa_dict is not None:
            self.kappa.update(kappa_dict)

    def compute_hierarchical_posterior(
        self,
        level_stats: Dict[str, Dict[str, float]],
        global_prior: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Combines multi-level historical summary statistics into shrunk posterior price states.

        Hierarchy resolution order (top-down shrinkage):
        Global (L5) -> Commodity+UOM (L4) -> Family+UOM (L3) -> Brand (L2) -> Model (L1) -> Identity (L0)
        """
        # L5: Global Prior
        mu_global = float(global_prior.get("weighted_log_mean", 4.0))
        var_global = float(global_prior.get("weighted_log_variance", 2.25))
        w_global = float(global_prior.get("effective_support", 1000.0))

        # L4: Commodity + UOM
        st_comm = level_stats.get("commodity_uom", {})
        w_comm = float(st_comm.get("effective_support", 0.0))
        y_comm = float(st_comm.get("weighted_log_mean", mu_global)) if w_comm > 0 else mu_global
        s2_comm = float(st_comm.get("weighted_log_variance", var_global)) if w_comm > 0 else var_global
        r_comm = w_comm / (w_comm + self.kappa["commodity_uom"]) if w_comm > 0 else 0.0
        mu_comm_post = r_comm * y_comm + (1.0 - r_comm) * mu_global
        var_comm_post = (1.0 - r_comm) * var_global + r_comm * (s2_comm / max(w_comm, 1.0)) + r_comm * (1.0 - r_comm) * ((y_comm - mu_global) ** 2)

        # L3: Family + UOM
        st_fam = level_stats.get("family_uom", {})
        w_fam = float(st_fam.get("effective_support", 0.0))
        y_fam = float(st_fam.get("weighted_log_mean", mu_comm_post)) if w_fam > 0 else mu_comm_post
        s2_fam = float(st_fam.get("weighted_log_variance", var_comm_post)) if w_fam > 0 else var_comm_post
        r_fam = w_fam / (w_fam + self.kappa["family_uom"]) if w_fam > 0 else 0.0
        mu_fam_post = r_fam * y_fam + (1.0 - r_fam) * mu_comm_post
        var_fam_post = (1.0 - r_fam) * var_comm_post + r_fam * (s2_fam / max(w_fam, 1.0)) + r_fam * (1.0 - r_fam) * ((y_fam - mu_comm_post) ** 2)

        # L2: Brand
        st_brand = level_stats.get("brand", {})
        w_brand = float(st_brand.get("effective_support", 0.0))
        y_brand = float(st_brand.get("weighted_log_mean", mu_fam_post)) if w_brand > 0 else mu_fam_post
        s2_brand = float(st_brand.get("weighted_log_variance", var_fam_post)) if w_brand > 0 else var_fam_post
        r_brand = w_brand / (w_brand + self.kappa["brand"]) if w_brand > 0 else 0.0
        mu_brand_post = r_brand * y_brand + (1.0 - r_brand) * mu_fam_post
        var_brand_post = (1.0 - r_brand) * var_fam_post + r_brand * (s2_brand / max(w_brand, 1.0)) + r_brand * (1.0 - r_brand) * ((y_brand - mu_fam_post) ** 2)

        # L1: Model / Part
        st_model = level_stats.get("model", {})
        w_model = float(st_model.get("effective_support", 0.0))
        parent_for_model = mu_brand_post if w_brand > 0 else mu_comm_post
        parent_var_for_model = var_brand_post if w_brand > 0 else var_comm_post
        y_model = float(st_model.get("weighted_log_mean", parent_for_model)) if w_model > 0 else parent_for_model
        s2_model = float(st_model.get("weighted_log_variance", parent_var_for_model)) if w_model > 0 else parent_var_for_model
        r_model = w_model / (w_model + self.kappa["model"]) if w_model > 0 else 0.0
        mu_model_post = r_model * y_model + (1.0 - r_model) * parent_for_model
        var_model_post = (1.0 - r_model) * parent_var_for_model + r_model * (s2_model / max(w_model, 1.0)) + r_model * (1.0 - r_model) * ((y_model - parent_for_model) ** 2)

        # L0: Exact Identity
        st_ident = level_stats.get("identity", {})
        w_ident = float(st_ident.get("effective_support", 0.0))
        parent_for_ident = mu_model_post if w_model > 0 else (mu_brand_post if w_brand > 0 else mu_comm_post)
        parent_var_for_ident = var_model_post if w_model > 0 else (var_brand_post if w_brand > 0 else var_comm_post)
        y_ident = float(st_ident.get("weighted_log_mean", parent_for_ident)) if w_ident > 0 else parent_for_ident
        s2_ident = float(st_ident.get("weighted_log_variance", parent_var_for_ident)) if w_ident > 0 else parent_var_for_ident
        r_ident = w_ident / (w_ident + self.kappa["identity"]) if w_ident > 0 else 0.0
        mu_ident_post = r_ident * y_ident + (1.0 - r_ident) * parent_for_ident
        var_ident_post = (1.0 - r_ident) * parent_var_for_ident + r_ident * (s2_ident / max(w_ident, 1.0)) + r_ident * (1.0 - r_ident) * ((y_ident - parent_for_ident) ** 2)

        # Composite Posterior Price State:
        # Highest fidelity non-empty level determines primary posterior
        final_posterior_state = mu_ident_post
        final_posterior_var = max(var_ident_post, 0.001)
        final_posterior_std = math.sqrt(final_posterior_var)

        # Diagnostic gaps and disagreements
        identity_vs_model_gap = float(mu_ident_post - mu_model_post)
        model_vs_family_gap = float(mu_model_post - mu_fam_post)
        family_vs_commodity_gap = float(mu_fam_post - mu_comm_post)
        total_hierarchical_disagreement = float(abs(mu_ident_post - mu_global))

        return {
            # States (log scale)
            "identity_state": float(mu_ident_post),
            "model_state": float(mu_model_post),
            "brand_state": float(mu_brand_post),
            "family_state": float(mu_fam_post),
            "commodity_state": float(mu_comm_post),
            "global_state": float(mu_global),
            "posterior_price_state": float(final_posterior_state),
            "posterior_price_natural": float(math.exp(min(final_posterior_state, 15.0))),
            # Uncertainties & Dispersions
            "posterior_variance": float(final_posterior_var),
            "posterior_uncertainty": float(final_posterior_std),
            "identity_dispersion": float(math.sqrt(s2_ident)),
            "model_dispersion": float(math.sqrt(s2_model)),
            "family_dispersion": float(math.sqrt(s2_fam)),
            "commodity_dispersion": float(math.sqrt(s2_comm)),
            # Supports & Sample Counts
            "identity_support": float(w_ident),
            "model_support": float(w_model),
            "brand_support": float(w_brand),
            "family_support": float(w_fam),
            "commodity_support": float(w_comm),
            "identity_count": int(st_ident.get("count", 0)),
            "model_count": int(st_model.get("count", 0)),
            "brand_count": int(st_brand.get("count", 0)),
            "family_count": int(st_fam.get("count", 0)),
            "commodity_count": int(st_comm.get("count", 0)),
            # Recencies (minimum days elapsed)
            "identity_recency": float(st_ident.get("min_days_elapsed", 9999.0)),
            "model_recency": float(st_model.get("min_days_elapsed", 9999.0)),
            "brand_recency": float(st_brand.get("min_days_elapsed", 9999.0)),
            "family_recency": float(st_fam.get("min_days_elapsed", 9999.0)),
            "commodity_recency": float(st_comm.get("min_days_elapsed", 9999.0)),
            # Hierarchical Gaps & Disagreement
            "identity_vs_model_gap": identity_vs_model_gap,
            "model_vs_family_gap": model_vs_family_gap,
            "family_vs_commodity_gap": family_vs_commodity_gap,
            "total_hierarchical_disagreement": total_hierarchical_disagreement,
            # Shrinkage reliabilities
            "reliability_identity": float(r_ident),
            "reliability_model": float(r_model),
            "reliability_brand": float(r_brand),
            "reliability_family": float(r_fam),
            "reliability_commodity": float(r_comm)
        }
