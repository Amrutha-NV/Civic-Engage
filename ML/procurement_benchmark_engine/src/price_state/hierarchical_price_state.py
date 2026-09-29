"""
Procurement Intelligence System — Hierarchical Bayesian Price State Model (Step 22).

Coordinates causal timeline stores across 6 hierarchy levels, computes temporal decay
weights, and evaluates empirical-Bayes posterior price states.
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

from temporal_state import CausalTimelineStore, TemporalDecayEstimator
from empirical_bayes import HierarchicalShrinkageEngine


class HierarchicalPriceStateModel:
    """
    Unified Hierarchical Bayesian / Partial-Pooling Price State Model.
    """

    def __init__(
        self,
        default_half_life: float = 180.0,
        epoch_start: str = "2009-01-01",
        kappa_dict: Optional[Dict[str, float]] = None
    ):
        self.default_half_life = default_half_life
        self.epoch_start = epoch_start
        self.epoch_ts = pd.Timestamp(epoch_start)

        self.store_identity = CausalTimelineStore("identity_key")
        self.store_model = CausalTimelineStore("model_key")
        self.store_brand = CausalTimelineStore("brand_key")
        self.store_family = CausalTimelineStore("family_uom_key")
        self.store_commodity = CausalTimelineStore("commodity_uom_key")
        self.shrinkage_engine = HierarchicalShrinkageEngine(kappa_dict)

        self._is_fitted = False

    @staticmethod
    def _create_hierarchy_keys(df: pd.DataFrame) -> pd.DataFrame:
        """Constructs standardized composite keys for each hierarchy level."""
        df_out = df.copy()

        # Level 0: Exact Identity Key
        if "identity_key" not in df_out.columns:
            df_out["identity_key"] = df_out.get("product_identity_key", "UNKNOWN")

        # Level 1: Model Key (commodity_code + '|' + normalized_model)
        models = df_out.get("normalized_model", df_out.get("model", pd.Series(index=df_out.index, dtype=object))).fillna("")
        comms = df_out.get("commodity_code", "").astype(str)
        df_out["model_key"] = np.where(models != "", comms + "|" + models.astype(str), None)

        # Level 2: Brand Key (commodity_family + '|' + normalized_brand)
        brands = df_out.get("normalized_brand", df_out.get("brand", pd.Series(index=df_out.index, dtype=object))).fillna("")
        fams = df_out.get("commodity_family", "").astype(str)
        df_out["brand_key"] = np.where(brands != "", fams + "|" + brands.astype(str), None)

        # Level 3: Family + UOM Key
        uoms = df_out.get("uom_standardized", "").astype(str)
        df_out["family_uom_key"] = fams + "|" + uoms

        # Level 4: Commodity + UOM Key
        df_out["commodity_uom_key"] = comms + "|" + uoms

        return df_out

    def fit_timelines(self, df: pd.DataFrame) -> None:
        """
        Builds chronological timeline stores for all hierarchy levels.
        Must contain training / historical observations.
        """
        df_keys = self._create_hierarchy_keys(df)

        self.store_identity.build_from_dataframe(df_keys, "identity_key", epoch_start=self.epoch_start)
        self.store_model.build_from_dataframe(df_keys, "model_key", epoch_start=self.epoch_start)
        self.store_brand.build_from_dataframe(df_keys, "brand_key", epoch_start=self.epoch_start)
        self.store_family.build_from_dataframe(df_keys, "family_uom_key", epoch_start=self.epoch_start)
        self.store_commodity.build_from_dataframe(df_keys, "commodity_uom_key", epoch_start=self.epoch_start)

        self._is_fitted = True

    fit_hierarchical_states = fit_timelines

    def estimate_query_state(
        self,
        q_row: Dict[str, Any],
        q_date: Any,
        half_life_days: Optional[float] = None,
        use_adaptive_half_life: bool = False
    ) -> Dict[str, Any]:
        """
        Estimates the hierarchical price state for a single query row at date q_date.
        Strictly uses observations with award_date < q_date.
        """
        if not self._is_fitted:
            raise RuntimeError("HierarchicalPriceStateModel must be fitted before estimating state.")

        q_dt = pd.to_datetime(q_date)
        q_epoch_day = int((q_dt - self.epoch_ts).days)
        tau = half_life_days if half_life_days is not None else self.default_half_life

        # Construct query hierarchy keys
        ident_key = q_row.get("identity_key", q_row.get("product_identity_key"))
        model_val = q_row.get("normalized_model", q_row.get("model"))
        brand_val = q_row.get("normalized_brand", q_row.get("brand"))
        comm_val = str(q_row.get("commodity_code", ""))
        fam_val = str(q_row.get("commodity_family", ""))
        uom_val = str(q_row.get("uom_standardized", ""))

        model_key = f"{comm_val}|{model_val}" if model_val and pd.notna(model_val) and str(model_val).strip() != "" else None
        brand_key = f"{fam_val}|{brand_val}" if brand_val and pd.notna(brand_val) and str(brand_val).strip() != "" else None
        family_uom_key = f"{fam_val}|{uom_val}"
        commodity_uom_key = f"{comm_val}|{uom_val}"

        # 1. Query causal histories
        h_ident = self.store_identity.query_history(ident_key, q_epoch_day)
        h_model = self.store_model.query_history(model_key, q_epoch_day)
        h_brand = self.store_brand.query_history(brand_key, q_epoch_day)
        h_fam = self.store_family.query_history(family_uom_key, q_epoch_day)
        h_comm = self.store_commodity.query_history(commodity_uom_key, q_epoch_day)
        h_glob = self.store_identity.query_global_history(q_epoch_day)

        # 2. Adaptive half-life calculation if requested
        eff_tau = tau
        if use_adaptive_half_life and h_ident is not None:
            eff_tau = TemporalDecayEstimator.compute_adaptive_half_life(
                h_ident[0], h_ident[1], q_epoch_day, base_half_life=tau
            )
        elif use_adaptive_half_life and h_model is not None:
            eff_tau = TemporalDecayEstimator.compute_adaptive_half_life(
                h_model[0], h_model[1], q_epoch_day, base_half_life=tau
            )

        # 3. Compute weighted summary statistics at each level
        st_ident = TemporalDecayEstimator.compute_weighted_statistics(
            h_ident[0] if h_ident is not None else np.empty(0),
            h_ident[1] if h_ident is not None else np.empty(0),
            q_epoch_day, half_life_days=eff_tau
        )
        st_model = TemporalDecayEstimator.compute_weighted_statistics(
            h_model[0] if h_model is not None else np.empty(0),
            h_model[1] if h_model is not None else np.empty(0),
            q_epoch_day, half_life_days=eff_tau
        )
        st_brand = TemporalDecayEstimator.compute_weighted_statistics(
            h_brand[0] if h_brand is not None else np.empty(0),
            h_brand[1] if h_brand is not None else np.empty(0),
            q_epoch_day, half_life_days=eff_tau
        )
        st_fam = TemporalDecayEstimator.compute_weighted_statistics(
            h_fam[0] if h_fam is not None else np.empty(0),
            h_fam[1] if h_fam is not None else np.empty(0),
            q_epoch_day, half_life_days=eff_tau
        )
        st_comm = TemporalDecayEstimator.compute_weighted_statistics(
            h_comm[0] if h_comm is not None else np.empty(0),
            h_comm[1] if h_comm is not None else np.empty(0),
            q_epoch_day, half_life_days=eff_tau
        )
        if not hasattr(self, "_glob_cache"):
            self._glob_cache = {}
        glob_k = (q_epoch_day, eff_tau)
        if glob_k not in self._glob_cache:
            self._glob_cache[glob_k] = TemporalDecayEstimator.compute_weighted_statistics(
                h_glob[0] if h_glob is not None else np.empty(0),
                h_glob[1] if h_glob is not None else np.empty(0),
                q_epoch_day, half_life_days=eff_tau
            )
        st_glob = self._glob_cache[glob_k]

        level_stats = {
            "identity": st_ident,
            "model": st_model,
            "brand": st_brand,
            "family_uom": st_fam,
            "commodity_uom": st_comm
        }

        # 4. Compute empirical Bayes shrinkage posterior
        posterior_dict = self.shrinkage_engine.compute_hierarchical_posterior(level_stats, st_glob)
        posterior_dict["effective_half_life_days"] = eff_tau

        return posterior_dict
