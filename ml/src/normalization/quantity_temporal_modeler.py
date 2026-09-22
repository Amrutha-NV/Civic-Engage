"""
Procurement Intelligence System — Step 30 Quantity & Temporal Modeler.

Implements:
1. Causal historical quantity elasticity estimation with empirical Bayes shrinkage:
   log(P) = alpha + beta * log(Q)
2. Historical temporal price slope and recent price window tracking.
3. Quantity tier discretization and percentile mapping.
4. Causal price transformation diagnostics.
"""

import os
import sys
import math
import bisect
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, LinearRegression


QUANTITY_TIER_BINS = [0.0, 5.0, 50.0, 500.0, float("inf")]
QUANTITY_TIER_LABELS = [0.0, 1.0, 2.0, 3.0]  # Single/Micro, Small, Medium, Bulk


def get_quantity_tier(qty: float) -> float:
    """Classify quantity into discrete order scale tiers."""
    if qty <= 5.0:
        return 0.0  # Single / Micro
    elif qty <= 50.0:
        return 1.0  # Small
    elif qty <= 500.0:
        return 2.0  # Medium
    else:
        return 3.0  # Bulk


class CausalQuantityTemporalModeler:
    """
    Maintains strictly causal historical indexes for:
    - Quantity elasticity by commodity family and product identity.
    - Historical price trends and recent transaction medians.
    """

    def __init__(self, default_shrinkage_k: float = 10.0):
        self.default_shrinkage_k = default_shrinkage_k
        self.global_beta = -0.05
        self.commodity_betas: Dict[str, float] = {}
        self.commodity_support: Dict[str, int] = {}
        
        # Identity-level timelines: identity_key -> sorted list of (timestamp, log_q, log_p, unit_price)
        self.identity_timelines: Dict[str, List[Tuple[float, float, float, float]]] = {}
        # Commodity timelines: commodity_family -> sorted list of (timestamp, log_q, log_p, unit_price)
        self.commodity_timelines: Dict[str, List[Tuple[float, float, float, float]]] = {}

    def fit_causal_history(self, df: pd.DataFrame):
        """Build sorted historical timelines from dataset."""
        df_sorted = df.sort_values(by=["award_date_parsed", "PURCHASE_ORDER"]).reset_index(drop=True)
        
        # Fit global baseline elasticity on training subset (pre-2023)
        train_mask = (df_sorted["award_date_parsed"] <= "2022-12-31")
        df_tr = df_sorted[train_mask]
        
        if len(df_tr) > 100:
            lq = np.log(np.maximum(df_tr["quantity_numeric"].values, 1e-3)).reshape(-1, 1)
            lp = np.log(np.maximum(df_tr["target_unit_price"].values, 1e-3))
            try:
                hb = HuberRegressor().fit(lq, lp)
                self.global_beta = float(np.clip(hb.coef_[0], -0.35, 0.02))
            except Exception:
                self.global_beta = -0.05

        # Commodity-level baseline elasticity
        for fam, f_df in df_tr.groupby("commodity_family"):
            if len(f_df) >= 30:
                flq = np.log(np.maximum(f_df["quantity_numeric"].values, 1e-3)).reshape(-1, 1)
                flp = np.log(np.maximum(f_df["target_unit_price"].values, 1e-3))
                try:
                    hb = HuberRegressor().fit(flq, flp)
                    self.commodity_betas[str(fam)] = float(np.clip(hb.coef_[0], -0.40, 0.05))
                except Exception:
                    self.commodity_betas[str(fam)] = self.global_beta
            else:
                self.commodity_betas[str(fam)] = self.global_beta
            self.commodity_support[str(fam)] = len(f_df)

        # Build timeline entries for fast bisect lookups
        for row in df_sorted.itertuples():
            dt = getattr(row, "award_date_parsed")
            ts = dt.timestamp() if hasattr(dt, "timestamp") else pd.to_datetime(dt).timestamp()
            q = float(getattr(row, "quantity_numeric", 1.0))
            p = float(getattr(row, "target_unit_price", 100.0))
            lq = math.log(max(q, 1e-3))
            lp = math.log(max(p, 1e-4))
            
            p_key = str(getattr(row, "product_identity_key", ""))
            fam_key = str(getattr(row, "commodity_family", "OTHER_GOODS"))

            if p_key:
                if p_key not in self.identity_timelines:
                    self.identity_timelines[p_key] = []
                self.identity_timelines[p_key].append((ts, lq, lp, p))

            if fam_key:
                if fam_key not in self.commodity_timelines:
                    self.commodity_timelines[fam_key] = []
                self.commodity_timelines[fam_key].append((ts, lq, lp, p))

    def get_causal_identity_stats(
        self,
        product_identity_key: str,
        commodity_family: str,
        query_timestamp: float
    ) -> Dict[str, Any]:
        """
        Compute causal quantity elasticity and recent price stats strictly BEFORE query_timestamp.
        """
        comm_beta = self.commodity_betas.get(str(commodity_family), self.global_beta)
        
        timeline = self.identity_timelines.get(str(product_identity_key), [])
        if not timeline:
            return {
                "identity_beta": comm_beta,
                "commodity_beta": comm_beta,
                "shrunk_beta": comm_beta,
                "beta_confidence": 0.0,
                "support_count": 0,
                "recent_support_count": 0,
                "recent_median_price": None,
                "recent_iqr_price": 0.0,
                "annualized_slope": 0.0
            }

        # Find strictly prior transactions using bisect
        timestamps = [item[0] for item in timeline]
        idx = bisect.bisect_left(timestamps, query_timestamp)
        prior_records = timeline[:idx]

        n_prior = len(prior_records)
        if n_prior == 0:
            return {
                "identity_beta": comm_beta,
                "commodity_beta": comm_beta,
                "shrunk_beta": comm_beta,
                "beta_confidence": 0.0,
                "support_count": 0,
                "recent_support_count": 0,
                "recent_median_price": None,
                "recent_iqr_price": 0.0,
                "annualized_slope": 0.0
            }

        # Recent transactions within 365 days
        ts_365 = query_timestamp - (365.25 * 86400.0)
        recent_records = [r for r in prior_records if r[0] >= ts_365]
        n_recent = len(recent_records)
        
        if n_recent > 0:
            recent_prices = [r[3] for r in recent_records]
            recent_med = float(np.median(recent_prices))
            q75, q25 = np.percentile(recent_prices, [75, 25])
            recent_iqr = float(q75 - q25)
        else:
            all_prices = [r[3] for r in prior_records]
            recent_med = float(np.median(all_prices))
            q75, q25 = np.percentile(all_prices, [75, 25])
            recent_iqr = float(q75 - q25)

        # Elasticity estimation if sufficient quantity variance
        id_beta = comm_beta
        if n_prior >= 5:
            lq_arr = np.array([r[1] for r in prior_records])
            lp_arr = np.array([r[2] for r in prior_records])
            if np.std(lq_arr) > 0.15:
                try:
                    lr = LinearRegression().fit(lq_arr.reshape(-1, 1), lp_arr)
                    raw_b = float(lr.coef_[0])
                    id_beta = float(np.clip(raw_b, -0.60, 0.10))
                except Exception:
                    id_beta = comm_beta

        # Empirical Bayes shrinkage: beta_shrunk = (N / (N + K)) * id_beta + (K / (N + K)) * comm_beta
        k = self.default_shrinkage_k
        weight_id = n_prior / (n_prior + k)
        shrunk_beta = float(weight_id * id_beta + (1.0 - weight_id) * comm_beta)
        beta_conf = float(min(1.0, math.sqrt(n_prior) / 5.0))

        # Temporal slope (annualized log-price trend)
        annualized_slope = 0.0
        if n_prior >= 3:
            t_days = np.array([(r[0] - prior_records[0][0]) / 86400.0 / 365.25 for r in prior_records])
            if np.max(t_days) > 0.5:
                try:
                    lp_arr = np.array([r[2] for r in prior_records])
                    lr_t = LinearRegression().fit(t_days.reshape(-1, 1), lp_arr)
                    annualized_slope = float(np.clip(lr_t.coef_[0], -0.50, 0.50))
                except Exception:
                    annualized_slope = 0.0

        return {
            "identity_beta": id_beta,
            "commodity_beta": comm_beta,
            "shrunk_beta": shrunk_beta,
            "beta_confidence": beta_conf,
            "support_count": n_prior,
            "recent_support_count": n_recent,
            "recent_median_price": recent_med,
            "recent_iqr_price": recent_iqr,
            "annualized_slope": annualized_slope
        }
