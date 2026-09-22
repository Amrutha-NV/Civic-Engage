"""
Procurement Intelligence System — Temporal Price State Engine (Step 22).

Implements causal timeline indexing and exponential temporal decay estimators:
- CausalTimelineStore: High-performance binary search timeline index for historical transactions.
- TemporalDecayEstimator: Exponential temporal weighting across fixed (30d, 90d, 180d, 365d)
  and volatility-adaptive half-lives.
- Strict Anti-Leakage & Causality: Enforces historical_date < query_date for 100% of queries.
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


LN2 = math.log(2.0)


class CausalTimelineStore:
    """
    High-performance chronological index for categorical groups.
    Enables O(log N) binary search extraction of strictly historical observations.
    """

    def __init__(self, key_name: str = "group_key"):
        self.key_name = key_name
        self.groups: Dict[Any, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.global_days: Optional[np.ndarray] = None
        self.global_log_prices: Optional[np.ndarray] = None
        self.global_quantities: Optional[np.ndarray] = None

    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        key_col: Optional[str],
        date_col: str = "award_date_parsed",
        price_col: str = "target_unit_price",
        qty_col: str = "quantity_numeric",
        epoch_start: str = "2009-01-01"
    ) -> None:
        """
        Builds the chronological timeline index from a DataFrame.
        Assumes df contains training and historical records.
        """
        epoch_ts = pd.Timestamp(epoch_start)
        dates = pd.to_datetime(df[date_col])
        epoch_days = (dates - epoch_ts).dt.days.values.astype(np.int32)
        prices = np.maximum(df[price_col].values.astype(np.float64), 1e-4)
        log_prices = np.log(prices)
        quantities = np.maximum(df[qty_col].fillna(1.0).values.astype(np.float64), 1.0)

        # Sort chronologically
        sort_order = np.argsort(epoch_days, kind="mergesort")
        epoch_days = epoch_days[sort_order]
        log_prices = log_prices[sort_order]
        quantities = quantities[sort_order]

        self.global_days = epoch_days
        self.global_log_prices = log_prices
        self.global_quantities = quantities

        if key_col is not None and key_col in df.columns:
            keys = df[key_col].values[sort_order]
            group_days: Dict[Any, List[int]] = {}
            group_prices: Dict[Any, List[float]] = {}
            group_qtys: Dict[Any, List[float]] = {}
            for day, lp, qty, k in zip(epoch_days, log_prices, quantities, keys):
                if k is not None and not (isinstance(k, float) and math.isnan(k)):
                    if k not in group_days:
                        group_days[k] = []
                        group_prices[k] = []
                        group_qtys[k] = []
                    group_days[k].append(int(day))
                    group_prices[k].append(float(lp))
                    group_qtys[k].append(float(qty))
            for k, d_list in group_days.items():
                self.groups[k] = (
                    np.array(d_list, dtype=np.int32),
                    np.array(group_prices[k], dtype=np.float64),
                    np.array(group_qtys[k], dtype=np.float64)
                )

    def query_history(
        self,
        key: Any,
        query_epoch_day: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Retrieves historical (epoch_days, log_prices, quantities) strictly before query_epoch_day.
        Returns None if no historical observations exist.
        """
        if key is None or pd.isna(key) or key not in self.groups:
            return None

        days, lprices, qtys = self.groups[key]
        idx = int(np.searchsorted(days, query_epoch_day, side="left"))
        if idx <= 0:
            return None

        return (days[:idx], lprices[:idx], qtys[:idx])

    def query_global_history(
        self,
        query_epoch_day: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Retrieves all historical observations across the dataset strictly before query_epoch_day.
        """
        if self.global_days is None:
            return None

        idx = int(np.searchsorted(self.global_days, query_epoch_day, side="left"))
        if idx <= 0:
            return None

        return (self.global_days[:idx], self.global_log_prices[:idx], self.global_quantities[:idx])


class TemporalDecayEstimator:
    """
    Computes exponentially-decayed summary statistics for historical price series.
    """

    @staticmethod
    def compute_weighted_statistics(
        days: np.ndarray,
        log_prices: np.ndarray,
        query_epoch_day: int,
        half_life_days: float = 180.0,
        min_variance: float = 0.01
    ) -> Dict[str, float]:
        """
        Computes causally-weighted mean, variance, support, and recency for historical prices.
        """
        n_obs = len(days)
        if n_obs == 0:
            return {
                "count": 0,
                "effective_support": 0.0,
                "weighted_log_mean": np.nan,
                "weighted_log_variance": np.nan,
                "weighted_log_std": np.nan,
                "min_days_elapsed": np.nan,
                "weighted_days_elapsed": np.nan,
                "half_life_used": half_life_days
            }

        dt = np.maximum(query_epoch_day - days, 1.0).astype(np.float64)

        if math.isinf(half_life_days) or half_life_days > 1e7:
            weights = np.ones(n_obs, dtype=np.float64)
        else:
            weights = np.exp(-LN2 * dt / half_life_days)

        w_sum = float(np.sum(weights))
        if w_sum <= 1e-9:
            w_sum = 1e-9

        w_mean = float(np.sum(weights * log_prices) / w_sum)

        if n_obs > 1:
            w_var = float(np.sum(weights * (log_prices - w_mean) ** 2) / w_sum)
            w_var = max(w_var, min_variance)
        else:
            w_var = max(0.25, min_variance)

        min_dt = float(np.min(dt))
        w_dt = float(np.sum(weights * dt) / w_sum)

        return {
            "count": n_obs,
            "effective_support": w_sum,
            "weighted_log_mean": w_mean,
            "weighted_log_variance": w_var,
            "weighted_log_std": math.sqrt(w_var),
            "min_days_elapsed": min_dt,
            "weighted_days_elapsed": w_dt,
            "half_life_used": half_life_days
        }

    @staticmethod
    def compute_adaptive_half_life(
        days: np.ndarray,
        log_prices: np.ndarray,
        query_epoch_day: int,
        base_half_life: float = 180.0,
        prior_std: float = 0.60,
        min_half_life: float = 30.0,
        max_half_life: float = 730.0
    ) -> float:
        """
        Computes volatility-adapted half-life. Highly volatile items receive shorter half-lives;
        stable standard commodities receive longer half-lives.
        """
        n_obs = len(days)
        if n_obs <= 2:
            return base_half_life

        raw_std = float(np.std(log_prices))
        if raw_std <= 1e-4:
            return max_half_life

        scale = prior_std / (raw_std + 0.10)
        adapted = base_half_life * scale
        return float(np.clip(adapted, min_half_life, max_half_life))
