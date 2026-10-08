"""Metrics evaluated only after a candidate rank has been selected."""

from __future__ import annotations

from typing import Any

import pandas as pd


def absolute_percentage_error(selected_price: pd.Series, actual_price: pd.Series) -> pd.Series:
    """Return APE in percent; target prices must be strictly positive."""
    if (actual_price <= 0).any():
        raise ValueError("actual_unit_price must be strictly positive for APE evaluation")
    return (selected_price.sub(actual_price).abs().div(actual_price)).mul(100.0)


def selection_metrics(selected_price: pd.Series, actual_price: pd.Series) -> dict[str, Any]:
    """Compute the frozen protocol's price-selection metrics."""
    if selected_price.isna().any():
        raise ValueError("selection metrics require a price for every evaluated query")

    ape = absolute_percentage_error(selected_price, actual_price)
    absolute_error = selected_price.sub(actual_price).abs()
    total = int(len(ape))

    def accuracy(threshold: float) -> dict[str, float | int]:
        hits = int((ape <= threshold).sum())
        return {"hits": hits, "total": total, "percentage": round(hits / total * 100.0, 6)}

    return {
        "accuracy_at_5pct": accuracy(5.0),
        "accuracy_at_10pct": accuracy(10.0),
        "accuracy_at_20pct": accuracy(20.0),
        "mae": round(float(absolute_error.mean()), 6),
        "mdape": round(float(ape.median()), 6),
        "mean_ape": round(float(ape.mean()), 6),
    }
