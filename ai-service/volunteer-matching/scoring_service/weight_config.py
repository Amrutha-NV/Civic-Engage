from __future__ import annotations

from typing import Dict


CATEGORY_WEIGHTS: Dict[str, Dict[str, float]] = {
    "medical": {
        "similarity": 0.35,
        "business": 0.65,
        "attendance": 0.20,
        "availability": 0.15,
        "social_impact": 0.25,
        "verification": 0.20,
        "max_campaign": 0.05,
    },
    "environmental": {
        "similarity": 0.45,
        "business": 0.55,
        "attendance": 0.15,
        "availability": 0.10,
        "social_impact": 0.20,
        "verification": 0.10,
        "max_campaign": 0.10,
    },
    "education": {
        "similarity": 0.40,
        "business": 0.60,
        "attendance": 0.15,
        "availability": 0.15,
        "social_impact": 0.25,
        "verification": 0.15,
        "max_campaign": 0.10,
    },
}


def get_weights_for_category(category: str) -> Dict[str, float]:
    return CATEGORY_WEIGHTS.get(category.lower(), CATEGORY_WEIGHTS["environmental"])
