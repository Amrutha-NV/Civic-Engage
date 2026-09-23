"""
Procurement Intelligence System — Supervised Entity Matcher Feature Extractor (Step 15).

Extracts non-price similarity & comparability features for pairwise product comparisons:
- Identity exact matches (brand, model, MPN, part number, identity key)
- Text & semantic similarity (char n-gram Jaccard, word Jaccard, specification SVD cosine)
- Structural compatibility (commodity code, family, UOM)
- Numeric specification compatibility & conflict flags (power, voltage, RAM, storage)
- Zero Price Leakage Invariant: Price, ITM_TOT_AM, and vendor information are NEVER accessed.
"""

import os
import sys
import math
import re
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


def get_char_ngrams(text: str, n: int = 3) -> set:
    """Extracts character n-grams from text."""
    t = str(text).lower().strip()
    if len(t) < n:
        return {t}
    return {t[i:i+n] for i in range(len(t) - n + 1)}


def jaccard_sim(set_a: set, set_b: set) -> float:
    """Computes Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


class EntityMatcherFeatureExtractor:
    """Extracts pairwise comparability features between Query Record A and Candidate Record B."""

    FEATURE_NAMES = [
        # Identity Exact Matches
        "brand_exact_match",
        "brand_one_missing",
        "model_exact_match",
        "mpn_exact_match",
        "part_num_exact_match",
        "identity_key_match",
        "identity_conf_query",
        "identity_conf_cand",
        # Text & Semantic Similarity
        "char_3gram_jaccard",
        "word_overlap_jaccard",
        "spec_cosine_similarity",
        # Structural Compatibility
        "same_commodity_code",
        "same_family",
        "same_uom",
        "spec_completeness_diff",
        # Numeric Spec Compatibility & Conflicts
        "power_match",
        "voltage_match",
        "ram_match",
        "storage_match",
        "has_spec_conflict"
    ]

    def __init__(self):
        pass

    @staticmethod
    def _conf_to_num(conf_str: Any) -> float:
        s = str(conf_str).upper().strip()
        if s == "HIGH": return 3.0
        if s == "MEDIUM": return 2.0
        return 1.0

    def extract_pair_features(
        self,
        q_row: Dict[str, Any],
        c_row: Dict[str, Any],
        spec_cos_sim: float = 0.0
    ) -> List[float]:
        """
        Extracts 20 pairwise comparability features for (Query, Candidate).
        Strictly zero price fields accessed.
        """
        # 1. Identity Matches
        q_b = q_row.get("normalized_brand")
        c_b = c_row.get("normalized_brand")
        brand_match = 1.0 if (q_b and c_b and str(q_b).strip().upper() == str(c_b).strip().upper()) else 0.0
        brand_one_missing = 1.0 if ((q_b is None) ^ (c_b is None)) else 0.0

        q_m = q_row.get("normalized_model")
        c_m = c_row.get("normalized_model")
        model_match = 1.0 if (q_m and c_m and str(q_m).strip().upper() == str(c_m).strip().upper()) else 0.0

        q_mpn = q_row.get("normalized_mpn")
        c_mpn = c_row.get("normalized_mpn")
        mpn_match = 1.0 if (q_mpn and c_mpn and str(q_mpn).strip().upper() == str(c_mpn).strip().upper()) else 0.0

        q_part = q_row.get("normalized_part_number")
        c_part = c_row.get("normalized_part_number")
        part_match = 1.0 if (q_part and c_part and str(q_part).strip().upper() == str(c_part).strip().upper()) else 0.0

        q_key = q_row.get("identity_key")
        c_key = c_row.get("identity_key")
        key_match = 1.0 if (q_key and c_key and str(q_key).strip() == str(c_key).strip()) else 0.0

        q_conf = self._conf_to_num(q_row.get("identity_confidence", "LOW"))
        c_conf = self._conf_to_num(c_row.get("identity_confidence", "LOW"))

        # 2. Text Similarity
        q_txt = str(q_row.get("product_text_normalized", "")).lower()
        c_txt = str(c_row.get("product_text_normalized", "")).lower()

        q_char_set = get_char_ngrams(q_txt, n=3)
        c_char_set = get_char_ngrams(c_txt, n=3)
        char_sim = jaccard_sim(q_char_set, c_char_set)

        q_words = set(re.findall(r"\w+", q_txt))
        c_words = set(re.findall(r"\w+", c_txt))
        word_sim = jaccard_sim(q_words, c_words)

        # 3. Structural Compatibility
        q_comm = str(q_row.get("commodity_code", "")).strip()
        c_comm = str(c_row.get("commodity_code", "")).strip()
        same_comm = 1.0 if (q_comm and q_comm == c_comm) else 0.0

        q_fam = str(q_row.get("commodity_family", "")).strip()
        c_fam = str(c_row.get("commodity_family", "")).strip()
        same_fam = 1.0 if (q_fam and q_fam == c_fam) else 0.0

        q_uom = str(q_row.get("uom_standardized", "")).strip()
        c_uom = str(c_row.get("uom_standardized", "")).strip()
        same_uom = 1.0 if (q_uom and q_uom == c_uom) else 0.0

        q_comp = float(q_row.get("spec_completeness_score", 0.0))
        c_comp = float(c_row.get("spec_completeness_score", 0.0))
        comp_diff = abs(q_comp - c_comp)

        # 4. Numeric Spec Compatibility & Conflicts
        has_conflict = 0.0

        # Power
        q_p = q_row.get("power_rating")
        c_p = c_row.get("power_rating")
        if pd.notnull(q_p) and pd.notnull(c_p) and float(q_p) > 0 and float(c_p) > 0:
            if abs(float(q_p) - float(c_p)) / max(float(q_p), 1e-3) <= 0.05:
                power_match = 1.0
            else:
                power_match = 0.0
                has_conflict = 1.0
        else:
            power_match = 0.5  # Neutral when missing

        # Voltage
        q_v = q_row.get("voltage")
        c_v = c_row.get("voltage")
        if q_v and c_v and str(q_v).strip().upper() not in ("NAN", "NONE", "NULL", "") and str(c_v).strip().upper() not in ("NAN", "NONE", "NULL", ""):
            if str(q_v).strip().upper() == str(c_v).strip().upper():
                voltage_match = 1.0
            else:
                voltage_match = 0.0
                has_conflict = 1.0
        else:
            voltage_match = 0.5

        # RAM
        q_ram = q_row.get("ram_gb")
        c_ram = c_row.get("ram_gb")
        if pd.notnull(q_ram) and pd.notnull(c_ram) and float(q_ram) > 0 and float(c_ram) > 0:
            if float(q_ram) == float(c_ram):
                ram_match = 1.0
            else:
                ram_match = 0.0
                has_conflict = 1.0
        else:
            ram_match = 0.5

        # Storage
        q_stor = q_row.get("storage_gb")
        c_stor = c_row.get("storage_gb")
        if pd.notnull(q_stor) and pd.notnull(c_stor) and float(q_stor) > 0 and float(c_stor) > 0:
            if float(q_stor) == float(c_stor):
                storage_match = 1.0
            else:
                storage_match = 0.0
                has_conflict = 1.0
        else:
            storage_match = 0.5

        return [
            brand_match,
            brand_one_missing,
            model_match,
            mpn_match,
            part_match,
            key_match,
            q_conf,
            c_conf,
            char_sim,
            word_sim,
            float(spec_cos_sim),
            same_comm,
            same_fam,
            same_uom,
            comp_diff,
            power_match,
            voltage_match,
            ram_match,
            storage_match,
            has_conflict
        ]
