"""
Procurement Intelligence System — Commercial Unit & Price Normalizer.

Implements:
1. Price basis auditing (COMMERCIAL_UNIT vs BASE_UNIT).
2. Safe base-unit price normalization with confidence tracking while preserving raw prices.
3. Query-candidate pairwise commercial unit & price-tier feature extraction.
"""

import math
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
try:
    from commercial_unit_parser import CommercialUnitParser
except ImportError:
    from src.normalization.commercial_unit_parser import CommercialUnitParser


CONFIDENCE_NUM_MAP = {
    "HIGH": 3.0,
    "MEDIUM": 2.0,
    "LOW": 1.0,
    "NONE": 0.0
}


class CommercialUnitNormalizer:
    """Normalizes commercial unit pricing and extracts pairwise packaging compatibility features."""

    @classmethod
    def audit_price_basis(cls, record: Dict[str, Any], parsed_comm: Dict[str, Any]) -> Tuple[str, str]:
        """Determine whether UNIT_PRICE represents commercial container price or base item price."""
        uom = parsed_comm.get("uom_normalized", "UNKNOWN")
        unit_type = parsed_comm.get("commercial_unit_type", "OTHER")
        pack_size = float(parsed_comm.get("commercial_pack_size", 1.0))
        conf = parsed_comm.get("pack_parse_confidence", "NONE")

        # In Austin procurement data:
        # If UOM is BOX, CASE, PK, DZ, PR, SET, ROLL, DRUM and pack_size > 1:
        # UNIT_PRICE represents the line-item unit price per commercial container (COMMERCIAL_UNIT).
        if uom in ["BOX", "BX", "CASE", "CS", "PK", "PKG", "PACK", "DZ", "DOZEN", "PR", "PAIR", "SET", "DRUM", "CART"]:
            if pack_size > 1.0 and conf in ["HIGH", "MEDIUM"]:
                return "COMMERCIAL_UNIT", conf
            elif pack_size > 1.0:
                return "COMMERCIAL_UNIT", "LOW"
            else:
                return "COMMERCIAL_UNIT", "MEDIUM"

        # If UOM is EA, EACH, PCS, FT, GAL, LB, TON:
        # If text explicitly says BOX/100, but UOM is EA:
        # Often UNIT_PRICE was entered per box or per each depending on contract line item.
        if uom in ["EA", "EACH", "PCS", "PC", "UNIT", "NOS"]:
            if pack_size > 1.0 and conf == "HIGH":
                # High-confidence pack expression in text with EA UOM
                return "COMMERCIAL_UNIT", "MEDIUM"
            else:
                return "BASE_UNIT", "HIGH"

        if parsed_comm.get("is_bulk_item", 0) == 1:
            return "BASE_UNIT", "HIGH"

        return "UNKNOWN", "LOW"

    @classmethod
    def normalize_record_pricing(cls, record: Dict[str, Any], parsed_comm: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Compute normalized base unit price while preserving original raw unit price."""
        if parsed_comm is None:
            parsed_comm = CommercialUnitParser.parse_record(record)

        raw_price = float(record.get("target_unit_price") or record.get("UNIT_PRICE") or record.get("unit_price_numeric") or 100.0)
        pack_size = float(parsed_comm.get("commercial_pack_size", 1.0))
        
        price_basis, basis_conf = cls.audit_price_basis(record, parsed_comm)

        # Normalize price per base unit only when price basis is COMMERCIAL_UNIT and pack_size > 1
        if price_basis == "COMMERCIAL_UNIT" and pack_size > 1.0 and basis_conf in ["HIGH", "MEDIUM"]:
            norm_price = raw_price / pack_size
            norm_conf = basis_conf
        else:
            norm_price = raw_price
            norm_conf = "HIGH" if price_basis == "BASE_UNIT" else "LOW"

        return {
            **parsed_comm,
            "raw_unit_price": raw_price,
            "normalized_base_unit_price": norm_price,
            "price_basis": price_basis,
            "price_normalization_confidence": norm_conf
        }

    @classmethod
    def extract_pair_commercial_features(cls, q_comm: Dict[str, Any], c_comm: Dict[str, Any],
                                         q_state: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Compute pairwise commercial unit and pack size compatibility features."""
        q_uom = q_comm.get("uom_normalized", "UNKNOWN")
        c_uom = c_comm.get("uom_normalized", "UNKNOWN")
        q_base = q_comm.get("base_unit", "OTHER")
        c_base = c_comm.get("base_unit", "OTHER")
        q_type = q_comm.get("commercial_unit_type", "OTHER")
        c_type = c_comm.get("commercial_unit_type", "OTHER")

        q_pack = max(float(q_comm.get("commercial_pack_size", 1.0)), 1e-3)
        c_pack = max(float(c_comm.get("commercial_pack_size", 1.0)), 1e-3)

        q_conf = CONFIDENCE_NUM_MAP.get(q_comm.get("pack_parse_confidence", "NONE"), 0.0)
        c_conf = CONFIDENCE_NUM_MAP.get(c_comm.get("pack_parse_confidence", "NONE"), 0.0)
        min_conf = min(q_conf, c_conf)

        uom_exact_match = 1.0 if q_uom == c_uom and q_uom != "UNKNOWN" else 0.0
        uom_base_unit_match = 1.0 if q_base == c_base and q_base != "OTHER" else 0.0
        comm_unit_match = 1.0 if q_type == c_type and q_type != "OTHER" else 0.0
        pack_size_exact_match = 1.0 if abs(q_pack - c_pack) < 1e-3 else 0.0

        pack_ratio = q_pack / c_pack
        pack_log_ratio = math.log(pack_ratio)
        abs_pack_log_diff = abs(pack_log_ratio)
        pack_diff = abs(q_pack - c_pack)

        is_cross_packaging = 1.0 if (q_pack > 1.0 and c_pack <= 1.0) or (q_pack <= 1.0 and c_pack > 1.0) else 0.0

        # Normalized price comparisons
        c_raw_price = float(c_comm.get("raw_unit_price", 100.0))
        c_norm_price = float(c_comm.get("normalized_base_unit_price", c_raw_price))

        # Relative state bounds if available
        state_mean = float(q_state.get("posterior_mean", math.log(max(c_raw_price, 1e-2)))) if q_state else math.log(max(c_raw_price, 1e-2))
        state_std = max(float(q_state.get("posterior_std", 0.5)), 0.1) if q_state else 0.5

        log_c_price = math.log(max(c_raw_price, 1e-4))
        log_c_norm_price = math.log(max(c_norm_price, 1e-4))

        z_raw_to_state = (log_c_price - state_mean) / state_std
        z_norm_to_state = (log_c_norm_price - state_mean) / state_std

        # Pack-adjusted price distance: is the candidate's normalized price closer to state than raw price?
        norm_price_improvement = abs(z_raw_to_state) - abs(z_norm_to_state)

        # Severe pack mismatch flag (e.g. 10x pack difference)
        severe_pack_mismatch = 1.0 if abs_pack_log_diff > math.log(3.0) else 0.0

        # Tier consistency score
        tier_consistency_score = 1.0 / (1.0 + abs_pack_log_diff)

        return {
            "uom_exact_match": uom_exact_match,
            "uom_base_unit_match": uom_base_unit_match,
            "commercial_unit_match": comm_unit_match,
            "pack_size_exact_match": pack_size_exact_match,
            "same_pack_size": pack_size_exact_match,
            "same_base_unit": uom_base_unit_match,
            "pack_size_ratio": float(np.clip(pack_ratio, 0.001, 1000.0)),
            "pack_size_log_ratio": float(np.clip(pack_log_ratio, -10.0, 10.0)),
            "abs_pack_size_log_diff": float(np.clip(abs_pack_log_diff, 0.0, 10.0)),
            "pack_size_difference": float(np.clip(pack_diff, 0.0, 10000.0)),
            "is_cross_packaging": is_cross_packaging,
            "pack_parse_confidence_min": min_conf,
            "severe_pack_mismatch": severe_pack_mismatch,
            "tier_consistency_score": tier_consistency_score,
            "norm_price_improvement": float(np.clip(norm_price_improvement, -10.0, 10.0)),
            "cand_norm_base_price": c_norm_price,
            "log_cand_norm_base_price": log_c_norm_price,
            "z_norm_to_state": float(np.clip(z_norm_to_state, -10.0, 10.0)),
            "is_pkg_q": float(q_comm.get("is_packaged_item", 0)),
            "is_pkg_c": float(c_comm.get("is_packaged_item", 0)),
            "is_single_q": float(q_comm.get("is_single_unit", 1)),
            "is_single_c": float(c_comm.get("is_single_unit", 1)),
            "is_bulk_q": float(q_comm.get("is_bulk_item", 0)),
            "is_bulk_c": float(c_comm.get("is_bulk_item", 0))
        }
