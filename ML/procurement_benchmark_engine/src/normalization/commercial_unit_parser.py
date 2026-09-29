"""
Procurement Intelligence System — Commercial Unit & Pack Size Parser.

Extracts commercial packaging, quantity expressions, UOMs, and pack multipliers
from structured procurement fields and unstructured item descriptions.
"""

import re
import math
from typing import Dict, Any, Optional, Tuple, List


# Standardized UOM mapping dictionary
UOM_CANONICAL_MAP = {
    # Single discrete units
    "EA": ("EACH", 1.0, "EACH", False, False, True),
    "EACH": ("EACH", 1.0, "EACH", False, False, True),
    "PCS": ("EACH", 1.0, "EACH", False, False, True),
    "PC": ("EACH", 1.0, "EACH", False, False, True),
    "PIECE": ("EACH", 1.0, "EACH", False, False, True),
    "PIECES": ("EACH", 1.0, "EACH", False, False, True),
    "UNIT": ("EACH", 1.0, "EACH", False, False, True),
    "UNITS": ("EACH", 1.0, "EACH", False, False, True),
    "ITEM": ("EACH", 1.0, "EACH", False, False, True),
    "NOS": ("EACH", 1.0, "EACH", False, False, True),
    "NO": ("EACH", 1.0, "EACH", False, False, True),

    # Packaged discrete units
    "BOX": ("BOX", 1.0, "EACH", True, False, False),
    "BX": ("BOX", 1.0, "EACH", True, False, False),
    "CASE": ("CASE", 1.0, "EACH", True, False, False),
    "CS": ("CASE", 1.0, "EACH", True, False, False),
    "CART": ("CASE", 1.0, "EACH", True, False, False),
    "CARTON": ("CASE", 1.0, "EACH", True, False, False),
    "PK": ("PACK", 1.0, "EACH", True, False, False),
    "PKG": ("PACK", 1.0, "EACH", True, False, False),
    "PACK": ("PACK", 1.0, "EACH", True, False, False),
    "PACKAGE": ("PACK", 1.0, "EACH", True, False, False),
    "PKT": ("PACK", 1.0, "EACH", True, False, False),
    "PACKET": ("PACK", 1.0, "EACH", True, False, False),
    "SLV": ("SLEEVE", 1.0, "EACH", True, False, False),
    "SLEEVE": ("SLEEVE", 1.0, "EACH", True, False, False),
    "BAG": ("BAG", 1.0, "EACH", True, False, False),
    "SACK": ("BAG", 1.0, "EACH", True, False, False),
    "PAIL": ("PAIL", 1.0, "EACH", True, False, False),
    "BNDL": ("BUNDLE", 1.0, "EACH", True, False, False),
    "BUNDLE": ("BUNDLE", 1.0, "EACH", True, False, False),

    # Predetermined multi-packs
    "PR": ("PAIR", 2.0, "EACH", True, False, False),
    "PAIR": ("PAIR", 2.0, "EACH", True, False, False),
    "PAIRS": ("PAIR", 2.0, "EACH", True, False, False),
    "PRS": ("PAIR", 2.0, "EACH", True, False, False),
    "DZ": ("DOZEN", 12.0, "EACH", True, False, False),
    "DOZEN": ("DOZEN", 12.0, "EACH", True, False, False),
    "SET": ("SET", 1.0, "EACH", True, False, False),
    "SETS": ("SET", 1.0, "EACH", True, False, False),
    "KIT": ("KIT", 1.0, "EACH", True, False, False),
    "KT": ("KIT", 1.0, "EACH", True, False, False),
    "RL": ("ROLL", 1.0, "EACH", True, False, False),
    "ROLL": ("ROLL", 1.0, "EACH", True, False, False),
    "ROLLS": ("ROLL", 1.0, "EACH", True, False, False),

    # Bulk / Continuous / Materials
    "GAL": ("GALLON", 1.0, "GALLON", False, True, False),
    "GALLON": ("GALLON", 1.0, "GALLON", False, True, False),
    "DRUM": ("DRUM", 55.0, "GALLON", True, True, False),
    "CYL": ("CYLINDER", 1.0, "CYLINDER", False, True, False),
    "CYLINDER": ("CYLINDER", 1.0, "CYLINDER", False, True, False),
    "CAN": ("CAN", 1.0, "EACH", True, False, False),
    "BTL": ("BOTTLE", 1.0, "EACH", True, False, False),
    "DEWR": ("DEWAR", 1.0, "DEWAR", False, True, False),

    "FT": ("FOOT", 1.0, "FOOT", False, True, False),
    "FOOT": ("FOOT", 1.0, "FOOT", False, True, False),
    "FEET": ("FOOT", 1.0, "FOOT", False, True, False),
    "LNFT": ("FOOT", 1.0, "FOOT", False, True, False),
    "MFT": ("FOOT", 1000.0, "FOOT", True, True, False),
    "SQFT": ("SQ_FOOT", 1.0, "SQ_FOOT", False, True, False),
    "CUYD": ("CU_YARD", 1.0, "CU_YARD", False, True, False),
    "TON": ("TON", 1.0, "TON", False, True, False),
    "LB": ("POUND", 1.0, "POUND", False, True, False),
    "POUND": ("POUND", 1.0, "POUND", False, True, False),
    "LOT": ("LOT", 1.0, "LOT", False, False, False),
    "M": ("THOUSAND", 1000.0, "EACH", True, False, False),
}


class CommercialUnitParser:
    """Parser for commercial units, package quantities, and UOM normalization."""

    # Explicit slash patterns: BOX/100, PK/12, CS/50, 100/BOX, 12/PK
    RE_SLASH_COUNT = re.compile(
        r"\b(?:bx|box|cs|case|pk|pkg|pack|ctn|carton|bag|roll|set|kit|lot)/(\d{1,5})\b",
        re.IGNORECASE
    )

    RE_COUNT_SLASH = re.compile(
        r"\b(\d{1,5})/(?:bx|box|cs|case|pk|pkg|pack|ctn|carton|bag|roll|set|kit|lot)\b",
        re.IGNORECASE
    )

    # Count before unit: 100 PK, 50 CASE, 12 PACK, 100/BX, 14 PC SET
    RE_PC_SET = re.compile(
        r"\b(\d{1,4})\s*(?:pc|pcs|piece|pieces)\s+(?:set|kit|pack)\b",
        re.IGNORECASE
    )

    RE_NUM_PACK = re.compile(
        r"\b(\d{1,5})\s*(?:per\s+|/)?(?:box|bx|case|cs|pack|pkg|pk|carton|bag|roll|pair|prs|dz|dozen|sleeve|can|btl)\b",
        re.IGNORECASE
    )

    # Box/Case of N (requires 'of' or colon/dash to avoid model numbers like 'Case 580' or 'PK 22-18')
    RE_BOX_OF = re.compile(
        r"\b(?:box|case|pack|pkg|pk|carton|sleeve|slv|bag|bundle|pail|drum|can|bottle|btl|jar|tub|roll|set|kit|lot)"
        r"(?:\s+of|\s*[:=])\s*(\d{1,5})\b",
        re.IGNORECASE
    )

    RE_SET_OF = re.compile(
        r"\b(?:set|kit|pack|pair|bundle)\s+of\s+(\d{1,4})\b",
        re.IGNORECASE
    )

    RE_NUM_COUNT = re.compile(
        r"\b(\d{1,5})\s*(?:pcs|pc|ea|each|ct|count|cnt|units|items)\b",
        re.IGNORECASE
    )

    @classmethod
    def parse_uom_string(cls, uom_raw: Optional[str], uom_desc_raw: Optional[str] = None) -> Dict[str, Any]:
        """Normalize structured UOM code and description."""
        uom_clean = str(uom_raw).strip().upper() if uom_raw and not (isinstance(uom_raw, float) and math.isnan(uom_raw)) else ""
        desc_clean = str(uom_desc_raw).strip().upper() if uom_desc_raw and not (isinstance(uom_desc_raw, float) and math.isnan(uom_desc_raw)) else ""

        # Check raw code
        if uom_clean in UOM_CANONICAL_MAP:
            unit_type, pack_size, base_unit, is_pkg, is_bulk, is_single = UOM_CANONICAL_MAP[uom_clean]
            return {
                "uom_normalized": uom_clean,
                "commercial_unit_type": unit_type,
                "commercial_pack_size": pack_size,
                "base_unit": base_unit,
                "is_packaged_item": int(is_pkg),
                "is_bulk_item": int(is_bulk),
                "is_single_unit": int(is_single),
                "uom_confidence": "HIGH"
            }

        # Check description
        if desc_clean in UOM_CANONICAL_MAP:
            unit_type, pack_size, base_unit, is_pkg, is_bulk, is_single = UOM_CANONICAL_MAP[desc_clean]
            return {
                "uom_normalized": desc_clean,
                "commercial_unit_type": unit_type,
                "commercial_pack_size": pack_size,
                "base_unit": base_unit,
                "is_packaged_item": int(is_pkg),
                "is_bulk_item": int(is_bulk),
                "is_single_unit": int(is_single),
                "uom_confidence": "HIGH"
            }

        # Substring search in description
        for k, v in UOM_CANONICAL_MAP.items():
            if len(k) > 2 and k in desc_clean:
                unit_type, pack_size, base_unit, is_pkg, is_bulk, is_single = v
                return {
                    "uom_normalized": k,
                    "commercial_unit_type": unit_type,
                    "commercial_pack_size": pack_size,
                    "base_unit": base_unit,
                    "is_packaged_item": int(is_pkg),
                    "is_bulk_item": int(is_bulk),
                    "is_single_unit": int(is_single),
                    "uom_confidence": "MEDIUM"
                }

        # Default fallback
        return {
            "uom_normalized": uom_clean if uom_clean else "UNKNOWN",
            "commercial_unit_type": "OTHER",
            "commercial_pack_size": 1.0,
            "base_unit": "OTHER",
            "is_packaged_item": 0,
            "is_bulk_item": 0,
            "is_single_unit": 1 if uom_clean in ["EA", "EACH", ""] else 0,
            "uom_confidence": "LOW" if uom_clean else "NONE"
        }

    @classmethod
    def extract_pack_size_from_text(cls, text: str) -> Dict[str, Any]:
        """Extract explicit pack sizes, counts, and packaging types from text."""
        if not text or not isinstance(text, str):
            return {
                "extracted_pack_size": None,
                "extracted_unit_type": None,
                "pack_parse_confidence": "NONE",
                "extracted_pattern": None
            }

        t = text.strip()

        # 1. Explicit Slash: BOX/100, PK/12, CS/50
        m = cls.RE_SLASH_COUNT.search(t)
        if m:
            val = float(m.group(1))
            if 1 <= val <= 100000:
                raw = m.group(0).upper()
                u_type = "BOX" if "BX" in raw or "BOX" in raw else ("CASE" if "CS" in raw or "CASE" in raw else "PACK")
                return {
                    "extracted_pack_size": val,
                    "extracted_unit_type": u_type,
                    "pack_parse_confidence": "HIGH",
                    "extracted_pattern": "SLASH_COUNT"
                }

        # 2. Count before slash: 100/BOX, 12/PK
        m = cls.RE_COUNT_SLASH.search(t)
        if m:
            val = float(m.group(1))
            if 1 <= val <= 100000:
                raw = m.group(0).upper()
                u_type = "BOX" if "BX" in raw or "BOX" in raw else ("CASE" if "CS" in raw or "CASE" in raw else "PACK")
                return {
                    "extracted_pack_size": val,
                    "extracted_unit_type": u_type,
                    "pack_parse_confidence": "HIGH",
                    "extracted_pattern": "COUNT_SLASH"
                }

        # 3. Piece Set: 14 PC SET, 7 PIECE KIT
        m = cls.RE_PC_SET.search(t)
        if m:
            val = float(m.group(1))
            if 1 <= val <= 500:
                return {
                    "extracted_pack_size": val,
                    "extracted_unit_type": "SET",
                    "pack_parse_confidence": "HIGH",
                    "extracted_pattern": "PC_SET"
                }

        # 4. Box/Case of N: BOX OF 100, CASE OF 50, PACK: 12
        m = cls.RE_BOX_OF.search(t)
        if m:
            val = float(m.group(1))
            if 1 <= val <= 100000:
                raw = m.group(0).upper()
                u_type = "BOX" if "BOX" in raw else ("CASE" if "CASE" in raw or "CS" in raw else "PACK")
                return {
                    "extracted_pack_size": val,
                    "extracted_unit_type": u_type,
                    "pack_parse_confidence": "HIGH",
                    "extracted_pattern": "BOX_OF"
                }

        # 5. Set of N: SET OF 4
        m = cls.RE_SET_OF.search(t)
        if m:
            val = float(m.group(1))
            if 1 <= val <= 500:
                return {
                    "extracted_pack_size": val,
                    "extracted_unit_type": "SET",
                    "pack_parse_confidence": "HIGH",
                    "extracted_pattern": "SET_OF"
                }

        # 6. Num Pack: 100 PK, 50 CASE, 12 PACK
        m = cls.RE_NUM_PACK.search(t)
        if m:
            val = float(m.group(1))
            if 1 <= val <= 100000:
                raw = m.group(0).upper()
                u_type = "BOX" if "BOX" in raw or "BX" in raw else ("CASE" if "CASE" in raw or "CS" in raw else "PACK")
                return {
                    "extracted_pack_size": val,
                    "extracted_unit_type": u_type,
                    "pack_parse_confidence": "HIGH",
                    "extracted_pattern": "NUM_PACK"
                }

        # 7. Num Count: 100 PCS, 50 COUNT
        m = cls.RE_NUM_COUNT.search(t)
        if m:
            val = float(m.group(1))
            if 1 <= val <= 100000:
                return {
                    "extracted_pack_size": val,
                    "extracted_unit_type": "PACK",
                    "pack_parse_confidence": "MEDIUM",
                    "extracted_pattern": "NUM_COUNT"
                }

        return {
            "extracted_pack_size": None,
            "extracted_unit_type": None,
            "pack_parse_confidence": "NONE",
            "extracted_pattern": None
        }

    @classmethod
    def parse_record(cls, record: Dict[str, Any]) -> Dict[str, Any]:
        """Combine structured UOM and text fields into canonical commercial unit metadata."""
        uom_raw = record.get("UNIT_OF_MEASURE") or record.get("uom_standardized")
        uom_desc_raw = record.get("UNIT_OF_MEAS_DESC")

        uom_info = cls.parse_uom_string(uom_raw, uom_desc_raw)

        # Search across text fields
        texts_to_search = [
            str(record.get("COMMODITY_DESCRIPTION") or ""),
            str(record.get("EXTENDED_DESCRIPTION") or ""),
            str(record.get("product_text_normalized") or ""),
            str(record.get("model") or "")
        ]
        full_text = " ".join([t for t in texts_to_search if t and t != "None"])

        text_pack_info = cls.extract_pack_size_from_text(full_text)

        # Synthesize canonical pack size and unit type
        if text_pack_info["pack_parse_confidence"] in ["HIGH", "MEDIUM"] and text_pack_info["extracted_pack_size"] is not None:
            final_pack_size = float(text_pack_info["extracted_pack_size"])
            final_unit_type = text_pack_info["extracted_unit_type"] or uom_info["commercial_unit_type"]
            confidence = text_pack_info["pack_parse_confidence"]
            is_packaged = 1 if final_pack_size > 1.0 else uom_info["is_packaged_item"]
            is_single = 1 if final_pack_size == 1.0 else 0
        else:
            final_pack_size = float(uom_info["commercial_pack_size"])
            final_unit_type = uom_info["commercial_unit_type"]
            confidence = uom_info["uom_confidence"]
            is_packaged = uom_info["is_packaged_item"]
            is_single = uom_info["is_single_unit"]

        multiplier = final_pack_size if final_pack_size > 0 else 1.0

        return {
            "commercial_unit_type": final_unit_type,
            "commercial_pack_size": final_pack_size,
            "commercial_quantity_per_pack": final_pack_size,
            "commercial_pack_multiplier": multiplier,
            "base_unit": uom_info["base_unit"],
            "uom_normalized": uom_info["uom_normalized"],
            "is_packaged_item": is_packaged,
            "is_bulk_item": uom_info["is_bulk_item"],
            "is_single_unit": is_single,
            "pack_parse_confidence": confidence
        }
