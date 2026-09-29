"""
Procurement Intelligence System — Product Identity Parser & Normalizer (Step 23).

Parses raw & normalized procurement line descriptions into structured identity fields:
- brand / manufacturer
- model / model_family / series
- part_number / MPN / SKU
- product_type / variant / generation
- enriched_identity_key & model_family_key

Strict Anti-Collision Invariant:
Distinct hardware capacities and models (e.g., Dell Latitude 5420 vs 5520,
Cisco Meraki MS250-48FP vs MS250-24P) are strictly separated.
"""

import re
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd


# -----------------------------------------------------------------------------
# 1. CANONICAL BRAND ALIASES & RECOGNITION
# -----------------------------------------------------------------------------
BRAND_PATTERNS = [
    (r"\b(?:HEWLETT\s+PACKARD|HEWLETT-PACKARD|HP\s+INC|HP\s+COMPANY|HP)\b", "HP"),
    (r"\b(?:DELL\s+MARKETING|DELL\s+COMPUTER|DELL\s+INC|DELL)\b", "DELL"),
    (r"\b(?:CISCO\s+SYSTEMS|CISCO\s+MERAKI|CISCO|MERAKI)\b", "CISCO"),
    (r"\b(?:GENERAL\s+ELECTRIC|GE\s+ENERGY|GE\s+LIGHTING|GE)\b", "GE"),
    (r"\b(?:SQUARE\s+D|SQUARED)\b", "SQUARE D"),
    (r"\b(?:SCHNEIDER\s+ELECTRIC|SCHNEIDER)\b", "SCHNEIDER"),
    (r"\b(?:THOMAS\s*&\s*BETTS|THOMAS\s+AND\s+BETTS|T\s*&\s*B)\b", "THOMAS & BETTS"),
    (r"\b(?:CROUSE[\s\-]+HINDS|COOPER\s+CROUSE[\s\-]+HINDS)\b", "CROUSE HINDS"),
    (r"\b(?:CUTLER[\s\-]+HAMMER|EATON[\s\/]+CUTLER[\s\-]+HAMMER|EATON)\b", "EATON"),
    (r"\b(?:BRIGGS\s*&\s*STRATTON|BRIGGS\s+AND\s+STRATTON)\b", "BRIGGS & STRATTON"),
    (r"\b(?:KLEIN\s+TOOLS|KLEIN)\b", "KLEIN TOOLS"),
    (r"\b(?:TRIPP\s+LITE|TRIPPLITE)\b", "TRIPP LITE"),
    (r"\b(?:FORD\s+METER\s+BOX|FORD\s+METER)\b", "FORD METER BOX"),
    (r"\b(?:MUELLER\s+CO|MUELLER\s+COMPANY|MUELLER)\b", "MUELLER"),
    (r"\b(?:VICTAULIC\s+CO|VICTAULIC\s+COMPANY|VICTAULIC)\b", "VICTAULIC"),
    (r"\b(?:MOTOROLA\s+SOLUTIONS|MOTOROLA)\b", "MOTOROLA"),
    (r"\b(?:3M\s+COMPANY|3M\s+CORP|3M)\b", "3M"),
    (r"\b(?:CATERPILLAR\s+INC|CATERPILLAR|CAT)\b", "CATERPILLAR"),
    (r"\b(?:CUMMINS\s+ENGINE|CUMMINS\s+POWER|CUMMINS)\b", "CUMMINS"),
    (r"\b(?:TYLER\s+PIPE|TYLER\s+UNION|TYLER)\b", "TYLER"),
    (r"\b(?:KENNEDY\s+VALVE|KENNEDY)\b", "KENNEDY"),
    (r"\b(?:BADGER\s+METER|BADGER)\b", "BADGER"),
    (r"\b(?:SENSUS\s+METERING|SENSUS)\b", "SENSUS"),
    (r"\b(?:NEPTUNE\s+TECHNOLOGY|NEPTUNE)\b", "NEPTUNE"),
    (r"\b(?:NIBCO\s+INC|NIBCO)\b", "NIBCO"),
    (r"\b(?:CHEVROLET|CHEVY)\b", "CHEVROLET"),
    (r"\b(?:AC\s*DELCO|ACDELCO)\b", "ACDELCO"),
    (r"\b(?:MOTORCRAFT)\b", "MOTORCRAFT"),
    (r"\b(?:LENOVO|THINKPAD|THINKCENTRE)\b", "LENOVO"),
    (r"\b(?:APPLE|MACBOOK|IPAD)\b", "APPLE"),
    (r"\b(?:SAMSUNG)\b", "SAMSUNG"),
    (r"\b(?:ZEBRA)\b", "ZEBRA"),
    (r"\b(?:EPSON)\b", "EPSON"),
    (r"\b(?:CANON)\b", "CANON"),
    (r"\b(?:BROTHER)\b", "BROTHER"),
    (r"\b(?:PANASONIC|TOUGHBOOK)\b", "PANASONIC"),
    (r"\b(?:SONY)\b", "SONY"),
    (r"\b(?:LOGITECH)\b", "LOGITECH"),
    (r"\b(?:APC|AMERICAN\s+POWER\s+CONVERSION)\b", "APC"),
    (r"\b(?:XEROX)\b", "XEROX"),
    (r"\b(?:RICOH)\b", "RICOH"),
    (r"\b(?:LEXMARK)\b", "LEXMARK"),
    (r"\b(?:FORTINET)\b", "FORTINET"),
    (r"\b(?:ARUBA)\b", "ARUBA"),
    (r"\b(?:SIEMENS)\b", "SIEMENS"),
    (r"\b(?:ABB)\b", "ABB"),
    (r"\b(?:HUBBELL)\b", "HUBBELL"),
    (r"\b(?:LEVITON)\b", "LEVITON"),
    (r"\b(?:LUTRON)\b", "LUTRON"),
    (r"\b(?:SYLVANIA|OSRAM)\b", "SYLVANIA"),
    (r"\b(?:PHILIPS|SIGNIFY)\b", "PHILIPS"),
    (r"\b(?:KOHLER)\b", "KOHLER"),
    (r"\b(?:HONDA)\b", "HONDA"),
    (r"\b(?:GENERAC)\b", "GENERAC"),
    (r"\b(?:DEWALT)\b", "DEWALT"),
    (r"\b(?:MILWAUKEE)\b", "MILWAUKEE"),
    (r"\b(?:BOSCH)\b", "BOSCH"),
    (r"\b(?:FLUKE)\b", "FLUKE"),
    (r"\b(?:HACH)\b", "HACH"),
    (r"\b(?:CARRIER)\b", "CARRIER"),
    (r"\b(?:TRANE)\b", "TRANE"),
    (r"\b(?:YORK)\b", "YORK"),
]

COMPILED_BRANDS = [(re.compile(pat, re.IGNORECASE), canon) for pat, canon in BRAND_PATTERNS]


# -----------------------------------------------------------------------------
# 2. COMMON PRODUCT TYPES
# -----------------------------------------------------------------------------
PRODUCT_TYPE_PATTERNS = [
    (r"\b(?:laptop|notebook|ultrabook|macbook)\b", "LAPTOP"),
    (r"\b(?:desktop|tower|workstation|microcomputer|optiplex|sff|thinkcentre)\b", "DESKTOP"),
    (r"\b(?:server|poweredge|proliant|blade)\b", "SERVER"),
    (r"\b(?:switch|switches|catalyst|meraki\s+ms\d+)\b", "NETWORK_SWITCH"),
    (r"\b(?:router|firewall|access\s+point|ap|wap|meraki\s+mr\d+)\b", "ROUTER_AP"),
    (r"\b(?:printer|copier|mfp|all-in-one|laserjet|deskjet)\b", "PRINTER"),
    (r"\b(?:transformer|padmount|substation)\b", "TRANSFORMER"),
    (r"\b(?:pump|motor|centrifugal\s+pump|submersible\s+pump)\b", "PUMP_MOTOR"),
    (r"\b(?:valve|ball\s+valve|gate\s+valve|check\s+valve|butterfly\s+valve|plug\s+valve)\b", "VALVE"),
    (r"\b(?:pipe|tubing|hose|conduit)\b", "PIPE"),
    (r"\b(?:fitting|coupling|elbow|tee|adapter|nipple|bushing|flange|plug)\b", "FITTING"),
    (r"\b(?:light|fixture|luminaire|ballast|lamp|bulb|led|floodlight)\b", "LIGHTING"),
    (r"\b(?:meter|watt-hour|flowmeter|kwh)\b", "METER"),
    (r"\b(?:chemical|reagent|solvent|chlorine|acid|polymer)\b", "CHEMICAL"),
    (r"\b(?:tire|wheel|battery|filter|brake|alternator|starter)\b", "VEHICLE_PART"),
    (r"\b(?:uniform|shirt|pant|jacket|boot|glove|vest|hat)\b", "APPAREL"),
    (r"\b(?:chair|desk|table|cabinet|shelf|bookcase)\b", "FURNITURE"),
]

COMPILED_PRODUCT_TYPES = [(re.compile(pat, re.IGNORECASE), canon) for pat, canon in PRODUCT_TYPE_PATTERNS]


# -----------------------------------------------------------------------------
# 3. EXPLICIT MPN / PART NUMBER / MODEL REGEXES
# -----------------------------------------------------------------------------
R_MPN = re.compile(
    r"\b(?:part\s*#|part\s*no\.?|part\s*number|p\/n|p\.n\.|pn|mfg\s*#|mfg\s*part\s*#|mfr\s*#|cat\s*#|catalog\s*#|item\s*#)\s*[:\-=\/]?\s*([a-zA-Z0-9\-\/\.]{3,35})\b",
    re.IGNORECASE
)

R_MODEL = re.compile(
    r"\b(?:model\s*#|model\s*no\.?|model\s*number|model)\s*[:\-=\/]?\s*([a-zA-Z0-9\-\/\.]{3,35})\b",
    re.IGNORECASE
)

# Hardware model codes (e.g., MS250-48FP, C9300-48P, Latitude 5420, Optiplex 5000)
R_FAMILY_MODEL = re.compile(
    r"\b(Latitude|Optiplex|Precision|ThinkPad|ThinkCentre|PowerEdge|ProLiant|Catalyst|Meraki|LaserJet|DeskJet)\s+([a-zA-Z0-9\-]{3,15})\b",
    re.IGNORECASE
)

R_ALPHANUM_MODEL = re.compile(
    r"\b([A-Z]{1,5}[0-9]{2,5}[A-Z0-9\-]*(?:-[0-9]{1,3}[A-Z0-9]*)?)\b"
)


class ProductIdentityParser:
    """Structured product identity parsing & normalization engine."""

    @staticmethod
    def normalize_token(token: Optional[str]) -> Optional[str]:
        if not token or pd.isna(token):
            return None
        cleaned = re.sub(r"[\s\-_/]+", "-", str(token).strip().upper())
        cleaned = re.sub(r"^[^A-Z0-9]+|[^A-Z0-9]+$", "", cleaned)
        return cleaned if len(cleaned) >= 2 else None

    def parse_product_identity(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses full structured product identity from row text and metadata.
        """
        text = str(row.get("product_text_normalized", "")) + " " + str(row.get("EXTENDED_DESCRIPTION", ""))

        # 1. Extract Brand
        brand = row.get("normalized_brand")
        if not brand or pd.isna(brand):
            for pat, canon in COMPILED_BRANDS:
                if pat.search(text):
                    brand = canon
                    break

        # 2. Extract Product Type
        prod_type = "OTHER"
        for pat, canon in COMPILED_PRODUCT_TYPES:
            if pat.search(text):
                prod_type = canon
                break

        # 3. Extract MPN / Part Number
        mpn = row.get("normalized_mpn") or row.get("normalized_part_number")
        if not mpn or pd.isna(mpn):
            m_mpn = R_MPN.search(text)
            if m_mpn:
                mpn = self.normalize_token(m_mpn.group(1))

        # 4. Extract Model & Model Family
        model = row.get("normalized_model")
        model_family = None
        series = None
        variant = None

        m_fam = R_FAMILY_MODEL.search(text)
        if m_fam:
            series = m_fam.group(1).upper()
            m_num = m_fam.group(2).upper()
            model = f"{series} {m_num}"
            # Model family: e.g. Latitude 5000 for 5420, Optiplex 5000 for 5090
            num_match = re.search(r"(\d{2,4})", m_num)
            if num_match:
                n_str = num_match.group(1)
                if len(n_str) == 4:
                    model_family = f"{series} {n_str[0]}000"
                else:
                    model_family = f"{series} {n_str}"
            else:
                model_family = series

        if not model or pd.isna(model):
            m_mod = R_MODEL.search(text)
            if m_mod:
                model = self.normalize_token(m_mod.group(1))

        # Variant detection (e.g. SFF, Micro, PoE, 48FP, 24P)
        if re.search(r"\b(SFF|SMALL\s+FORM\s+FACTOR)\b", text, re.I):
            variant = "SFF"
        elif re.search(r"\b(MICRO|USFF|TINY)\b", text, re.I):
            variant = "MICRO"
        elif re.search(r"\b(TOWER|MT)\b", text, re.I):
            variant = "TOWER"

        # Construct Keys
        c_code = str(row.get("commodity_code", "")).strip()
        fam = str(row.get("commodity_family", "")).strip()
        uom = str(row.get("uom_standardized", "")).strip()

        norm_brand = self.normalize_token(brand)
        norm_model = self.normalize_token(model)
        norm_mpn = self.normalize_token(mpn)
        norm_fam_model = self.normalize_token(model_family)

        # Enriched Identity Key: MPN if present, else Brand + Model, else Commodity + Family
        if norm_mpn:
            enriched_key = f"{c_code}|MPN:{norm_mpn}|{uom}"
            conf = "HIGH"
        elif norm_brand and norm_model:
            enriched_key = f"{c_code}|{norm_brand}|{norm_model}|{uom}"
            conf = "HIGH"
        elif norm_brand:
            enriched_key = f"{fam}|{norm_brand}|{prod_type}|{uom}"
            conf = "MEDIUM"
        else:
            enriched_key = f"{c_code}|{prod_type}|{uom}"
            conf = "LOW"

        return {
            "parsed_brand": norm_brand,
            "parsed_model": norm_model,
            "parsed_model_family": norm_fam_model,
            "parsed_mpn": norm_mpn,
            "parsed_product_type": prod_type,
            "parsed_variant": variant,
            "enriched_identity_key": enriched_key,
            "identity_confidence": conf
        }

    def parse_product_text(self, text: str, family: str = "OTHER_GOODS") -> Dict[str, Any]:
        """Convenience method to parse text and commodity family directly."""
        dummy_row = {
            "product_text_normalized": text,
            "commodity_family": family,
            "commodity_code": "00000",
            "uom_standardized": "EA"
        }
        return self.parse_product_identity(dummy_row)

