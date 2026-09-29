"""
Procurement Intelligence System — Procurement Scope & Configuration Parser.

Extracts procurement scope types, configuration levels, bundled services,
and installation/maintenance/accessory indicators from procurement line items.
"""

import re
import math
from typing import Dict, Any, Optional, Tuple, List


class ScopeConfigurationParser:
    """Extracts procurement scope, configuration levels, and bundle indicators."""

    # Regex patterns for Scope Classification
    RE_SERVICE = re.compile(
        r"\b(?:labor|services?|consulting|professional\s+services?|hourly\s+rate|per\s+hour|inspection|testing|calibration|troubleshooting|repair\s+labor|training|preventative\s+maintenance)\b",
        re.IGNORECASE
    )

    RE_INSTALLATION = re.compile(
        r"\b(?:furnish\s+and\s+install|installation\s+included|install(?:ed)?|turnkey|commissioning|setup\s+(?:and|&)\s+deploy|field\s+install(?:ation)?|to\s+include\s+installation)\b",
        re.IGNORECASE
    )

    RE_MAINTENANCE = re.compile(
        r"\b(?:maintenance\s+agreement|support\s+contract|annual\s+maintenance|software\s+assurance|warranty\s+extension|subscriptions?|license\s+renewals?|sla|annual\s+support)\b",
        re.IGNORECASE
    )

    RE_ACCESSORY_PART = re.compile(
        r"\b(?:parts?|spares?|replacements?|brackets?|adapters?|cables?|harness(?:es)?|gaskets?|seals?|filters?|refills?|blades?|cartridges?|covers?|screws?|fittings?|hoses?|connectors?|attachments?|accessories)\b",
        re.IGNORECASE
    )

    RE_BULK = re.compile(
        r"\b(?:asphaltic?\s+concrete|hot\s+mix|diesel|fuel|gasoline|gravel|sand|crushed\s+stone|rock|soil|mulch|bulk\s+chemicals?)\b",
        re.IGNORECASE
    )

    RE_CONFIGURED = re.compile(
        r"\b(?:customized|configured|turnkey\s+system|rack\s+mount(?:ed)?|redundant\s+power|\d+\s*ports?|\d+\s*channels?|complete\s+system|integrated\s+system|multi[\s\-]channel)\b",
        re.IGNORECASE
    )

    RE_WARRANTY = re.compile(
        r"\b(?:\d+[\s\-]*(?:yr|year|mo|month)[\s\-]*warranty|extended\s+warranty|warranty\s+included|standard\s+warranty)\b",
        re.IGNORECASE
    )

    RE_CAPACITY = re.compile(
        r"\b(?:\d+(?:\.\d+)?[\s\-]*(?:gb|tb|mb|kva|kv|kw|hp|amp|amperes?|watts?|volts?|v|psi|gpm|gallons?|cf|cfm|mhz|ghz))\b",
        re.IGNORECASE
    )

    RE_INCLUDED_COMPONENTS = re.compile(
        r"\b(?:includes?|with|equipped\s+with|complete\s+with|w/|plus|and)\s+([a-zA-Z0-9\s,\/\-]+)",
        re.IGNORECASE
    )

    @classmethod
    def parse_record(cls, record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract procurement scope and configuration metadata from record."""
        texts_to_search = [
            str(record.get("COMMODITY_DESCRIPTION") or ""),
            str(record.get("EXTENDED_DESCRIPTION") or ""),
            str(record.get("CONTRACT_NAME") or ""),
            str(record.get("product_text_normalized") or ""),
            str(record.get("model") or ""),
            str(record.get("COMMODITY") or "")
        ]
        full_text = " ".join([t for t in texts_to_search if t and t != "None"]).lower()
        uom = str(record.get("UNIT_OF_MEASURE") or record.get("uom_standardized") or "").upper().strip()

        # Check indicators
        has_service = 1.0 if cls.RE_SERVICE.search(full_text) else 0.0
        has_install = 1.0 if cls.RE_INSTALLATION.search(full_text) else 0.0
        has_maint = 1.0 if cls.RE_MAINTENANCE.search(full_text) else 0.0
        has_warranty = 1.0 if cls.RE_WARRANTY.search(full_text) else 0.0
        has_accessory = 1.0 if cls.RE_ACCESSORY_PART.search(full_text) else 0.0
        has_bulk_text = 1.0 if cls.RE_BULK.search(full_text) else 0.0
        has_configured = 1.0 if cls.RE_CONFIGURED.search(full_text) else 0.0
        has_capacity = 1.0 if cls.RE_CAPACITY.search(full_text) else 0.0

        is_bulk_uom = 1.0 if uom in ["TON", "CUYD", "GAL", "MFT", "SQFT", "LNFT"] else 0.0

        # Included components count estimation
        inc_m = cls.RE_INCLUDED_COMPONENTS.search(full_text)
        included_components_count = 0
        if inc_m:
            inc_text = inc_m.group(1)
            # count comma or 'and' separated items
            items = re.split(r"[,&]|\band\b", inc_text)
            included_components_count = min(len([it for it in items if len(it.strip()) > 2]), 5)

        # Determine Primary Scope Type
        scope_confidence = "HIGH"
        if has_install == 1.0:
            scope_type = "INSTALLATION_BUNDLE"
            config_level = 4
        elif has_maint == 1.0:
            scope_type = "MAINTENANCE_BUNDLE"
            config_level = 4
        elif has_service == 1.0 and has_accessory == 0.0 and has_configured == 0.0:
            scope_type = "SERVICE"
            config_level = 4
        elif has_bulk_text == 1.0 or is_bulk_uom == 1.0:
            scope_type = "BULK_COMMODITY"
            config_level = 1
        elif has_accessory == 1.0 and has_configured == 0.0:
            scope_type = "ACCESSORY_OR_PART"
            config_level = 0
        elif has_configured == 1.0 or included_components_count >= 2:
            scope_type = "CONFIGURED_PRODUCT"
            config_level = 3
        else:
            scope_type = "BARE_PRODUCT"
            config_level = 2
            scope_confidence = "MEDIUM"

        config_confidence = "HIGH" if (has_configured or has_accessory or has_install or has_maint or has_service) else "MEDIUM"

        return {
            "scope_type": scope_type,
            "configuration_level": config_level,
            "service_bundle_indicator": has_service,
            "installation_indicator": has_install,
            "maintenance_indicator": has_maint,
            "warranty_indicator": has_warranty,
            "accessory_indicator": has_accessory,
            "capacity_indicator": has_capacity,
            "included_components_count": included_components_count,
            "scope_parse_confidence": scope_confidence,
            "configuration_parse_confidence": config_confidence,
            "vendor_code": str(record.get("VENDOR_CODE") or "").strip().upper(),
            "master_agreement": str(record.get("MASTER_AGREEMENT") or "").strip().upper(),
            "contract_name": str(record.get("CONTRACT_NAME") or "").strip().lower()
        }
