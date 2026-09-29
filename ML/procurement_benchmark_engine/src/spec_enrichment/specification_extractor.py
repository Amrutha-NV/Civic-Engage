"""
Procurement Intelligence System — Technical Specification Extractor & Normalizer (Step 23).

Extracts structured technical specifications from procurement descriptions:
- IT/Hardware: RAM (GB), Storage (GB, SSD/HDD), Ports, Speed (Gbps), PoE
- Electrical: Voltage (V), Power (Watts), kVA, Phase (1/3), Lumens, Amperage
- Mechanical: Pipe Diameter (inches), Pressure (PSI/Schedule), Material, Horsepower, Flow (GPM)
- Canonical Unit Normalization: Fraction to decimal (1/2" -> 0.5), MB to GB, Mbps to Gbps, kW to W
"""

import re
import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


# Fractional conversions
FRACTION_MAP = {
    "1/8": 0.125, "1/4": 0.25, "3/8": 0.375, "1/2": 0.5, "5/8": 0.625, "3/4": 0.75, "7/8": 0.875,
    "1-1/4": 1.25, "1-1/2": 1.5, "2-1/2": 2.5, "3-1/2": 3.5, "4-1/2": 4.5,
    "1 1/4": 1.25, "1 1/2": 1.5, "2 1/2": 2.5, "3 1/2": 3.5, "4 1/2": 4.5
}

# -----------------------------------------------------------------------------
# REGEX DEFINITIONS
# -----------------------------------------------------------------------------
R_RAM = re.compile(r"\b(\d{1,3})\s*(?:GB|MB)\s*(?:RAM|MEMORY|DDR\d?)?\b", re.IGNORECASE)
R_STORAGE = re.compile(r"\b(\d{1,4})\s*(GB|TB|MB)\s*(?:SSD|HDD|NVME|DRIVE|STORAGE|EMMC)?\b", re.IGNORECASE)
R_SSD = re.compile(r"\b(SSD|NVME|SOLID\s+STATE)\b", re.IGNORECASE)
R_HDD = re.compile(r"\b(HDD|HARD\s+DRIVE|SATA\s+HDD)\b", re.IGNORECASE)

R_PORTS = re.compile(r"\b(\d{1,3})\s*(?:-| )?(?:PORT|PT|PRT|SWITCH\s+PORTS?)\b", re.IGNORECASE)
R_SPEED = re.compile(r"\b(\d+(?:\.\d+)?)\s*(GBPS|MBPS|GBE|10GBE|GIGABIT|G)\b", re.IGNORECASE)
R_POE = re.compile(r"\b(POE\+?|POWER\s+OVER\s+ETHERNET)\b", re.IGNORECASE)

R_VOLTAGE = re.compile(r"\b(\d{2,4}(?:\/\d{2,4})?)\s*(?:V|VOLT|VAC|VDC|KV)\b", re.IGNORECASE)
R_POWER_W = re.compile(r"\b(\d+(?:\.\d+)?)\s*(W|WATT|WATTS|KW|KVA)\b", re.IGNORECASE)
R_HP = re.compile(r"\b(\d+(?:\.\d+)?|\d+\/\d+)\s*(?:HP|HORSEPOWER)\b", re.IGNORECASE)
R_PHASE = re.compile(r"\b(1|3|SINGLE|THREE)\s*(?:-| )?(?:PHASE|PH)\b", re.IGNORECASE)
R_LUMENS = re.compile(r"\b(\d{3,6})\s*(?:LM|LUMENS?)\b", re.IGNORECASE)

R_PIPE_DIA = re.compile(r"\b(\d+(?:[\s\-]\d+\/\d+|\/\d+)?|\d+\.\d+)\s*(?:IN|INCH|\"|\'\'|-IN)\b", re.IGNORECASE)
R_PRESSURE_PSI = re.compile(r"\b(\d{2,5})\s*(?:PSI|LBS|LB)\b", re.IGNORECASE)
R_SCHEDULE = re.compile(r"\b(?:SCHED(?:ULE)?|SCH)\s*(\d{2,3})\b", re.IGNORECASE)
R_CLASS = re.compile(r"\bCLASS\s*(\d{2,4})\b", re.IGNORECASE)
R_GPM = re.compile(r"\b(\d+(?:\.\d+)?)\s*GPM\b", re.IGNORECASE)

R_MATERIALS = [
    (re.compile(r"\b(PVC|POLYVINYL)\b", re.IGNORECASE), "PVC"),
    (re.compile(r"\b(DUCTILE\s+IRON|DI)\b", re.IGNORECASE), "DUCTILE_IRON"),
    (re.compile(r"\b(BRASS)\b", re.IGNORECASE), "BRASS"),
    (re.compile(r"\b(COPPER)\b", re.IGNORECASE), "COPPER"),
    (re.compile(r"\b(STAINLESS\s+STEEL|SS|304|316)\b", re.IGNORECASE), "STAINLESS_STEEL"),
    (re.compile(r"\b(CARBON\s+STEEL|STEEL)\b", re.IGNORECASE), "STEEL"),
    (re.compile(r"\b(CAST\s+IRON|CI)\b", re.IGNORECASE), "CAST_IRON"),
    (re.compile(r"\b(ALUMINUM|ALUM)\b", re.IGNORECASE), "ALUMINUM"),
]


class TechnicalSpecificationExtractor:
    """Extracts and normalizes technical specification attributes."""

    @staticmethod
    def _parse_numeric(val_str: Any) -> Optional[float]:
        if val_str is None or pd.isna(val_str):
            return None
        val_str = str(val_str).strip()
        if val_str in FRACTION_MAP:
            return FRACTION_MAP[val_str]
        try:
            m = re.search(r"(\d+(?:\.\d+)?|\d+\/\d+)", val_str)
            if m:
                s = m.group(1)
                if "/" in s:
                    parts = s.split("/")
                    return float(parts[0]) / max(float(parts[1]), 1e-4)
                return float(s)
            return float(val_str)
        except Exception:
            return None

    def extract_specifications(self, row_or_text: Any, family: Optional[str] = None) -> Dict[str, Any]:
        """
        Extracts structured numeric and categorical specifications for a line item or text.
        """
        if isinstance(row_or_text, str):
            row = {"product_text_normalized": row_or_text, "commodity_family": family or "OTHER_GOODS"}
        elif isinstance(row_or_text, dict):
            row = row_or_text
        else:
            row = {"product_text_normalized": str(row_or_text), "commodity_family": family or "OTHER_GOODS"}

        text = str(row.get("product_text_normalized", "")) + " " + str(row.get("EXTENDED_DESCRIPTION", ""))


        specs = {
            "ram_gb": None,
            "storage_gb": None,
            "is_ssd": None,
            "ports": None,
            "speed_gbps": None,
            "is_poe": None,
            "voltage_v": None,
            "power_w": None,
            "horsepower_hp": None,
            "phase": None,
            "lumens": None,
            "pipe_diameter_in": None,
            "pressure_psi": None,
            "schedule": None,
            "material": None,
            "flow_gpm": None,
        }

        # 1. Existing parsed columns if available
        if pd.notnull(row.get("ram_gb")):
            specs["ram_gb"] = self._parse_numeric(row["ram_gb"])
        if pd.notnull(row.get("storage_gb")):
            specs["storage_gb"] = self._parse_numeric(row["storage_gb"])
        if pd.notnull(row.get("power_rating")):
            specs["power_w"] = self._parse_numeric(row["power_rating"])
        if pd.notnull(row.get("voltage")):
            specs["voltage_v"] = self._parse_numeric(row["voltage"])

        # 2. RAM Extraction
        if specs["ram_gb"] is None:
            m_ram = R_RAM.search(text)
            if m_ram:
                r_val = self._parse_numeric(m_ram.group(1))
                if r_val and 1 <= r_val <= 512:
                    specs["ram_gb"] = r_val

        # 3. Storage Extraction
        if specs["storage_gb"] is None:
            m_stor = R_STORAGE.search(text)
            if m_stor:
                s_num = self._parse_numeric(m_stor.group(1))
                s_unit = m_stor.group(2).upper()
                if s_num:
                    if s_unit == "TB":
                        specs["storage_gb"] = s_num * 1024.0
                    elif s_unit == "MB":
                        specs["storage_gb"] = s_num / 1024.0
                    else:
                        specs["storage_gb"] = s_num

        if R_SSD.search(text):
            specs["is_ssd"] = 1.0
        elif R_HDD.search(text):
            specs["is_ssd"] = 0.0

        # 4. Network Ports & Speed & PoE
        m_port = R_PORTS.search(text)
        if m_port:
            p_val = self._parse_numeric(m_port.group(1))
            if p_val and p_val in [4, 8, 12, 16, 24, 48, 52, 96]:
                specs["ports"] = float(p_val)

        m_speed = R_SPEED.search(text)
        if m_speed:
            sp_val = self._parse_numeric(m_speed.group(1))
            sp_unit = m_speed.group(2).upper()
            if sp_val:
                if sp_unit in ["MBPS", "100MBPS"]:
                    specs["speed_gbps"] = sp_val / 1000.0
                elif "10" in sp_unit:
                    specs["speed_gbps"] = 10.0
                else:
                    specs["speed_gbps"] = sp_val

        if R_POE.search(text):
            specs["is_poe"] = 1.0

        # 5. Voltage & Power & HP & Phase
        if specs["voltage_v"] is None:
            m_volt = R_VOLTAGE.search(text)
            if m_volt:
                v_str = m_volt.group(1).split("/")[0]
                v_num = self._parse_numeric(v_str)
                if v_num and 12 <= v_num <= 34500:
                    specs["voltage_v"] = v_num

        if specs["power_w"] is None:
            m_pwr = R_POWER_W.search(text)
            if m_pwr:
                p_num = self._parse_numeric(m_pwr.group(1))
                p_unit = m_pwr.group(2).upper()
                if p_num:
                    if p_unit in ["KW", "KVA"]:
                        specs["power_w"] = p_num * 1000.0
                    else:
                        specs["power_w"] = p_num

        m_hp = R_HP.search(text)
        if m_hp:
            hp_val = self._parse_numeric(m_hp.group(1))
            if hp_val and 0.1 <= hp_val <= 5000:
                specs["horsepower_hp"] = hp_val

        m_phase = R_PHASE.search(text)
        if m_phase:
            ph_str = m_phase.group(1).upper()
            specs["phase"] = 1.0 if ph_str in ["1", "SINGLE"] else 3.0

        m_lum = R_LUMENS.search(text)
        if m_lum:
            specs["lumens"] = self._parse_numeric(m_lum.group(1))

        # 6. Pipe Diameter & Pressure & Material
        m_dia = R_PIPE_DIA.search(text)
        if m_dia:
            dia_val = self._parse_numeric(m_dia.group(1))
            if dia_val and 0.125 <= dia_val <= 96:
                specs["pipe_diameter_in"] = dia_val

        m_psi = R_PRESSURE_PSI.search(text)
        if m_psi:
            psi_val = self._parse_numeric(m_psi.group(1))
            if psi_val and 50 <= psi_val <= 10000:
                specs["pressure_psi"] = psi_val

        m_sch = R_SCHEDULE.search(text)
        if m_sch:
            specs["schedule"] = self._parse_numeric(m_sch.group(1))

        m_cls = R_CLASS.search(text)
        if m_cls and specs["pressure_psi"] is None:
            specs["pressure_psi"] = self._parse_numeric(m_cls.group(1))

        for pat, mat_canon in R_MATERIALS:
            if pat.search(text):
                specs["material"] = mat_canon
                break

        m_gpm = R_GPM.search(text)
        if m_gpm:
            specs["flow_gpm"] = self._parse_numeric(m_gpm.group(1))

        return specs
