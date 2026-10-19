
import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
INPUT = BASE / "market_evidence.csv"

COLUMNS = [
    "item_name", "specification", "unit", "observed_price",
    "currency", "source_name", "source_url", "date_collected",
    "location", "notes",
]

STOPWORDS = {
    "the", "and", "for", "with", "from", "new", "item",
    "product", "procurement", "model", "brand", "of", "a",
    "an", "to", "in", "each", "computer", "accessories",
    "accessory", "supplies", "qty", "environmentally",
    "certified", "handheld", "microcomputers", "laptop",
    "notebook", "tablets",
}

CATEGORY_TERMS = {
    "backpack": ("backpack", "backpacks", "rucksack"),
    "dock": ("dock", "docking"),
    "monitor": ("monitor", "monitors", "display"),
    "laptop": ("laptop", "notebook", "workstation"),
    "stylus": ("apple pencil", "stylus", "digital pen"),
    "tablet": ("ipad", "tablet", "tablets"),
    "adapter": ("adapter", "charger", "power supply"),
    "keyboard_mouse": ("keyboard", "mouse", "keyboard mouse"),
    "router": ("router", "routers"),
}


def _normalize(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _tokens(value):
    return {
        token
        for token in _normalize(value).split()
        if len(token) > 1 and token not in STOPWORDS
    }


def _identifiers(value):
    text = str(value)

    ids = {
        re.sub(r"[^a-z0-9]", "", item.lower())
        for item in re.findall(
            r"(?<![A-Za-z0-9])[A-Za-z]{1,6}[- ]?\d[A-Za-z0-9-]*(?![A-Za-z0-9])",
            text,
        )
    }

    for family, number in re.findall(
        r"\b(latitude|precision)\s+(\d{3,4})\b",
        text,
        flags=re.I,
    ):
        ids.add(f"{family.lower()}{number}")

    return ids


def _category(value):
    text = _normalize(value)

    for category in (
        "stylus", "backpack", "keyboard_mouse", "adapter",
        "dock", "router", "monitor", "tablet", "laptop",
    ):
        terms = CATEGORY_TERMS[category]
        if any(_normalize(term) in text for term in terms):
            return category

    return None


def _family(value):
    text = _normalize(value)

    families = (
        "latitude", "precision", "pro max", "ultrasharp",
        "pro plus", "ipad mini", "ipad pro", "ipad a16",
        "performance dock", "smart dock", "catalyst",
    )

    for family in families:
        if _normalize(family) in text:
            return family

    return None


def _compatible_unit(value):
    unit = _normalize(value)

    aliases = {
        "each": "each",
        "ea": "each",
        "unit": "each",
        "units": "each",
        "piece": "each",
        "pieces": "each",
        "set": "set",
        "pair": "pair",
        "hour": "hour",
        "hours": "hour",
        "kg": "kg",
        "kilogram": "kg",
        "kilograms": "kg",
    }

    return aliases.get(unit, unit)


def search_market(
    item_name,
    max_results=10,
    currency=None,
    unit=None,
    min_score=0.35,
):
    """Find candidate evidence; a match is not price approval."""

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Market evidence file not found: {INPUT}"
        )

    if max_results < 1:
        raise ValueError("max_results must be at least 1")

    if not 0 <= min_score <= 1:
        raise ValueError("min_score must be between 0 and 1")

    df = pd.read_csv(INPUT)

    missing = [column for column in COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"Market evidence is missing columns: {missing}"
        )

    query = str(item_name).strip()

    output_columns = COLUMNS + [
        "match_score",
        "match_reason",
        "match_confidence",
    ]

    if not query:
        return pd.DataFrame(columns=output_columns)

    q_tokens = _tokens(query)
    q_ids = _identifiers(query)
    q_category = _category(query)
    q_family = _family(query)

    q_currency = str(currency).strip().upper() if currency else None
    q_unit = _compatible_unit(unit) if unit else None

    matches = []

    for _, row in df.iterrows():
        evidence_text = " ".join(
            str(row.get(column, "") or "")
            for column in ("item_name", "specification")
        )

        e_tokens = _tokens(evidence_text)
        e_ids = _identifiers(evidence_text)
        e_category = _category(evidence_text)
        e_family = _family(evidence_text)

        # Reject category mismatches, including unknown categories.
        if q_category != e_category:
            continue

        # If the query has model identifiers, require them in the evidence.
        if q_ids and not q_ids.issubset(e_ids):
            continue

        # Avoid matching different identified product families.
        if q_family and e_family and q_family != e_family:
            continue

        overlap = len(q_tokens & e_tokens)
        score = overlap / len(q_tokens) if q_tokens else 0.0

        if len(q_tokens) > 1 and overlap < 2 and not q_ids:
            continue

        if score < min_score:
            continue

        row_currency = str(row.get("currency", "")).strip().upper()

        if q_currency and row_currency != q_currency:
            continue

        row_unit = _compatible_unit(row.get("unit", ""))

        if q_unit and row_unit != q_unit:
            continue

        price = pd.to_numeric(
            row.get("observed_price"),
            errors="coerce",
        )

        if pd.isna(price) or price <= 0 or not row_currency or not row_unit:
            continue

        result = row[COLUMNS].to_dict()
        result["observed_price"] = float(price)
        result["match_score"] = round(score, 4)

        if q_ids:
            result["match_reason"] = "exact_model_identifier"
            result["match_confidence"] = "review_exact_model"
        else:
            result["match_reason"] = "category_and_token_match"
            result["match_confidence"] = "review_generic_match"

        matches.append(result)

    if not matches:
        return pd.DataFrame(columns=output_columns)

    return (
        pd.DataFrame(matches)
        .sort_values(
            ["match_score", "item_name"],
            ascending=[False, True],
        )
        .head(max_results)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    results = search_market(
        "Dell Pro 24 Plus Monitor P2425H"
    )

    print(f"[OK] Matching evidence records: {len(results)}")

    if not results.empty:
        print(
            results[
                [
                    "item_name",
                    "observed_price",
                    "currency",
                    "unit",
                    "match_score",
                    "match_reason",
                    "match_confidence",
                ]
            ].to_string(index=False)
        )
