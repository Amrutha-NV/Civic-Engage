
def detect_currency(text: str) -> str | None:
    """Return a currency only when the text explicitly identifies it."""
    import re

    text = str(text).upper()

    if "INR" in text or "₹" in text or "RUPEES" in text:
        return "INR"
    if "USD" in text or re.search(r"\$\s*\d", text):
        return "USD"

    return None


def find_compatible_market_prices(
    target_description: str,
    target_model: str = "",
    target_brand: str = "",
) -> tuple[list[float], str | None]:
    """Find evidence for the exact model; reject ambiguous currency or units."""
    import re

    model = str(target_model).strip()
    brand = str(target_brand).strip()
    description = str(target_description).strip()

    if not model or model.casefold() in {"nan", "none", "unknown"}:
        return [], None

    market_df = search_market(model, max_results=50)

    if market_df.empty:
        return [], None

    required = {
        "item_name", "specification", "observed_price", "currency", "unit"
    }
    if not required.issubset(market_df.columns):
        return [], None

    # Require the complete model identifier, not a common substring.
    model_pattern = re.compile(
        r"(?<![A-Za-z0-9])" + re.escape(model) + r"(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )

    item_text = (
        market_df["item_name"].fillna("").astype(str)
        + " "
        + market_df["specification"].fillna("").astype(str)
    )

    matched = market_df.loc[item_text.apply(
        lambda value: bool(model_pattern.search(value))
    )].copy()

    if matched.empty:
        return [], None

    # If the target identifies a brand, require that brand in the evidence.
    if brand and brand.casefold() not in {"nan", "none", "unknown"}:
        matched = matched.loc[
            (
                matched["item_name"].fillna("").astype(str)
                + " "
                + matched["specification"].fillna("").astype(str)
            ).str.contains(re.escape(brand), case=False, regex=True)
        ]

    if matched.empty:
        return [], None

    matched["observed_price"] = pd.to_numeric(
        matched["observed_price"], errors="coerce"
    )
    matched["currency"] = (
        matched["currency"].fillna("").astype(str).str.strip().str.upper()
    )
    matched["unit"] = (
        matched["unit"].fillna("").astype(str).str.strip().str.casefold()
    )
    matched = matched.loc[
        matched["observed_price"].gt(0)
        & matched["currency"].ne("")
        & matched["unit"].ne("")
    ]

    if matched.empty:
        return [], None

    currencies = matched["currency"].unique()
    units = matched["unit"].unique()

    # Do not combine market evidence from different currencies or units.
    if len(currencies) != 1 or len(units) != 1:
        return [], None

    evidence_currency = str(currencies[0])

    # The historical candidate-price currency must be identifiable too.
    target_currency = detect_currency(description)
    if target_currency != evidence_currency:
        return [], None

    return matched["observed_price"].astype(float).tolist(), evidence_currency
