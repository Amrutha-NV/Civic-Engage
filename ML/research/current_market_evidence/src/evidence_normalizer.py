import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
INPUT = BASE / "market_evidence.csv"
OUTPUT = BASE / "evaluation" / "normalized_market_evidence.csv"

REQUIRED = [
    "item_name", "specification", "unit", "observed_price",
    "currency", "source_name", "source_url",
    "date_collected", "location", "notes"
]

def normalize_evidence():
    df = pd.read_csv(INPUT)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    df["item_name"] = df["item_name"].fillna("").astype(str).str.strip()
    df["specification"] = df["specification"].fillna("").astype(str).str.strip()
    df["unit"] = df["unit"].fillna("").astype(str).str.strip()
    df["currency"] = df["currency"].fillna("").astype(str).str.upper().str.strip()
    df["source_name"] = df["source_name"].fillna("").astype(str).str.strip()
    df["source_url"] = df["source_url"].fillna("").astype(str).str.strip()
    df["location"] = df["location"].fillna("").astype(str).str.strip()
    df["notes"] = df["notes"].fillna("").astype(str).str.strip()

    df["observed_price"] = pd.to_numeric(df["observed_price"], errors="coerce")
    df["date_collected"] = pd.to_datetime(df["date_collected"], errors="coerce")

    df["price_valid"] = df["observed_price"].notna() & (df["observed_price"] > 0)
    df["source_valid"] = df["source_name"].ne("") & df["source_url"].str.startswith(("http://", "https://"))
    df["date_valid"] = df["date_collected"].notna()
    df["item_valid"] = df["item_name"].ne("")
    df["evidence_quality"] = "LOW"

    strong = df["price_valid"] & df["source_valid"] & df["date_valid"] & df["item_valid"]
    df.loc[strong, "evidence_quality"] = "HIGH"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print(f"[OK] Normalized evidence records: {len(df)}")
    print(f"[OK] HIGH quality: {(df['evidence_quality'] == 'HIGH').sum()}")
    print(f"[OK] LOW quality: {(df['evidence_quality'] == 'LOW').sum()}")
    print(f"[OK] Output: {OUTPUT}")

if __name__ == "__main__":
    normalize_evidence()
