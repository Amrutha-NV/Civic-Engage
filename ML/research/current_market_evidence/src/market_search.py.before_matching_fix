import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
INPUT = BASE / "market_evidence.csv"

def search_market(item_name, max_results=10):
    df = pd.read_csv(INPUT)

    query = str(item_name).strip().lower()

    mask = (
        df["item_name"].fillna("").str.lower().str.contains(query, regex=False)
        | df["specification"].fillna("").str.lower().str.contains(query, regex=False)
    )

    results = df.loc[
        mask,
        [
            "item_name",
            "specification",
            "unit",
            "observed_price",
            "currency",
            "source_name",
            "source_url",
            "date_collected",
            "location",
            "notes"
        ]
    ].head(max_results)

    return results

if __name__ == "__main__":
    results = search_market("Dell")
    print(f"[OK] Search results: {len(results)}")
    if not results.empty:
        print(results[["item_name", "observed_price", "currency", "source_name"]].to_string(index=False))
