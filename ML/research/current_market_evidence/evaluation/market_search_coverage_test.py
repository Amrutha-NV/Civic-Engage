import sys
from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
SEARCH_DIR = ROOT / "ML" / "research" / "current_market_evidence" / "src"
DATASET = ROOT / "ML" / "research" / "procurement_benchmark_topk" / "frozen_evaluation_dataset.parquet"
OUT_DIR = ROOT / "ML" / "research" / "current_market_evidence" / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SEARCH_DIR))
from market_search import search_market

df = pd.read_parquet(DATASET)

required = ["query_index", "target_description", "target_unit_of_measure"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required dataset columns: {missing}")

# One record per procurement query, not one per candidate row.
queries = df[required].drop_duplicates("query_index").copy()
queries["target_description"] = queries["target_description"].fillna("").astype(str)
queries["target_unit_of_measure"] = queries["target_unit_of_measure"].fillna("").astype(str)

records = []
for row in queries.itertuples(index=False):
    description = row.target_description.strip()
    unit = row.target_unit_of_measure.strip()

    try:
        # Measure text-match coverage first. Currency is not available
        # in the frozen dataset, so this does not approve price comparisons.
        results = search_market(description, unit=unit or None)
        error = ""
    except Exception as exc:
        results = pd.DataFrame()
        error = f"{type(exc).__name__}: {exc}"

    records.append({
        "query_index": row.query_index,
        "target_description": description,
        "target_unit_of_measure": unit,
        "matched_evidence_count": len(results),
        "matched_evidence_items": (
            " || ".join(results["item_name"].astype(str).head(3))
            if not results.empty else ""
        ),
        "matched_evidence_currencies": (
            " || ".join(sorted(results["currency"].astype(str).str.upper().unique()))
            if not results.empty else ""
        ),
        "matched_evidence_units": (
            " || ".join(sorted(results["unit"].astype(str).unique()))
            if not results.empty else ""
        ),
        "error": error,
    })

report = pd.DataFrame(records)
output_csv = OUT_DIR / "market_search_coverage.csv"
report.to_csv(output_csv, index=False)

total = len(report)
matched = int(report["matched_evidence_count"].gt(0).sum())
failed = int(report["error"].ne("").sum())

print("=== MARKET SEARCH COVERAGE TEST ===")
print("Dataset rows:", len(df))
print("Unique queries:", total)
print("Queries with matching evidence:", matched)
print("Queries without matching evidence:", total - matched)
print("Coverage (%):", round(100 * matched / total, 2) if total else 0)
print("Query errors:", failed)
print("Currency available in frozen dataset: No")
print("Market-price ranking enabled: No")
print("Report:", output_csv)

if matched:
    print("\nSample matches:")
    print(
        report.loc[report["matched_evidence_count"].gt(0),
                   ["query_index", "target_description",
                    "matched_evidence_count", "matched_evidence_items"]]
        .head(10).to_string(index=False)
    )
