from pathlib import Path
import pandas as pd

root = Path.cwd()
base = root / "ML/research/current_market_evidence"
out = base / "evaluation"

coverage = pd.read_csv(out / "market_search_coverage.csv")
dataset = pd.read_parquet(root / "ML/research/procurement_benchmark_topk/frozen_evaluation_dataset.parquet")
evidence = pd.read_csv(base / "market_evidence.csv")

targets = dataset.drop_duplicates("query_index")[[
    "query_index", "target_brand", "target_model",
    "target_commodity_family"
]]

matches = coverage[coverage["matched_evidence_count"] > 0].merge(
    targets, on="query_index", how="left"
)

print("Matched queries to audit:", len(matches))

for _, r in matches.iterrows():
    print("\nQuery:", r["query_index"])
    print("Target:", r["target_description"])
    print("Target model:", r["target_model"])
    print("Target category:", r["target_commodity_family"])

    for name in str(r["matched_evidence_items"]).split(" || "):
        found = evidence[evidence["item_name"].astype(str) == name]
        print("Evidence item:", name)
        if not found.empty:
            item = found.iloc[0]
            print("  Specification:", item.get("specification", ""))
            print("  Price/currency:", item.get("observed_price"), item.get("currency"))
            print("  Unit:", item.get("unit"))

matches.to_csv(out / "market_match_audit.csv", index=False)
print("\nAudit saved to:", out / "market_match_audit.csv")
