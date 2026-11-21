import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
INPUT = BASE / "evaluation" / "normalized_market_evidence.csv"
OUTPUT = BASE / "evaluation" / "market_benchmark.csv"

df = pd.read_csv(INPUT)

df["observed_price"] = pd.to_numeric(df["observed_price"], errors="coerce")

summary = (
    df.groupby(["item_name", "currency"], as_index=False)
      .agg(
          evidence_count=("observed_price", "count"),
          min_observed_price=("observed_price", "min"),
          median_observed_price=("observed_price", "median"),
          max_observed_price=("observed_price", "max"),
          source_count=("source_name", "nunique"),
          locations=("location", lambda x: "; ".join(sorted(set(x.dropna().astype(str)))))
      )
)

summary["evidence_quality"] = summary.apply(
    lambda r: "HIGH" if r["evidence_count"] >= 1 and r["source_count"] >= 1 else "LOW",
    axis=1
)

summary.to_csv(OUTPUT, index=False)

print(f"[OK] Benchmark summaries: {len(summary)}")
print(f"[OK] HIGH quality summaries: {(summary['evidence_quality'] == 'HIGH').sum()}")
print(f"[OK] Output: {OUTPUT}")
