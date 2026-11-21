import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

files = {
    "raw_evidence": BASE / "market_evidence.csv",
    "normalized_evidence": BASE / "evaluation" / "normalized_market_evidence.csv",
    "validation": BASE / "evaluation" / "evidence_validation.csv",
    "benchmark": BASE / "evaluation" / "market_benchmark.csv"
}

print("=== PERSON 3 MARKET EVIDENCE TEST ===")

raw = pd.read_csv(files["raw_evidence"])
normalized = pd.read_csv(files["normalized_evidence"])
validation = pd.read_csv(files["validation"])
benchmark = pd.read_csv(files["benchmark"])

checks = {
    "raw_records": len(raw),
    "normalized_records": len(normalized),
    "validation_records": len(validation),
    "valid_records": int(validation["valid"].sum()),
    "benchmark_records": len(benchmark)
}

for name, value in checks.items():
    print(f"[OK] {name}: {value}")

if len(raw) != 15:
    raise RuntimeError("Raw evidence record count is not 15.")

if len(normalized) != len(raw):
    raise RuntimeError("Normalization changed record count.")

if len(validation) != len(raw):
    raise RuntimeError("Validation record count mismatch.")

if not validation["valid"].all():
    raise RuntimeError("Validation contains invalid records.")

if len(benchmark) == 0:
    raise RuntimeError("Benchmark output is empty.")

print("[OK] All Person 3 evidence checks PASSED")
print("[OK] Current-market evidence pipeline is operational")
