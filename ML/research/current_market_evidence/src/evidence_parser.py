import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
INPUT = BASE / "market_evidence.csv"
OUTPUT = BASE / "evaluation" / "evidence_validation.csv"

df = pd.read_csv(INPUT)

checks = []

for i, row in df.iterrows():
    issues = []

    if not str(row.get("item_name", "")).strip():
        issues.append("missing_item_name")

    if not str(row.get("specification", "")).strip():
        issues.append("missing_specification")

    try:
        price = float(row.get("observed_price"))
        if price <= 0:
            issues.append("invalid_price")
    except:
        issues.append("invalid_price")

    if not str(row.get("currency", "")).strip():
        issues.append("missing_currency")

    if not str(row.get("source_name", "")).strip():
        issues.append("missing_source")

    url = str(row.get("source_url", "")).strip()
    if not url.startswith(("http://", "https://")):
        issues.append("invalid_source_url")

    if not str(row.get("date_collected", "")).strip():
        issues.append("missing_date")

    checks.append({
        "record_index": i + 1,
        "item_name": row.get("item_name", ""),
        "valid": len(issues) == 0,
        "issues": ";".join(issues) if issues else "NONE"
    })

result = pd.DataFrame(checks)
result.to_csv(OUTPUT, index=False)

print(f"[OK] Records checked: {len(result)}")
print(f"[OK] Valid records: {result['valid'].sum()}")
print(f"[OK] Records with issues: {(~result['valid']).sum()}")
print(f"[OK] Output: {OUTPUT}")
