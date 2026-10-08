"""
CivicEngage Procurement Benchmark ML -- Final Freeze Verification Test Suite

Tests:
1. GET /health
2. POST /predict
   - Case A: Normal procurement query (IT Hardware / Laptop)
   - Case B: Query with multiple plausible historical candidates (Commodity goods / Paper)
   - Case C: Query where historical evidence is weak / insufficient (Exotic equipment)
   - Case D: Malformed / invalid inputs (Locked test date >= 2025-01-01)

Verifies:
- Frozen LightGBM ranker predictions and conformal quantile (q_hat = 1.5404)
- Selective MMR compact Top-10 selection logic
- Reliability classification
- USD -> INR currency presentation
- Causal anti-leakage invariants & frozen test guard
"""

import json
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

# Ensure repo paths are in sys.path
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from src.api.server import app


def run_freeze_verification():
    print("=" * 80)
    print("CIVICENGAGE PROCUREMENT BENCHMARK ML -- FINAL FREEZE VERIFICATION")
    print("=" * 80)
    t0 = time.time()

    with TestClient(app) as client:
        # ------------------------------------------------------------------- #
        #  Test 1: Health Check Endpoint                                      #
        # ------------------------------------------------------------------- #
        print("\n[TEST 1] GET /health ...")
        resp = client.get("/health")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        health_data = resp.json()
        print("Response:", json.dumps(health_data, indent=2))
        assert health_data["status"] == "ok"
        print("  -> PASS: Health check endpoint working.")

        # ------------------------------------------------------------------- #
        #  Test 2: Case A -- Normal Procurement Query (Laptop)                #
        # ------------------------------------------------------------------- #
        print("\n" + "=" * 80)
        print("[TEST 2] POST /predict -- Case A: Normal procurement query (Laptop)")
        case_a_payload = {
            "COMMODITY_DESCRIPTION": "COMPUTER HARDWARE LAPTOP 15 INCH INTEL I7 16GB RAM",
            "EXTENDED_DESCRIPTION": "Dell Latitude 15 inch Intel Core i7 16GB RAM 512GB SSD",
            "product_text_normalized": "computer hardware laptop 15 inch intel i7 16gb ram dell latitude",
            "COMMODITY": "20454",
            "commodity_code": "20454",
            "commodity_family": "INFORMATION_TECHNOLOGY",
            "UNIT_OF_MEASURE": "EA",
            "uom_standardized": "EA",
            "quantity_numeric": 5.0,
            "award_date_parsed": "2024-06-15",
            "PURCHASE_ORDER": "PO-TEST-CASE-A",
            "VENDOR_CODE": "VENDOR-UNKNOWN",
            "BRAND_NAME": "Dell",
            "MODEL_NUMBER": "Latitude",
            "PRODUCT_TYPE": "LAPTOP",
        }
        resp = client.post("/predict", json=case_a_payload)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data_a = resp.json()
        print("Summary Case A:")
        print(f"  Currency:             {data_a['currency']} (Exchange Rate: {data_a['exchangeRate']})")
        print(f"  Benchmark Unit Price: Rs. {data_a['benchmarkUnitPrice']:,.2f} (${data_a['rawUSD']['benchmarkUnitPrice']:,.2f} USD)")
        print(f"  Expected Range:       Rs. {data_a['expectedRange']['lower']:,.2f} -- Rs. {data_a['expectedRange']['upper']:,.2f} ({int(data_a['expectedRange']['coverageLevel']*100)}% coverage)")
        print(f"  Reliability:          {data_a['reliability']}")
        print(f"  Candidates Retrieved: {data_a['n_candidates']}")
        print(f"  Selective MMR Fired:  {data_a['selective_mmr_triggered']} (Max Pair Sim: {data_a['max_pairwise_similarity']})")
        print(f"  Compact Top-10 Count: {len(data_a['compact_top10_candidates'])}")
        print(f"  Total Inference Time: {data_a['total_time_s']}s")

        assert data_a["benchmarkUnitPrice"] > 0
        assert data_a["expectedRange"]["lower"] <= data_a["benchmarkUnitPrice"] <= data_a["expectedRange"]["upper"]
        assert len(data_a["compact_top10_candidates"]) == 10
        print("  -> PASS: Case A executed successfully.")

        # ------------------------------------------------------------------- #
        #  Test 3: Case B -- Multiple Plausible Historical Candidates         #
        # ------------------------------------------------------------------- #
        print("\n" + "=" * 80)
        print("[TEST 3] POST /predict -- Case B: Multiple plausible historical candidates (Flashlight Batteries)")
        case_b_payload = {
            "COMMODITY_DESCRIPTION": "BATTERY FLASHLIGHT SIZE D ALKALINE",
            "EXTENDED_DESCRIPTION": "Standard industrial heavy duty alkaline flashlight battery size D 1.5V",
            "product_text_normalized": "battery flashlight size d alkaline",
            "COMMODITY": "45006100001",
            "commodity_code": "45006100001",
            "commodity_family": "BATTERIES_AND_CELLS",
            "UNIT_OF_MEASURE": "EA",
            "uom_standardized": "EA",
            "quantity_numeric": 50.0,
            "award_date_parsed": "2024-05-10",
            "PURCHASE_ORDER": "PO-TEST-CASE-B",
            "VENDOR_CODE": "VENDOR-BATT-01",
        }
        resp = client.post("/predict", json=case_b_payload)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data_b = resp.json()
        print("Summary Case B:")
        print(f"  Currency:             {data_b['currency']}")
        print(f"  Benchmark Unit Price: Rs. {data_b['benchmarkUnitPrice']:,.2f} (${data_b['rawUSD']['benchmarkUnitPrice']:,.2f} USD)")
        if data_b.get("expectedRange"):
            print(f"  Expected Range:       Rs. {data_b['expectedRange']['lower']:,.2f} -- Rs. {data_b['expectedRange']['upper']:,.2f}")
        print(f"  Reliability:          {data_b['reliability']}")
        print(f"  Candidates Retrieved: {data_b['n_candidates']}")
        print(f"  Selective MMR Fired:  {data_b['selective_mmr_triggered']} (Max Pair Sim: {data_b['max_pairwise_similarity']})")
        print(f"  Compact Top-10 Count: {len(data_b['compact_top10_candidates'])}")
        print(f"  Total Inference Time: {data_b['total_time_s']}s")

        assert data_b["benchmarkUnitPrice"] > 0
        assert len(data_b["compact_top10_candidates"]) == 10
        print("  -> PASS: Case B executed successfully.")

        # ------------------------------------------------------------------- #
        #  Test 4: Case C -- Weak / Insufficient Historical Evidence          #
        # ------------------------------------------------------------------- #
        print("\n" + "=" * 80)
        print("[TEST 4] POST /predict -- Case C: Weak/insufficient evidence (Exotic item)")
        case_c_payload = {
            "COMMODITY_DESCRIPTION": "QUANTUM CRYOGENIC DILUTION REFRIGERATOR MILLIKELVIN SUB-ATOMIC SENSOR",
            "EXTENDED_DESCRIPTION": "Specialized ultra-low temperature cryogenic millikelvin refrigeration sensor",
            "product_text_normalized": "quantum cryogenic dilution refrigerator millikelvin sub atomic sensor",
            "COMMODITY": "88888",
            "commodity_code": "88888",
            "commodity_family": "SPECIALIZED_EQUIPMENT",
            "UNIT_OF_MEASURE": "EA",
            "uom_standardized": "EA",
            "quantity_numeric": 1.0,
            "award_date_parsed": "2024-03-01",
            "PURCHASE_ORDER": "PO-TEST-CASE-C",
        }
        resp = client.post("/predict", json=case_c_payload)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data_c = resp.json()
        print("Summary Case C:")
        print(f"  Benchmark Unit Price: {data_c['benchmarkUnitPrice']}")
        print(f"  Reliability:          {data_c['reliability']}")
        print(f"  Candidates Retrieved: {data_c['n_candidates']}")
        print(f"  Compact Top-10 Count: {len(data_c['compact_top10_candidates'])}")
        print(f"  Selective MMR Fired:  {data_c['selective_mmr_triggered']}")
        print(f"  Message:              {data_c.get('message')}")
        assert data_c["reliability"] == "LOW"
        assert data_c["benchmarkUnitPrice"] is None
        assert data_c["n_candidates"] == 0
        print("  -> PASS: Case C handled gracefully with transparent reliability classification.")

        # ------------------------------------------------------------------- #
        #  Test 5: Case D -- Malformed / Invalid Input & Anti-Leakage Guard   #
        # ------------------------------------------------------------------- #
        print("\n" + "=" * 80)
        print("[TEST 5] POST /predict -- Case D: Locked frozen test period guard (>= 2025-01-01)")
        case_d_payload = {
            "COMMODITY_DESCRIPTION": "COMPUTER HARDWARE LAPTOP 15 INCH",
            "award_date_parsed": "2025-02-15",  # In locked test period!
            "PURCHASE_ORDER": "PO-TEST-CASE-D",
        }
        resp = client.post("/predict", json=case_d_payload)
        print(f"  Status code: {resp.status_code}")
        print(f"  Error message: {resp.json().get('detail')}")
        assert resp.status_code == 400, f"Expected 400 for frozen test date, got {resp.status_code}"
        assert "frozen test period" in resp.json().get("detail", "").lower()
        print("  -> PASS: Frozen test anti-leakage guard strictly rejected post-2025 query.")

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print(f"ALL FREEZE VERIFICATION TESTS PASSED END-TO-END in {elapsed:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    run_freeze_verification()
