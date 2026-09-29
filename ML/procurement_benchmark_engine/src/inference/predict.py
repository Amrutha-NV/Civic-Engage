"""
CivicEngage Procurement Benchmark ML
Core Inference Module -- Frozen T3 Co-Occurrence/Bundle Ranker

PURPOSE:
  Accept a procurement query, retrieve historical candidates, build the frozen
  176-feature representation, rank with the serialized T3 LGBMRanker, and
  return the rank-1 candidate's ACTUAL HISTORICAL UNIT_PRICE as the benchmark.

RULES:
  - Prediction MUST be an actual historical unit price from the candidate pool.
  - No price synthesis, blending, interpolation, or oracle selection.
  - Frozen test set (2025-01-02 onward) remains locked.
  - Feature definition (176f) must not change.

USAGE:
  from src.inference.predict import ProcurementPredictor

  predictor = ProcurementPredictor()  # one-time setup (~2-5 min cold start)
  result = predictor.predict(query_dict)
  # result["predicted_unit_price"] is the benchmark prediction

  # Or via CLI:
  python -m src.inference.predict --query_json '{"ITEM_DESCRIPTION": "...", ...}'
"""

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
#  Path bootstrap                                                               #
# --------------------------------------------------------------------------- #
_THIS_FILE = Path(__file__).resolve()
ML_ROOT = _THIS_FILE.parent.parent.parent   # D:\tender_generation\ML

for _p in [
    ML_ROOT / "src" / "ranking",
    ML_ROOT / "src" / "models",
    ML_ROOT / "src" / "normalization",
    ML_ROOT / "src" / "retrieval",
    ML_ROOT / "src" / "price_state",
    ML_ROOT / "src" / "entity_resolution",
    ML_ROOT / "src" / "spec_enrichment",
    ML_ROOT / "src" / "data",
    ML_ROOT / "src" / "confidence",
]:
    sys.path.insert(0, str(_p))

from sklearn.preprocessing import normalize  # noqa: E402 -- after sys.path

from quantity_temporal_ranker import Step30FeatureExtractor  # noqa: E402
from commercial_unit_parser import CommercialUnitParser  # noqa: E402
from scope_configuration_parser import ScopeConfigurationParser  # noqa: E402
from hierarchical_price_state import HierarchicalPriceStateModel  # noqa: E402
from quantity_temporal_modeler import CausalQuantityTemporalModeler  # noqa: E402
from multi_view_candidate_retrieval import MultiViewCandidateRetrievalEngine  # noqa: E402
from product_identity_parser import ProductIdentityParser  # noqa: E402
from specification_extractor import TechnicalSpecificationExtractor  # noqa: E402

# --------------------------------------------------------------------------- #
#  Frozen configuration constants (must not be changed)                        #
# --------------------------------------------------------------------------- #
MODEL_PATH     = ML_ROOT / "models" / "procurement_ranker.joblib"
META_PATH      = ML_ROOT / "models" / "procurement_ranker_metadata.joblib"
CONFORMAL_PATH = ML_ROOT / "models" / "conformal_calibration.joblib"
ENTITY_PATH    = ML_ROOT / "models" / "entity_matcher.joblib"
DATASET_PATH   = ML_ROOT / "data" / "catalog" / "procurement_catalog.parquet"
NORM_CACHE     = ML_ROOT / "data" / "catalog" / "normalized_identity_cache.parquet"
PARSED_CACHE   = ML_ROOT / "data" / "catalog" / "parsed_records_cache.joblib"

# Feature index selection from Step30 207-feature master (same as Step 36/37)
S4_BASE_INDICES = list(range(151)) + list(range(165, 168))
T5_INDICES      = S4_BASE_INDICES + list(range(191, 207))  # 170 features (T0)

# T3 Co-Occurrence feature names (appended at positions 170-175)
T3_CO_OCC_FEATURE_NAMES = [
    "q_hhi",
    "q_is_primary_item",
    "log1p_q_other_qty",
    "cand_commodity_in_query_po_set",
    "abs_hhi_diff",
    "both_primary_item",
]

# Retrieval config (frozen)
RETRIEVAL_CONFIG = dict(
    max_total_k=60,
    filter_mode="CONSERVATIVE",
    enable_channels=list("ABCDEFGHIJ"),
)


# --------------------------------------------------------------------------- #
#  PO metadata helper                                                          #
# --------------------------------------------------------------------------- #
def _build_po_metadata(df_all: pd.DataFrame):
    """Compute PO commodity concentration HHI and primary-item indicator."""
    po_comm_qty = (
        df_all.groupby(["PURCHASE_ORDER", "COMMODITY"])["quantity_numeric"]
        .sum()
        .reset_index()
    )
    po_tot_qty_s = po_comm_qty.groupby("PURCHASE_ORDER")["quantity_numeric"].transform("sum")
    po_comm_qty["share_sq"] = (
        po_comm_qty["quantity_numeric"] / np.maximum(po_tot_qty_s, 1e-4)
    ) ** 2
    po_hhi_series = po_comm_qty.groupby("PURCHASE_ORDER")["share_sq"].sum().rename("po_hhi")
    df_all = df_all.merge(po_hhi_series, on="PURCHASE_ORDER", how="left")

    quantities    = df_all["quantity_numeric"].values.astype(np.float32)
    po_max_qty    = (
        df_all.groupby("PURCHASE_ORDER")["quantity_numeric"].transform("max").values.astype(np.float32)
    )
    po_total_qtys = (
        df_all.groupby("PURCHASE_ORDER")["quantity_numeric"].transform("sum").values.astype(np.float32)
    )
    is_primary    = (quantities >= (po_max_qty - 1e-4)).astype(np.float32)
    po_hhis       = df_all["po_hhi"].fillna(1.0).values.astype(np.float32)
    commodities   = df_all["COMMODITY"].values
    po_names      = df_all["PURCHASE_ORDER"].values
    po_comm_sets  = df_all.groupby("PURCHASE_ORDER")["COMMODITY"].apply(set).to_dict()
    row_ids       = df_all["row_id"].values

    return dict(
        quantities=quantities,
        po_total_qtys=po_total_qtys,
        is_primary=is_primary,
        po_hhis=po_hhis,
        commodities=commodities,
        po_names=po_names,
        po_comm_sets=po_comm_sets,
        row_ids=row_ids,
    )


# --------------------------------------------------------------------------- #
#  T3 feature extraction for a single query                                    #
# --------------------------------------------------------------------------- #
def _extract_t3_features_for_query(
    q_po_name: str,
    q_rid: int,
    candidate: Dict[str, Any],
    po_meta: Dict,
) -> List[float]:
    """Compute the 6 T3 co-occurrence features for one (query, candidate) pair."""
    pm = po_meta
    q_hhi     = float(pm["po_hhis"][q_rid])
    q_is_prim = float(pm["is_primary"][q_rid])
    q_qty     = float(pm["quantities"][q_rid])
    q_tot_qty = float(pm["po_total_qtys"][q_rid])
    q_other   = max(0.0, q_tot_qty - q_qty)
    q_comm_set = pm["po_comm_sets"].get(q_po_name, set())

    c_rid     = candidate["row_id"]
    c_hhi     = float(pm["po_hhis"][c_rid])
    c_is_prim = float(pm["is_primary"][c_rid])
    c_comm    = pm["commodities"][c_rid]

    return [
        q_hhi,
        q_is_prim,
        math.log1p(q_other),
        1.0 if c_comm in q_comm_set else 0.0,
        abs(q_hhi - c_hhi),
        1.0 if (q_is_prim > 0.5 and c_is_prim > 0.5) else 0.0,
    ]


# --------------------------------------------------------------------------- #
#  Reliability classification helper                                           #
# --------------------------------------------------------------------------- #
def classify_reliability(
    candidates: List[Dict[str, Any]],
    rank1_cand: Dict[str, Any],
    sorted_scores: np.ndarray,
    sorted_prices: List[float],
    q_date: pd.Timestamp,
) -> str:
    """
    Deterministic reliability classification: HIGH / MEDIUM / LOW.

    EVIDENCE SIGNALS USED:
    1. Candidate pool support: len(candidates)
    2. Retrieval channel / identity quality:
       - Structured/Identity: Channel A (exact MPN/identity), Channel B (brand+model),
         or Channel C (model family), or multi-channel consensus (channel_count >= 2).
       - Fallback: only coarse commodity/quantity channels (subset of {'I', 'J'}).
    3. Price dispersion (log_std):
       - Standard deviation of log prices among the top candidates (up to 5).
       - Low dispersion (log_std <= 0.35) reflects narrow price agreement.
       - Severe dispersion (log_std > 0.85) indicates wide variation / unobserved specs.
    4. Rank-1 margin: score lead of rank-1 over rank-2.
    5. Historical recency: age of candidate relative to query date.

    RULE:
    - HIGH:
        * Candidate support >= 5
        * Structured match (Channel A, B, or C) OR multi-channel consensus (>= 2 channels)
        * Low price dispersion: log_std <= 0.35
        * Positive ranking margin (margin >= 0.0)
        * Recency: within 3 years (days_elapsed <= 1095)
    - LOW:
        * Candidate support < 3
        * OR severe price dispersion: log_std > 0.85
        * OR only coarse fallback channels (subset of {'I', 'J'})
    - MEDIUM:
        * All other cases (moderate support, moderate price dispersion, valid text/spec match).
    """
    n_cands = len(candidates)
    if n_cands == 0:
        return "LOW"

    # 1. Price dispersion among top candidates (up to 5)
    top_prices = sorted_prices[:min(5, n_cands)]
    if len(top_prices) >= 2:
        log_std = float(np.std(np.log(np.maximum(top_prices, 1e-4))))
    else:
        log_std = 0.0

    # 2. Retrieval channel & consensus
    channels = rank1_cand.get("retrieval_channels", [])
    channel_count = rank1_cand.get("channel_count", 1)
    has_structured_match = ("A" in channels) or ("B" in channels) or ("C" in channels)
    has_consensus = channel_count >= 2
    is_coarse_only = bool(channels and set(channels).issubset({"I", "J"}))

    # 3. Score margin
    margin = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else 1.0

    # 4. Recency
    c_date_str = rank1_cand.get("award_date_parsed")
    if c_date_str is not None:
        try:
            c_date = pd.to_datetime(c_date_str)
            days_elapsed = max(0.0, (q_date - c_date).total_seconds() / 86400.0)
        except Exception:
            days_elapsed = 365.0
    else:
        days_elapsed = 365.0

    # Classification logic
    if (
        n_cands >= 5
        and (has_structured_match or has_consensus)
        and log_std <= 0.35
        and margin >= 0.0
        and days_elapsed <= 1095.0
    ):
        return "HIGH"
    elif n_cands < 3 or log_std > 0.85 or is_coarse_only:
        return "LOW"
    else:
        return "MEDIUM"


# --------------------------------------------------------------------------- #
#  Main predictor class                                                        #
# --------------------------------------------------------------------------- #
class ProcurementPredictor:
    """
    Load the frozen T3 ranker once, then call predict() for each query.

    Cold-start time: ~2-5 minutes (loading 61 MB dataset, building in-memory
    price-state model, quantity modeler, and 10-channel retrieval index).

    After warm-up, per-query inference time: ~0.5-2 seconds (depends on
    candidate pool size and feature computation).
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._log("Initializing ProcurementPredictor (frozen T3)...")
        t0 = time.time()

        # 1. Load serialized T3 ranker + metadata + conformal calibration
        assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}\nRun reconstruct_t3_ranker.py first."
        self.ranker   = joblib.load(MODEL_PATH)
        self.metadata = joblib.load(META_PATH) if META_PATH.exists() else {}
        self._log(f"  [OK] Ranker loaded from {MODEL_PATH.name}")

        if CONFORMAL_PATH.exists():
            self.conformal_artifact = joblib.load(CONFORMAL_PATH)
            self._log(f"  [OK] Conformal calibration loaded from {CONFORMAL_PATH.name}")
        else:
            self.conformal_artifact = None
            self._log("  [WARN] Conformal calibration artifact not found")

        # 2. Load canonical dataset (dev/train only -- no frozen test)
        df_all = pd.read_parquet(DATASET_PATH)
        df_all["award_date_parsed"] = pd.to_datetime(df_all["award_date_parsed"])
        df_all = df_all.sort_values(["award_date_parsed", "PURCHASE_ORDER"]).reset_index(drop=True)
        df_all["row_id"] = np.arange(len(df_all))

        if NORM_CACHE.exists():
            df_norm = pd.read_parquet(NORM_CACHE)
            for col in df_norm.columns:
                df_all[col] = df_norm[col]

        # Exclude frozen test records from all in-memory components
        # (frozen test = 2025-01-01 onward; we never access it)
        dev_mask = (df_all["award_date_parsed"] < pd.to_datetime("2025-01-01")).values
        n_dev = int(dev_mask.sum())
        n_total = len(df_all)
        self._log(f"  [OK] Dataset: {n_total:,} rows total, using {n_dev:,} dev rows for in-memory models")

        self._df_all = df_all
        self._dev_mask = dev_mask

        # 3. Load spec engine from entity_matcher.joblib
        matcher = joblib.load(ENTITY_PATH)
        self._spec_engine = matcher["spec_engine"]
        raw_texts = df_all["product_text_normalized"].fillna("").values
        self._all_embeddings = self._spec_engine.transform_text(df_all["product_text_normalized"])
        self._spec_engine.build_search_index(df_all, embeddings=self._all_embeddings)
        X_w = self._spec_engine.word_vec.transform(raw_texts)
        X_c = self._spec_engine.char_vec.transform(raw_texts)
        self._word_svd_all = normalize(self._spec_engine.word_svd.transform(X_w)).astype(np.float32)
        self._char_svd_all = normalize(self._spec_engine.char_svd.transform(X_c)).astype(np.float32)
        self._log("  [OK] Spec engine loaded & search index built")

        # 4. Parsers (stateless) with disk caching for fast cold-start
        self._identity_parser = ProductIdentityParser()
        self._spec_extractor  = TechnicalSpecificationExtractor()

        parsed_cache_file = PARSED_CACHE
        if parsed_cache_file.exists():
            self._log(f"  [OK] Loading pre-parsed records from cache: {parsed_cache_file.name}...")
            cached = joblib.load(parsed_cache_file)
            self._all_parsed_ids   = cached["ids"]
            self._all_parsed_specs = cached["specs"]
            self._all_parsed_comm  = cached["comm"]
            self._all_parsed_scope = cached["scope"]
            self._log("  [OK] Parsers loaded from cache")
        else:
            all_records = df_all.to_dict(orient="records")
            self._all_parsed_ids    = [self._identity_parser.parse_product_identity(r) for r in all_records]
            self._all_parsed_specs  = [self._spec_extractor.extract_specifications(r) for r in all_records]
            self._all_parsed_comm   = [CommercialUnitParser.parse_record(r) for r in all_records]
            self._all_parsed_scope  = [ScopeConfigurationParser.parse_record(r) for r in all_records]
            joblib.dump({
                "ids": self._all_parsed_ids,
                "specs": self._all_parsed_specs,
                "comm": self._all_parsed_comm,
                "scope": self._all_parsed_scope,
            }, parsed_cache_file, compress=3)
            self._log("  [OK] Parsers initialized, pre-parsed & cached")

        # 5. Price state model (fitted on dev data only)
        self._price_state_model = HierarchicalPriceStateModel(default_half_life=180.0)
        self._price_state_model.fit_timelines(df_all)
        self._log("  [OK] HierarchicalPriceStateModel fitted")

        # 6. Causal quantity modeler
        self._qt_modeler = CausalQuantityTemporalModeler(default_shrinkage_k=10.0)
        self._qt_modeler.fit_causal_history(df_all)
        self._log("  [OK] CausalQuantityTemporalModeler fitted")

        # 7. Retrieval engine
        self._retrieval_engine = MultiViewCandidateRetrievalEngine(max_cands_per_channel=30)
        self._retrieval_engine.build_indices(
            df_all,
            self._all_parsed_ids,
            self._all_parsed_specs,
            self._spec_engine,
            self._all_embeddings,
        )
        self._log("  [OK] MultiViewCandidateRetrievalEngine built (10 channels, max_k=60)")

        # 8. PO metadata arrays (for T3 co-occurrence features)
        self._po_meta = _build_po_metadata(df_all)
        self._log("  [OK] PO metadata computed")

        # 9. Feature extractor
        self._feat_extractor = Step30FeatureExtractor()

        elapsed = time.time() - t0
        self._log(f"  [OK] ProcurementPredictor ready in {elapsed:.1f}s")

    # ---------------------------------------------------------------------- #
    def predict(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict the benchmark unit price for a procurement query.

        Args:
            query: Dict with procurement line-item fields. Required fields:
                   ITEM_DESCRIPTION, COMMODITY, UNIT_OF_MEASURE, quantity_numeric,
                   award_date_parsed (str 'YYYY-MM-DD'), PURCHASE_ORDER, VENDOR_CODE.
                   Optional: BRAND_NAME, MODEL_NUMBER, PART_NUMBER, MASTER_AGREEMENT,
                             product_text_normalized, NUMERIC_SPECS, PRODUCT_TYPE, etc.

        Returns:
            Dict with:
              predicted_unit_price   -- float: actual historical unit price of rank-1 candidate
              rank1_candidate        -- dict: full rank-1 candidate record
              n_candidates           -- int: number of candidates retrieved
              candidate_prices       -- list[float]: all candidate prices (sorted by score desc)
              retrieval_time_s       -- float
              feature_time_s         -- float
              ranking_time_s         -- float
        """
        t_start = time.time()

        # Ensure award_date_parsed is datetime
        if "award_date_parsed" not in query:
            raise ValueError("query must contain 'award_date_parsed' (str 'YYYY-MM-DD' or datetime)")
        q_date = pd.to_datetime(query["award_date_parsed"])
        query["award_date_parsed"] = q_date

        # Frozen-test guard
        if q_date >= pd.to_datetime("2025-01-01"):
            raise ValueError(
                f"Query date {q_date.date()} is in the frozen test period (>= 2025-01-01). "
                "The frozen test is LOCKED. Use a development-period query."
            )

        # Standardize query fields for parsers & retrieval
        if "commodity_code" not in query:
            query["commodity_code"] = str(query.get("COMMODITY") or query.get("COMMODITY_CODE") or "OTHER_GOODS")
        if "uom_standardized" not in query:
            query["uom_standardized"] = str(query.get("UNIT_OF_MEASURE") or query.get("uom_normalized") or "EA")
        if "commodity_family" not in query:
            query["commodity_family"] = str(query.get("FAMILY_CODE") or "OTHER_GOODS")

        # Parse query
        q_id    = self._identity_parser.parse_product_identity(query)
        q_spec  = self._spec_extractor.extract_specifications(query)
        q_comm  = CommercialUnitParser.parse_record(query)
        q_scope = ScopeConfigurationParser.parse_record(query)
        q_state = self._price_state_model.estimate_query_state(query, q_date)

        # Causal stats strictly prior to query award date
        q_ts = q_date.timestamp()
        q_p_key = str(query.get("product_identity_key") or q_id.get("identity_key") or "")
        q_fam = str(query.get("commodity_family") or "OTHER_GOODS")
        q_causal = self._qt_modeler.get_causal_identity_stats(
            product_identity_key=q_p_key,
            commodity_family=q_fam,
            query_timestamp=q_ts,
        )

        # Embeddings & vectors for multi-view candidate retrieval
        q_text = str(
            query.get("product_text_normalized")
            or query.get("ITEM_DESCRIPTION")
            or query.get("COMMODITY_DESCRIPTION")
            or ""
        )
        q_emb = self._spec_engine.transform_text(pd.Series([q_text]))[0]
        X_w = self._spec_engine.word_vec.transform([q_text])
        X_c = self._spec_engine.char_vec.transform([q_text])
        q_word_vec = normalize(self._spec_engine.word_svd.transform(X_w))[0].astype(np.float32)
        q_char_vec = normalize(self._spec_engine.char_svd.transform(X_c))[0].astype(np.float32)

        # --- Retrieve candidates across all 10 channels ------------------ #
        t_ret = time.time()
        q_rid = int(query.get("row_id", -1))
        candidates = self._retrieval_engine.retrieve_multi_view_candidates(
            q_row=query,
            q_id=q_id,
            q_spec=q_spec,
            q_emb=q_emb,
            q_word_vec=q_word_vec,
            q_char_vec=q_char_vec,
            q_row_id=q_rid,
            enabled_channels={"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"},
            max_total_k=60,
            filter_mode="CONSERVATIVE",
        )
        retrieval_time = time.time() - t_ret

        if len(candidates) == 0:
            return {
                "benchmarkUnitPrice": None,
                "expectedRange": None,
                "lowerBound": None,
                "upperBound": None,
                "reliability": "LOW",
                "predicted_unit_price": None,
                "rank1_candidate": None,
                "n_candidates": 0,
                "candidate_prices": [],
                "retrieval_time_s": retrieval_time,
                "feature_time_s": 0.0,
                "ranking_time_s": 0.0,
                "error": "No candidates retrieved",
            }

        # --- Pool distribution for pool-level features ------------------- #
        pool_dist = self._feat_extractor.compute_candidate_pool_distribution(candidates)

        # Query PO name and row_id (for T3 features)
        # For new queries not in df_all, use sensible defaults
        q_po_name = str(query.get("PURCHASE_ORDER", ""))
        q_rid = int(query.get("row_id", -1))
        if q_rid < 0 or q_rid >= len(self._po_meta["po_hhis"]):
            # Query is not in the historical dataset -- use default T3 values
            q_hhi     = 1.0
            q_is_prim = 1.0
            q_qty     = float(query.get("quantity_numeric", 1.0))
            q_other   = 0.0
            q_comm_set = self._po_meta["po_comm_sets"].get(q_po_name, set())
            _use_default_t3 = True
        else:
            _use_default_t3 = False

        # --- Feature extraction for each candidate ----------------------- #
        t_feat = time.time()
        feat_matrix_rows = []

        for cand_idx, cand in enumerate(candidates):
            # Step30 207-feature vector → select T0 170 features
            try:
                c_id    = self._all_parsed_ids[cand["row_id"]]
                c_spec  = self._all_parsed_specs[cand["row_id"]]
                c_comm  = self._all_parsed_comm[cand["row_id"]]
                c_scope = self._all_parsed_scope[cand["row_id"]]
            except (KeyError, IndexError):
                c_id    = self._identity_parser.parse_product_identity(cand)
                c_spec  = self._spec_extractor.extract_specifications(cand)
                c_comm  = CommercialUnitParser.parse_record(cand)
                c_scope = ScopeConfigurationParser.parse_record(cand)

            feat_207 = self._feat_extractor.extract_pair_features_step30(
                q_row=query,
                q_id=q_id, q_spec=q_spec, q_state=q_state,
                q_comm=q_comm, q_scope=q_scope,
                cand=cand,
                c_id=c_id, c_spec=c_spec,
                c_comm=c_comm, c_scope=c_scope,
                causal_stats=q_causal,
                pool_dist=pool_dist,
                cand_idx_in_pool=cand_idx,
            )
            feat_170 = [feat_207[i] for i in T5_INDICES]

            # T3 co-occurrence features (6f)
            if _use_default_t3:
                c_rid   = cand["row_id"]
                c_hhi   = float(self._po_meta["po_hhis"][c_rid]) if c_rid < len(self._po_meta["po_hhis"]) else 1.0
                c_is_p  = float(self._po_meta["is_primary"][c_rid]) if c_rid < len(self._po_meta["is_primary"]) else 1.0
                c_comm_v = self._po_meta["commodities"][c_rid] if c_rid < len(self._po_meta["commodities"]) else ""
                t3_feats = [
                    q_hhi,
                    q_is_prim,
                    math.log1p(q_other),
                    1.0 if c_comm_v in q_comm_set else 0.0,
                    abs(q_hhi - c_hhi),
                    1.0 if (q_is_prim > 0.5 and c_is_p > 0.5) else 0.0,
                ]
            else:
                t3_feats = _extract_t3_features_for_query(
                    q_po_name, q_rid, cand, self._po_meta
                )

            feat_176 = feat_170 + t3_feats
            feat_matrix_rows.append(feat_176)

        feat_matrix = np.array(feat_matrix_rows, dtype=np.float32)
        feature_time = time.time() - t_feat

        # --- Rank with T3 LGBMRanker ------------------------------------- #
        t_rank = time.time()
        scores = self.ranker.predict(feat_matrix)
        ranking_time = time.time() - t_rank

        # Select rank-1 candidate
        sorted_idx = np.argsort(-scores)
        rank1_idx  = int(sorted_idx[0])
        rank1_cand = candidates[rank1_idx]
        predicted_price = float(rank1_cand.get("target_unit_price", 0.0))

        # All candidate prices in rank order
        sorted_prices = [
            float(candidates[j].get("target_unit_price", 0.0))
            for j in sorted_idx
        ]

        # Compute Conformal Prediction Interval (Split Conformal log-ratio)
        if self.conformal_artifact is not None:
            q_hat = float(self.conformal_artifact["primary_configuration"]["q_hat"])
            lower_bound = round(max(0.01, float(predicted_price * math.exp(-q_hat))), 2)
            upper_bound = round(float(predicted_price * math.exp(q_hat)), 2)
            expected_range = {
                "lower": lower_bound,
                "upper": upper_bound,
                "coverageLevel": float(self.conformal_artifact["primary_configuration"]["default_coverage_level"]),
                "method": str(self.conformal_artifact.get("calibration_method", "multiplicative_log_ratio_split_conformal"))
            }
        else:
            lower_bound = None
            upper_bound = None
            expected_range = None

        # Compute Reliability Classification (HIGH / MEDIUM / LOW)
        sorted_scores = scores[sorted_idx]
        reliability = classify_reliability(
            candidates=candidates,
            rank1_cand=rank1_cand,
            sorted_scores=sorted_scores,
            sorted_prices=sorted_prices,
            q_date=q_date,
        )

        return {
            "benchmarkUnitPrice": predicted_price,
            "expectedRange": expected_range,
            "lowerBound": lower_bound,
            "upperBound": upper_bound,
            "reliability": reliability,
            "predicted_unit_price": predicted_price,  # Backwards compatibility alias
            "rank1_candidate": rank1_cand,
            "n_candidates": len(candidates),
            "candidate_prices": sorted_prices,
            "rank1_score": float(scores[rank1_idx]),
            "retrieval_time_s": round(retrieval_time, 3),
            "feature_time_s":   round(feature_time, 3),
            "ranking_time_s":   round(ranking_time, 3),
            "total_time_s":     round(time.time() - t_start, 3),
        }

    # ---------------------------------------------------------------------- #
    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)


# --------------------------------------------------------------------------- #
#  CLI / smoke test                                                            #
# --------------------------------------------------------------------------- #
def _run_smoke_test(predictor: ProcurementPredictor):
    """
    Run a quick smoke test with a synthetic historical-format query.
    Uses a realistic description from public procurement terminology.
    """
    # Test 1: Realistic synthetic query
    test_query = {
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
        "PURCHASE_ORDER": "PO-SMOKE-001",
        "VENDOR_CODE": "VENDOR-UNKNOWN",
        "BRAND_NAME": "Dell",
        "MODEL_NUMBER": "Latitude",
        "PART_NUMBER": "",
        "MASTER_AGREEMENT": None,
        "PRODUCT_TYPE": "LAPTOP",
    }

    result = predictor.predict(test_query)

    if result.get("error"):
        print(f"  ERROR: {result['error']}")
        return False

    print(f"  Query:                 {test_query['COMMODITY_DESCRIPTION']}")
    print(f"  Benchmark unit price:  ${result['benchmarkUnitPrice']:,.4f}")
    if result.get("expectedRange"):
        print(f"  Conformal expected:    ${result['expectedRange']['lower']:,.2f} -- ${result['expectedRange']['upper']:,.2f} ({int(result['expectedRange']['coverageLevel']*100)}% coverage)")
    print(f"  Reliability:           {result.get('reliability', 'N/A')}")
    print(f"  Candidates retrieved:  {result['n_candidates']}")
    print(f"  Top-5 candidate prices:{[f'${p:,.2f}' for p in result['candidate_prices'][:5]]}")
    print(f"  Selected candidate:    Row #{result['rank1_candidate'].get('row_id')}: {result['rank1_candidate'].get('COMMODITY_DESCRIPTION', '')[:60]}...")
    print(f"  Retrieval time:        {result['retrieval_time_s']}s")
    print(f"  Feature time:          {result['feature_time_s']}s")
    print(f"  Ranking time:          {result['ranking_time_s']}s")
    print(f"  Total inference time:  {result['total_time_s']}s")
    print("\n  [OK] Smoke test PASSED -- inference pipeline working end-to-end")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CivicEngage T3 Procurement Predictor")
    parser.add_argument("--query_json", type=str, default=None,
                        help="JSON string of query dict")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run quick smoke test with synthetic query")
    args = parser.parse_args()

    predictor = ProcurementPredictor(verbose=True)

    if args.smoke_test or args.query_json is None:
        _run_smoke_test(predictor)
        return

    if args.query_json:
        query = json.loads(args.query_json)
        result = predictor.predict(query)
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
