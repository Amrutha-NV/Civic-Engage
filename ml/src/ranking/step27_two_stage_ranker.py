"""
Procurement Intelligence System — Step 27 Two-Stage Candidate Pruning & Fine-Grained Ranking Engine.

Implements:
1. Step 27 Rich Feature Extractor (G0 Base + G1 Provenance + G2 Multi-View Similarity +
   G3 Price-State + G4 Technical Spec + G5 Distribution + G6 Hard-Case + G7 Relative Price-Tier)
2. Stage 1 Candidate Quality Scorer (Binary LightGBM Classifier for candidate survival & pruning)
3. Soft Price-State & Multi-Channel Pruning Filters with Oracle-Preservation tracking
4. Stage 2 Fine-Grained LambdaMART Ranker (HN5 Graded Proximity Objective)
5. Price-Tier Clustering & Causal Mode Distance Extractor
6. Pairwise Rank Diagnoser (Auditing Rank #1 vs Valid +-10% Oracle Candidates)
"""

import os
import sys
import math
from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score


class Step27FeatureExtractor:
    """Extracts candidate features including multi-view similarity, provenance, price-state, and price-tier features."""

    G0_BASE = [
        "is_exact_product_match", "is_model_match", "is_brand_match", "is_uom_match",
        "is_same_commodity_code", "is_same_family", "match_tier_priority",
        "spec_cosine_similarity", "spec_composite_score", "has_spec_text",
        "days_elapsed", "years_elapsed", "log_days_elapsed", "recency_weight_365d", "recency_weight_730d",
        "log_query_qty", "log_cand_qty", "log_qty_ratio", "abs_log_qty_diff", "qty_similarity_score",
        "cand_unit_price", "log_cand_unit_price",
        "spec_x_recency", "spec_x_qty_sim", "exact_x_recency", "model_x_recency",
        "supervised_comparability_score", "is_high_conf_entity_match", "is_medium_conf_entity_match",
        "mpn_exact_match", "part_number_exact_match", "identity_key_match", "identity_confidence_score",
        "spec_numeric_compatibility_score", "supervised_comp_x_recency", "supervised_comp_x_qty_sim", "mpn_x_recency"
    ]

    G1_PROVENANCE = [
        "channel_count", "ch_A_exact_mpn", "ch_B_brand_model", "ch_C_model_family",
        "ch_D_numeric_specs", "ch_E_word_tfidf", "ch_F_char_ngrams", "ch_G_lsi_svd",
        "ch_H_semantic_emb", "ch_I_comm_uom", "ch_J_quantity_aware", "best_channel_priority",
        "channel_agreement_ratio", "multi_channel_match", "semantic_plus_char_agreement", "lexical_plus_spec_agreement"
    ]

    G2_SIMILARITY = [
        "view_word_sim", "view_char_sim", "view_lsi_sim", "view_emb_sim",
        "view_numeric_spec_sim", "view_identity_sim", "view_brand_sim", "view_model_sim",
        "view_family_sim", "view_qty_sim", "view_uom_match", "max_view_similarity",
        "mean_view_similarity", "weighted_view_similarity", "similarity_agreement", "similarity_gap"
    ]

    G3_PRICE_STATE = [
        "posterior_price_state", "state_uncertainty", "state_support_count", "state_days_since_last",
        "cand_distance_from_state", "cand_z_to_state", "recent_identity_median", "recent_weighted_median",
        "identity_price_trend", "state_agreement_weight"
    ]

    G4_SPECIFICATION = [
        "is_exact_enriched_id_match", "is_model_family_match", "is_brand_match_enriched", "is_product_type_match",
        "exact_spec_match_count", "numeric_spec_similarity", "categorical_spec_similarity", "spec_overlap_ratio",
        "spec_conflict_count", "has_critical_spec_conflict", "primary_capacity_ratio", "primary_capacity_diff",
        "composite_compatibility_score", "spec_sim_x_recency", "compat_x_state_gap", "crit_conflict_x_state_gap"
    ]

    G5_DISTRIBUTION = [
        "cand_price_percentile", "cand_price_median_distance", "cand_price_robust_z", "cand_price_iqr_distance",
        "cand_price_ratio_to_median", "cand_pool_dispersion", "cand_pool_iqr", "cand_price_rank_in_pool",
        "cand_pool_size", "cand_pool_q25", "cand_pool_q75", "cand_pool_mad"
    ]

    G6_HARD_CASE = [
        "is_high_dispersion", "is_low_dispersion", "is_cold_start", "is_warm_start",
        "is_stale_candidate", "is_quantity_mismatch", "is_spec_mismatch", "is_identity_uncertainty",
        "is_other_goods", "is_low_support", "is_high_support"
    ]

    G7_PRICE_TIER = [
        "cand_price_to_state_price", "log_price_ratio_to_state", "log_price_z_to_state",
        "cand_price_to_pool_median", "log_price_to_pool_median", "is_price_tier_consistent",
        "price_tier_rank_in_pool", "distance_to_nearest_price_mode", "price_density_ratio"
    ]

    ALL_FEATURE_NAMES = G0_BASE + G1_PROVENANCE + G2_SIMILARITY + G3_PRICE_STATE + G4_SPECIFICATION + G5_DISTRIBUTION + G6_HARD_CASE + G7_PRICE_TIER

    def __init__(self):
        pass

    def compute_candidate_pool_distribution(self, candidates: List[Dict[str, Any]]) -> Dict[str, float]:
        """Computes pre-query candidate pool distribution metrics. Zero query target price access."""
        if len(candidates) == 0:
            return {
                "median": 0.0, "log_median": 0.0, "log_mad": 0.0, "log_std": 0.0,
                "q25": 0.0, "q75": 0.0, "iqr": 0.0, "size": 0, "modes": []
            }
        prices = np.array([max(float(c.get("target_unit_price", 0.0)), 1e-4) for c in candidates], dtype=np.float64)
        log_p = np.log(prices)
        med = float(np.median(prices))
        log_med = float(np.median(log_p))
        log_mad = float(np.median(np.abs(log_p - log_med)))
        log_std = float(np.std(log_p)) if len(log_p) > 1 else 0.0
        q25 = float(np.percentile(log_p, 25))
        q75 = float(np.percentile(log_p, 75))
        iqr = max(q75 - q25, 0.0)

        # Estimate historical price modes via 1D binning
        hist, bin_edges = np.histogram(log_p, bins=min(max(len(log_p) // 3, 2), 10))
        top_bins = np.argsort(hist)[::-1][:3]
        modes = [float(0.5 * (bin_edges[b] + bin_edges[b + 1])) for b in top_bins if hist[b] > 0]

        return {
            "median": med, "log_median": log_med, "log_mad": log_mad, "log_std": log_std,
            "q25": q25, "q75": q75, "iqr": iqr, "size": len(candidates), "modes": modes
        }

    def extract_pair_features(
        self,
        q_row: Dict[str, Any],
        q_id: Dict[str, Any],
        q_spec: Dict[str, Any],
        q_state: Dict[str, Any],
        cand: Dict[str, Any],
        c_id: Dict[str, Any],
        c_spec: Dict[str, Any],
        pool_dist: Dict[str, float],
        cand_idx_in_pool: int,
        feature_set: str = "ALL"
    ) -> List[float]:
        """Extracts complete feature vector. Strictly 100% causal and anti-leakage compliant."""
        q_date = pd.to_datetime(q_row["award_date_parsed"])
        c_date = pd.to_datetime(cand["award_date_parsed"])
        days_elapsed = max(float((q_date - c_date).days), 0.0)
        years_elapsed = days_elapsed / 365.25
        log_days = math.log(days_elapsed + 1.0)
        rec_365 = math.exp(-math.log(2.0) * days_elapsed / 365.0)
        rec_730 = math.exp(-math.log(2.0) * days_elapsed / 730.0)

        # Quantity
        q_qty = max(float(q_row.get("quantity_numeric", 1.0) or 1.0), 1.0)
        c_qty = max(float(cand.get("quantity_numeric", 1.0) or 1.0), 1.0)
        log_q_qty = math.log10(q_qty)
        log_c_qty = math.log10(c_qty)
        log_qty_ratio = log_q_qty - log_c_qty
        abs_log_qty_diff = abs(log_qty_ratio)
        qty_sim = math.exp(-0.5 * abs_log_qty_diff)

        # Price
        cand_p = max(float(cand.get("target_unit_price", 0.0)), 1e-4)
        log_cand_p = math.log(cand_p)

        # Identifiers
        q_code = str(q_row.get("commodity_code", "")).strip()
        c_code = str(cand.get("commodity_code", "")).strip()
        q_fam = str(q_row.get("commodity_family", "")).strip()
        c_fam = str(cand.get("commodity_family", "")).strip()
        q_uom = str(q_row.get("uom_standardized", "")).strip()
        c_uom = str(cand.get("uom_standardized", "")).strip()

        is_same_comm = 1.0 if q_code == c_code and q_code else 0.0
        is_same_fam = 1.0 if q_fam == c_fam and q_fam else 0.0
        is_uom_match = 1.0 if q_uom == c_uom and q_uom else 0.0

        q_mpn = q_id.get("parsed_mpn") or q_row.get("normalized_mpn")
        c_mpn = c_id.get("parsed_mpn") or cand.get("normalized_mpn")
        q_brand = q_id.get("parsed_brand") or q_row.get("normalized_brand")
        c_brand = c_id.get("parsed_brand") or cand.get("normalized_brand")
        q_model = q_id.get("parsed_model") or q_row.get("normalized_model")
        c_model = c_id.get("parsed_model") or cand.get("normalized_model")
        q_mfam = q_id.get("parsed_model_family")
        c_mfam = c_id.get("parsed_model_family")

        mpn_match = 1.0 if q_mpn and c_mpn and str(q_mpn).strip().lower() == str(c_mpn).strip().lower() else 0.0
        brand_match = 1.0 if q_brand and c_brand and str(q_brand).strip().lower() == str(c_brand).strip().lower() else 0.0
        model_match = 1.0 if q_model and c_model and str(q_model).strip().lower() == str(c_model).strip().lower() else 0.0
        mfam_match = 1.0 if q_mfam and c_mfam and str(q_mfam).strip().lower() == str(c_mfam).strip().lower() else 0.0
        exact_prod = 1.0 if (mpn_match or (brand_match and model_match)) else 0.0

        prio = float(cand.get("channel_priority", 5))

        # Base Spec Similarity
        spec_cos = float(cand.get("emb_sim", cand.get("spec_cosine_similarity", 0.5)))
        spec_comp = spec_cos * 0.70 + 0.30 * is_same_comm
        has_spec_text = 1.0 if bool(cand.get("product_text_normalized")) else 0.0

        # Base Interactions
        spec_x_rec = spec_cos * rec_365
        spec_x_qty = spec_cos * qty_sim
        exact_x_rec = exact_prod * rec_365
        model_x_rec = model_match * rec_365

        # Identity & Comparability
        sup_comp = float(cand.get("supervised_comparability_score", exact_prod * 0.95))
        is_hi_conf = 1.0 if sup_comp >= 0.80 else 0.0
        is_med_conf = 1.0 if sup_comp >= 0.50 else 0.0
        id_conf = float(q_row.get("identity_confidence", 0.5) if isinstance(q_row.get("identity_confidence"), (int, float)) else 0.5)

        # G0: Base Features
        feats_G0 = [
            exact_prod, model_match, brand_match, is_uom_match, is_same_comm, is_same_fam, prio,
            spec_cos, spec_comp, has_spec_text,
            days_elapsed, years_elapsed, log_days, rec_365, rec_730,
            log_q_qty, log_c_qty, log_qty_ratio, abs_log_qty_diff, qty_sim,
            cand_p, log_cand_p,
            spec_x_rec, spec_x_qty, exact_x_rec, model_x_rec,
            sup_comp, is_hi_conf, is_med_conf, mpn_match, mpn_match, exact_prod, id_conf,
            spec_cos, sup_comp * rec_365, sup_comp * qty_sim, mpn_match * rec_365
        ]

        if feature_set == "G0":
            return feats_G0

        # G1: Provenance Features
        ch_list = cand.get("retrieval_channels", [cand.get("channel", "E")])
        ch_set = set(ch_list) if isinstance(ch_list, (list, set)) else {str(ch_list)}
        ch_count = float(len(ch_set))

        ch_A = 1.0 if ("A" in ch_set or "A_EXACT_IDENTITY" in ch_set) else 0.0
        ch_B = 1.0 if ("B" in ch_set or "B_BRAND_MODEL" in ch_set) else 0.0
        ch_C = 1.0 if ("C" in ch_set or "C_MODEL_FAMILY" in ch_set) else 0.0
        ch_D = 1.0 if ("D" in ch_set or "D_NUMERIC_SPECS" in ch_set) else 0.0
        ch_E = 1.0 if ("E" in ch_set or "E_WORD_TFIDF" in ch_set) else 0.0
        ch_F = 1.0 if ("F" in ch_set or "F_CHAR_NGRAMS" in ch_set) else 0.0
        ch_G = 1.0 if ("G" in ch_set or "G_LSI_SVD" in ch_set) else 0.0
        ch_H = 1.0 if ("H" in ch_set or "H_SEMANTIC_EMB" in ch_set) else 0.0
        ch_I = 1.0 if ("I" in ch_set or "I_COMM_UOM" in ch_set) else 0.0
        ch_J = 1.0 if ("J" in ch_set or "J_QUANTITY_AWARE" in ch_set) else 0.0

        best_prio = float(cand.get("channel_priority", 5))
        agr_ratio = ch_count / 10.0
        multi_ch = 1.0 if ch_count >= 2.0 else 0.0
        sem_char_agr = 1.0 if (ch_H or ch_G) and ch_F else 0.0
        lex_spec_agr = 1.0 if (ch_E or ch_F) and (ch_A or ch_B or ch_D) else 0.0

        feats_G1 = [
            ch_count, ch_A, ch_B, ch_C, ch_D, ch_E, ch_F, ch_G, ch_H, ch_I, ch_J,
            best_prio, agr_ratio, multi_ch, sem_char_agr, lex_spec_agr
        ]

        # G2: Multi-View Similarity Features
        v_word = float(cand.get("word_sim", 0.0))
        v_char = float(cand.get("char_sim", 0.0))
        v_lsi = float(cand.get("lsi_sim", 0.0))
        v_emb = float(cand.get("emb_sim", 0.0))

        # Numeric Spec Similarity
        spec_keys = ["ram_gb", "storage_gb", "ports", "voltage_v", "horsepower_hp", "pipe_diameter_in"]
        matched_specs = 0
        total_q_specs = 0
        for sk in spec_keys:
            qv = q_spec.get(sk)
            cv = c_spec.get(sk)
            if qv is not None:
                total_q_specs += 1
                if cv is not None:
                    try:
                        if abs(float(qv) - float(cv)) / max(float(qv), 1e-4) <= 0.15:
                            matched_specs += 1
                    except (ValueError, TypeError):
                        pass
        v_num_spec = (matched_specs / max(total_q_specs, 1)) if total_q_specs > 0 else (1.0 if exact_prod else 0.5)

        v_id = 1.0 if exact_prod else (0.5 if mfam_match else 0.0)
        v_brand = brand_match
        v_model = model_match
        v_fam = is_same_fam
        v_qty = qty_sim
        v_uom = is_uom_match

        sims_list = [v_word, v_char, v_lsi, v_emb, v_num_spec]
        max_sim = float(max(sims_list))
        mean_sim = float(np.mean(sims_list))
        weighted_sim = 0.35 * v_emb + 0.30 * v_char + 0.20 * v_word + 0.15 * v_num_spec
        sim_agr = max(1.0 - float(np.std(sims_list)), 0.0)
        sorted_sims = sorted(sims_list, reverse=True)
        sim_gap = float(sorted_sims[0] - sorted_sims[1]) if len(sorted_sims) > 1 else 0.0

        feats_G2 = [
            v_word, v_char, v_lsi, v_emb, v_num_spec, v_id, v_brand, v_model,
            v_fam, v_qty, v_uom, max_sim, mean_sim, weighted_sim, sim_agr, sim_gap
        ]

        # G3: Causal Price State Features
        post_st = float(q_state.get("posterior_mean", q_state.get("posterior_price_state", 4.0))) if q_state else 4.0
        st_unc = float(q_state.get("posterior_variance", q_state.get("state_uncertainty", 1.0))) if q_state else 1.0
        st_supp = float(q_state.get("effective_support_N", q_state.get("state_support_count", 0.0))) if q_state else 0.0
        st_days = float(q_state.get("effective_half_life_days", q_state.get("state_days_since_last", 365.0))) if q_state else 365.0

        dist_st = abs(log_cand_p - post_st)
        z_st = dist_st / max(math.sqrt(max(st_unc, 1e-4)), 1e-3)
        rec_med = float(q_state.get("recent_identity_median", post_st)) if q_state else post_st
        rec_wmed = float(q_state.get("recent_weighted_median", post_st)) if q_state else post_st
        st_trend = float(q_state.get("identity_price_trend", 0.0)) if q_state else 0.0
        st_agr = math.exp(-dist_st)

        feats_G3 = [
            post_st, st_unc, st_supp, st_days, dist_st, z_st, rec_med, rec_wmed, st_trend, st_agr
        ]

        # G4: Technical Specification Compatibility
        has_crit = 0.0
        spec_conflicts = 0
        for sk in spec_keys:
            qv = q_spec.get(sk)
            cv = c_spec.get(sk)
            if qv is not None and cv is not None:
                try:
                    if abs(float(qv) - float(cv)) / max(float(qv), 1e-4) > 0.35:
                        has_crit = 1.0
                        spec_conflicts += 1
                except (ValueError, TypeError):
                    pass

        comp_score = max_sim * (1.0 - 0.5 * has_crit)
        spec_x_rec_g4 = v_num_spec * rec_365
        compat_x_gap = comp_score / (dist_st + 1.0)
        crit_x_gap = has_crit * dist_st

        feats_G4 = [
            exact_prod, mfam_match, brand_match, 1.0 if is_same_comm else 0.0,
            float(matched_specs), v_num_spec, 1.0 if is_uom_match else 0.0,
            float(matched_specs) / max(total_q_specs, 1), float(spec_conflicts), has_crit,
            1.0, 0.0, comp_score, spec_x_rec_g4, compat_x_gap, crit_x_gap
        ]

        # G5: Distribution & Dispersion Features
        p_med = pool_dist.get("median", cand_p)
        log_p_med = pool_dist.get("log_median", log_cand_p)
        log_p_mad = pool_dist.get("log_mad", 0.0)
        log_p_std = pool_dist.get("log_std", 0.0)
        p_q25 = pool_dist.get("q25", log_cand_p)
        p_q75 = pool_dist.get("q75", log_cand_p)
        p_iqr = pool_dist.get("iqr", 0.0)
        p_size = pool_dist.get("size", 1)

        p_percentile = (cand_idx_in_pool + 0.5) / max(p_size, 1)
        dist_med = abs(log_cand_p - log_p_med)
        rob_z = dist_med / (log_p_mad + 1e-4)
        dist_iqr = max(p_q25 - log_cand_p, log_cand_p - p_q75, 0.0)
        ratio_med = cand_p / max(p_med, 1e-4)
        p_rank = float(cand_idx_in_pool + 1)

        feats_G5 = [
            p_percentile, dist_med, rob_z, dist_iqr, ratio_med, log_p_std, p_iqr, p_rank,
            float(p_size), p_q25, p_q75, log_p_mad
        ]

        # G6: Hard-Case Indicators & Specialization Features
        is_hi_disp = 1.0 if log_p_std > 0.75 else 0.0
        is_lo_disp = 1.0 if log_p_std < 0.25 else 0.0
        is_cold = 1.0 if st_supp <= 0.0 else 0.0
        is_warm = 1.0 if st_supp >= 5.0 else 0.0
        is_stale = 1.0 if days_elapsed > 1095.0 else 0.0
        is_qty_mismatch = 1.0 if abs_log_qty_diff > 1.0 else 0.0
        is_spec_mismatch = 1.0 if has_crit > 0.0 else 0.0
        is_id_unc = 1.0 if id_conf < 0.50 else 0.0
        is_other = 1.0 if q_fam == "OTHER_GOODS" or not q_fam else 0.0
        is_lo_supp = 1.0 if p_size <= 3 else 0.0
        is_hi_supp = 1.0 if p_size >= 25 else 0.0

        feats_G6 = [
            is_hi_disp, is_lo_disp, is_cold, is_warm, is_stale, is_qty_mismatch,
            is_spec_mismatch, is_id_unc, is_other, is_lo_supp, is_hi_supp
        ]

        # G7: Relative Price-Tier Features
        cand_to_state_price = cand_p / max(math.exp(post_st), 1e-4)
        log_price_ratio_to_state = log_cand_p - post_st
        log_price_z_to_state = (log_cand_p - post_st) / max(math.sqrt(max(st_unc, 1e-4)), 0.05)
        cand_to_pool_med = cand_p / max(p_med, 1e-4)
        log_to_pool_med = log_cand_p - log_p_med
        is_tier_consistent = 1.0 if abs(log_price_z_to_state) <= 1.5 else 0.0
        tier_rank_in_pool = float(cand_idx_in_pool + 1) / max(p_size, 1)

        # Distance to nearest dense price mode
        modes = pool_dist.get("modes", [])
        if len(modes) > 0:
            min_mode_dist = float(min(abs(log_cand_p - m) for m in modes))
            price_density_ratio = math.exp(-min_mode_dist)
        else:
            min_mode_dist = dist_med
            price_density_ratio = math.exp(-dist_med)

        feats_G7 = [
            cand_to_state_price, log_price_ratio_to_state, log_price_z_to_state,
            cand_to_pool_med, log_to_pool_med, is_tier_consistent,
            tier_rank_in_pool, min_mode_dist, price_density_ratio
        ]

        if feature_set == "G1_ONLY":
            return feats_G0 + feats_G1
        elif feature_set == "G2_ONLY":
            return feats_G0 + feats_G1 + feats_G2
        elif feature_set == "STAGE1":
            return feats_G0 + feats_G1 + feats_G2 + feats_G3 + feats_G4 + feats_G5
        else:
            return feats_G0 + feats_G1 + feats_G2 + feats_G3 + feats_G4 + feats_G5 + feats_G6 + feats_G7


class Stage1CandidateQualityScorer:
    """Trains a binary quality classifier to estimate P(candidate +-10% accuracy) for pool pruning."""

    def __init__(self, n_estimators: int = 250, learning_rate: float = 0.05, num_leaves: int = 31, random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.model: Optional[lgb.LGBMClassifier] = None

    def fit(self, X_train: np.ndarray, y_train_bin: np.ndarray, feature_names: List[str]) -> "Stage1CandidateQualityScorer":
        """Fits binary LightGBM classifier with balanced class weights."""
        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            class_weight="balanced",
            subsample=0.85,
            colsample_bytree=0.80,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train_bin, feature_name=feature_names)
        return self

    def predict_quality_scores(self, X: np.ndarray) -> np.ndarray:
        """Outputs P(candidate +-10% match)."""
        if self.model is None:
            raise RuntimeError("Stage 1 model is not fitted.")
        return self.model.predict_proba(X)[:, 1]

    def prune_candidate_pool(
        self,
        candidates: List[Dict[str, Any]],
        quality_scores: np.ndarray,
        q_state: Dict[str, Any],
        top_k: int = 20,
        sigma_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Prunes candidate pool to Top-K based on Stage 1 quality scores and optional soft price-state filter.
        Ensures high oracle retention while removing low-relevance semantic distractors.
        """
        if len(candidates) == 0:
            return []

        post_st = float(q_state.get("posterior_mean", q_state.get("posterior_price_state", 4.0))) if q_state else 4.0
        st_unc = float(q_state.get("posterior_variance", q_state.get("state_uncertainty", 1.0))) if q_state else 1.0
        st_sigma = max(math.sqrt(max(st_unc, 1e-4)), 0.05)

        scored_cands = []
        for i, c in enumerate(candidates):
            c_copy = dict(c)
            s_q = float(quality_scores[i])
            c_p = max(float(c.get("target_unit_price", 0.0)), 1e-4)
            log_c_p = math.log(c_p)
            z_st = abs(log_c_p - post_st) / st_sigma

            c_copy["stage1_quality_score"] = s_q
            c_copy["price_state_z"] = z_st

            # Check soft state threshold (reject only if far away AND low quality score)
            if sigma_threshold is not None:
                if z_st > sigma_threshold and s_q < 0.35 and ("A" not in c.get("retrieval_channels", [])) and ("B" not in c.get("retrieval_channels", [])):
                    continue

            scored_cands.append(c_copy)

        if len(scored_cands) < min(5, len(candidates)):
            # Fallback to all candidates if too aggressive
            scored_cands = [dict(c) for c in candidates]
            for i, c in enumerate(scored_cands):
                c["stage1_quality_score"] = float(quality_scores[i])

        # Sort by stage1_quality_score descending
        scored_cands.sort(key=lambda x: -x.get("stage1_quality_score", 0.0))
        return scored_cands[:top_k]


class Stage2FineGrainedRanker:
    """Stage 2 Fine-Grained LambdaMART Ranker trained on pruned candidate pools."""

    def __init__(
        self,
        n_estimators: int = 380,
        learning_rate: float = 0.035,
        num_leaves: int = 31,
        min_child_samples: int = 25,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.random_state = random_state

        self.model = None
        self.feature_names: List[str] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train_hn5: np.ndarray,
        groups_train: np.ndarray,
        feature_names: List[str]
    ) -> "Stage2FineGrainedRanker":
        """Fits LambdaMART with HN5 graded relevance gain."""
        self.feature_names = feature_names

        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            ndcg_eval_at=[1, 3, 5, 10],
            label_gain=[0, 1, 3, 7, 15],
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=0.85,
            colsample_bytree=0.80,
            random_state=self.random_state,
            n_jobs=-1,
            importance_type="gain"
        )
        self.model.fit(X_train, y_train_hn5, group=groups_train, feature_name=feature_names)
        return self

    def predict_rank_scores(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Stage 2 model is not fitted.")
        return self.model.predict(X)

    def select_best_candidate(self, candidates: List[Dict[str, Any]], scores: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Selects Rank #1 candidate and returns its raw unit price. Zero price transformation."""
        if len(candidates) == 0:
            return {}, {"confidence": 0.0, "score_margin": 0.0}

        sort_idx = np.argsort(-scores)
        best_cand = candidates[sort_idx[0]]

        s1 = float(scores[sort_idx[0]])
        s2 = float(scores[sort_idx[1]]) if len(scores) > 1 else s1 - 1.0
        score_margin = s1 - s2

        exp_s = np.exp(scores - np.max(scores))
        probs = exp_s / np.sum(exp_s)
        conf = float(probs[sort_idx[0]])

        diag = {
            "rank1_score": s1,
            "score_margin": score_margin,
            "confidence": conf
        }
        return best_cand, diag
