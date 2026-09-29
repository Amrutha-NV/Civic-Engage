"""
Procurement Intelligence System — Advanced Product & Specification Similarity Engine (V2).

Implements multi-representation semantic & specification similarity search:
1. Word-level TF-IDF (1-2 ngrams) + TruncatedSVD (64 components)
2. Character n-gram TF-IDF (char_wb 3-5 ngrams) + TruncatedSVD (64 components)
3. Normalized Composite Embeddings (0.5 * Word + 0.5 * Char)
4. Structured Compatibility Scoring (Commodity Code, Family, UOM, Brand, Model)
5. Strict Anti-Leakage Invariant (historical.AWARD_DATE < query.AWARD_DATE)
6. Similarity-Weighted Recency Price Estimation
"""

import os
import sys
import time
import bisect
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


class SpecificationSimilarityEngine:
    """Advanced multi-representation semantic and specification retrieval engine."""

    def __init__(self, n_components: int = 64):
        self.n_components = n_components
        self.word_vec: Optional[TfidfVectorizer] = None
        self.char_vec: Optional[TfidfVectorizer] = None
        self.word_svd: Optional[TruncatedSVD] = None
        self.char_svd: Optional[TruncatedSVD] = None

        # Grouped search index: comm_code -> list of records sorted by award_date
        self.comm_index: Dict[str, Dict[str, Any]] = {}
        self.family_index: Dict[str, Dict[str, Any]] = {}
        self.is_indexed: bool = False

    def fit_representations(self, df_train: pd.DataFrame) -> None:
        """Fits word and character TF-IDF and SVD representations strictly on TRAIN data."""
        print("    * Fitting Word TF-IDF Vectorizer on TRAIN...", flush=True)
        self.word_vec = TfidfVectorizer(
            ngram_range=(1, 2), min_df=5, max_features=25000, sublinear_tf=True
        )
        X_w_tr = self.word_vec.fit_transform(df_train["product_text_normalized"].fillna("").values)

        print("    * Fitting Char n-gram TF-IDF Vectorizer on TRAIN...", flush=True)
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=10, max_features=35000, sublinear_tf=True
        )
        X_c_tr = self.char_vec.fit_transform(df_train["product_text_normalized"].fillna("").values)

        print(f"    * Fitting TruncatedSVD ({self.n_components} components) on Word TF-IDF...", flush=True)
        self.word_svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.word_svd.fit(X_w_tr)

        print(f"    * Fitting TruncatedSVD ({self.n_components} components) on Char TF-IDF...", flush=True)
        self.char_svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.char_svd.fit(X_c_tr)

    def transform_text(self, text_series: pd.Series) -> np.ndarray:
        """Transforms text into normalized composite 64-d embeddings."""
        raw_text = text_series.fillna("").values
        X_w = self.word_vec.transform(raw_text)
        X_c = self.char_vec.transform(raw_text)

        E_w = normalize(self.word_svd.transform(X_w))
        E_c = normalize(self.char_svd.transform(X_c))

        E_comb = normalize(0.5 * E_w + 0.5 * E_c)
        return E_comb.astype(np.float32)

    def build_search_index(self, df_full: pd.DataFrame, embeddings: Optional[np.ndarray] = None) -> None:
        """Builds partitioned chronological indices with composite embeddings."""
        if embeddings is None:
            print("    * Generating dense embeddings for full dataset...", flush=True)
            embeddings = self.transform_text(df_full["product_text_normalized"])

        n_rows = len(df_full)
        d_vals_raw = pd.to_datetime(df_full["award_date_parsed"]).values
        sort_idx = np.argsort(d_vals_raw)

        c_codes = df_full["commodity_code"].values[sort_idx]
        fams = df_full["commodity_family"].values[sort_idx]
        uoms = df_full["uom_standardized"].values[sort_idx]
        b_vals = df_full["brand"].values[sort_idx] if "brand" in df_full else [None] * n_rows
        m_vals = df_full["model"].values[sort_idx] if "model" in df_full else [None] * n_rows
        d_vals = d_vals_raw[sort_idx]
        p_vals = df_full["target_unit_price"].values[sort_idx].astype(np.float64)
        q_vals = df_full["quantity_numeric"].fillna(1.0).values[sort_idx].astype(np.float64)
        emb_sorted = embeddings[sort_idx]

        r_ids = df_full["row_id"].values[sort_idx] if "row_id" in df_full else np.arange(n_rows)[sort_idx]

        self.comm_index.clear()
        self.family_index.clear()

        # Populate partitioned commodity and family buckets
        for i in range(n_rows):
            c_code = str(c_codes[i]).strip()
            fam = str(fams[i]).strip()
            uom = str(uoms[i]).strip()
            b_val = b_vals[i]
            m_val = m_vals[i]
            d_val = d_vals[i]
            p_val = p_vals[i]
            q_val = q_vals[i]
            emb = emb_sorted[i]
            r_id = r_ids[i]

            # Commodity index
            if c_code not in self.comm_index:
                self.comm_index[c_code] = {
                    "dates": [], "prices": [], "quantities": [],
                    "uoms": [], "brands": [], "models": [], "embeddings": [], "row_ids": []
                }
            self.comm_index[c_code]["dates"].append(d_val)
            self.comm_index[c_code]["prices"].append(p_val)
            self.comm_index[c_code]["quantities"].append(q_val)
            self.comm_index[c_code]["uoms"].append(uom)
            self.comm_index[c_code]["brands"].append(str(b_val).strip() if pd.notnull(b_val) else "")
            self.comm_index[c_code]["models"].append(str(m_val).strip() if pd.notnull(m_val) else "")
            self.comm_index[c_code]["embeddings"].append(emb)
            self.comm_index[c_code]["row_ids"].append(r_id)

            # Family index
            if fam not in self.family_index:
                self.family_index[fam] = {
                    "dates": [], "prices": [], "quantities": [],
                    "uoms": [], "brands": [], "models": [], "embeddings": [], "row_ids": []
                }
            self.family_index[fam]["dates"].append(d_val)
            self.family_index[fam]["prices"].append(p_val)
            self.family_index[fam]["quantities"].append(q_val)
            self.family_index[fam]["uoms"].append(uom)
            self.family_index[fam]["brands"].append(str(b_val).strip() if pd.notnull(b_val) else "")
            self.family_index[fam]["models"].append(str(m_val).strip() if pd.notnull(m_val) else "")
            self.family_index[fam]["embeddings"].append(emb)
            self.family_index[fam]["row_ids"].append(r_id)

        # Convert embedding lists to numpy matrices for fast dot-product
        for c_code in self.comm_index:
            self.comm_index[c_code]["emb_mat"] = np.vstack(self.comm_index[c_code]["embeddings"])
        for fam in self.family_index:
            self.family_index[fam]["emb_mat"] = np.vstack(self.family_index[fam]["embeddings"])

        self.is_indexed = True
        print(f"    * Indexed {len(self.comm_index):,} commodity codes and {len(self.family_index):,} families.", flush=True)

    def query_similarity(
        self,
        commodity_code: str,
        commodity_family: str,
        brand: Optional[str],
        model: Optional[str],
        uom: str,
        query_date: Any,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_sim_threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Retrieves Top-K similar historical products strictly before query_date.
        """
        c_code = str(commodity_code).strip()
        fam = str(commodity_family).strip()
        uom_str = str(uom).strip()
        b_str = str(brand).strip() if pd.notnull(brand) and str(brand).strip() not in ("nan", "none", "null", "") else ""
        m_str = str(model).strip() if pd.notnull(model) and str(model).strip() not in ("nan", "none", "null", "") else ""
        q_dt = pd.to_datetime(query_date)

        # Target candidate group: prefer exact commodity code, fallback to commodity family
        candidate_group = None
        match_scope = "COMMODITY_CODE"

        if c_code in self.comm_index:
            d_list = self.comm_index[c_code]["dates"]
            pos = bisect.bisect_left(d_list, q_dt)
            if pos >= 1:
                candidate_group = self.comm_index[c_code]
                cand_pos = pos

        if candidate_group is None and fam in self.family_index:
            d_list = self.family_index[fam]["dates"]
            pos = bisect.bisect_left(d_list, q_dt)
            if pos >= 1:
                candidate_group = self.family_index[fam]
                cand_pos = pos
                match_scope = "FAMILY_FALLBACK"

        if candidate_group is None or cand_pos == 0:
            return {
                "match_type": "NO_MATCH",
                "similarity_score": 0.0,
                "predicted_unit_price": np.nan,
                "history_count": 0,
                "max_similarity": 0.0,
                "mean_similarity": 0.0,
                "price_dispersion": 0.0,
                "days_since_latest": -1,
                "match_scope": "NONE"
            }

        # Subsample historical candidates strictly before query date
        cand_dates = candidate_group["dates"][:cand_pos]
        cand_prices = np.array(candidate_group["prices"][:cand_pos], dtype=np.float64)
        cand_uoms = candidate_group["uoms"][:cand_pos]
        cand_brands = candidate_group["brands"][:cand_pos]
        cand_models = candidate_group["models"][:cand_pos]
        cand_emb_mat = candidate_group["emb_mat"][:cand_pos]

        # 1. Cosine similarity between query embedding and candidate embeddings
        cos_sims = np.dot(cand_emb_mat, query_embedding)

        # 2. Structured bonuses & Recency penalty
        days_diff = np.array([(q_dt - d).days for d in cand_dates], dtype=np.float64)
        uom_match = np.array([1.0 if u == uom_str else 0.0 for u in cand_uoms])
        brand_match = np.array([1.0 if (b_str and b == b_str) else 0.0 for b in cand_brands])
        model_match = np.array([1.0 if (m_str and m == m_str) else 0.0 for m in cand_models])

        recency_factor = np.exp(-days_diff / 730.0) # 2-year half-life

        composite_scores = (
            cos_sims * 0.60 +
            uom_match * 0.15 +
            brand_match * 0.10 +
            model_match * 0.15
        )

        # Filter by minimum threshold
        valid_indices = np.where(composite_scores >= min_sim_threshold)[0]
        if len(valid_indices) == 0:
            # Fallback: top 1 if cos_sim >= 0.35
            top1_idx = int(np.argmax(composite_scores))
            if composite_scores[top1_idx] >= 0.35:
                valid_indices = np.array([top1_idx])
            else:
                return {
                    "match_type": "NO_MATCH",
                    "similarity_score": float(composite_scores[top1_idx]),
                    "predicted_unit_price": np.nan,
                    "history_count": cand_pos,
                    "max_similarity": float(composite_scores[top1_idx]),
                    "mean_similarity": float(composite_scores[top1_idx]),
                    "price_dispersion": 0.0,
                    "days_since_latest": int(days_diff[-1]),
                    "match_scope": match_scope
                }

        # Select Top-K highest composite scores
        selected_scores = composite_scores[valid_indices]
        top_k_order = np.argsort(selected_scores)[::-1][:top_k]
        final_indices = valid_indices[top_k_order]

        top_prices = cand_prices[final_indices]
        top_scores = composite_scores[final_indices]
        top_recency = recency_factor[final_indices]

        # Weights: composite_score * recency
        weights = top_scores * top_recency
        weight_sum = np.sum(weights)

        if weight_sum > 0:
            sort_p_idx = np.argsort(top_prices)
            s_p = top_prices[sort_p_idx]
            s_w = weights[sort_p_idx]
            cum_w = np.cumsum(s_w) / weight_sum
            med_idx = np.searchsorted(cum_w, 0.5)
            pred_price = float(s_p[min(med_idx, len(s_p)-1)])
        else:
            pred_price = float(np.median(top_prices))

        log_arr = np.log(np.maximum(cand_prices, 1e-6))
        p25, p75 = np.percentile(log_arr, 25), np.percentile(log_arr, 75)
        iqr_disp = float(max(0.0, p75 - p25))

        return {
            "match_type": "SPEC_SIMILARITY_MATCH",
            "similarity_score": float(np.max(top_scores)),
            "predicted_unit_price": pred_price,
            "history_count": len(final_indices),
            "max_similarity": float(np.max(top_scores)),
            "mean_similarity": float(np.mean(top_scores)),
            "price_dispersion": iqr_disp,
            "days_since_latest": int(days_diff[final_indices[0]]),
            "match_scope": match_scope
        }
