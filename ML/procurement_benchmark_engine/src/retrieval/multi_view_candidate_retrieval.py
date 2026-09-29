"""
Procurement Intelligence System — Multi-View Candidate Retrieval Engine (Step 25).

Implements 10 Parallel Candidate Retrieval Channels (A–J) and their Multi-View Union:
- Channel A: Exact Identity (MPN, SKU, Part Number)
- Channel B: Brand + Model Matching
- Channel C: Model Family Series Matching
- Channel D: Numeric Technical Specification Matching (RAM, storage, ports, voltage, power, pipe diameter, etc.)
- Channel E: Word-Level TF-IDF Cosine Retrieval
- Channel F: Character N-Gram Subword & Model String Retrieval
- Channel G: LSI / SVD Latent Semantic Retrieval
- Channel H: Dense Composite Semantic Embedding Retrieval
- Channel I: Commodity + UOM Broad Fallback
- Channel J: Quantity-Aware Preference Retrieval

Strict Anti-Leakage & Causality Invariants:
- historical.award_date < query.award_date for 100% of candidate pairs.
- Zero query target price, contract price, or ITM_TOT_AM leakage.
- Frozen-test (2025–2026) records strictly forbidden and blocked.
"""

import os
import sys
import math
import bisect
from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


NUMERIC_SPEC_KEYS = [
    "ram_gb", "storage_gb", "ports", "voltage_v", "power_w",
    "horsepower_hp", "pipe_diameter_in", "pressure_psi", "flow_gpm",
    "lumens", "speed_gbps"
]


class MultiViewCandidateRetrievalEngine:
    """Multi-View Parallel Candidate Retrieval and Union Engine."""

    def __init__(
        self,
        max_cands_per_channel: int = 30,
        theta_high: float = 0.80
    ):
        self.max_cands_per_channel = max_cands_per_channel
        self.theta_high = theta_high

        # Channel Index Stores: key -> {'dates': [], 'prices': [], 'quantities': [], 'row_ids': []}
        self.index_mpn_comm: Dict[Tuple[str, str, str], Dict[str, list]] = {}
        self.index_mpn_global: Dict[str, Dict[str, list]] = {}
        self.index_brand_model_comm: Dict[Tuple[str, str, str, str], Dict[str, list]] = {}
        self.index_brand_model_fam: Dict[Tuple[str, str, str], Dict[str, list]] = {}
        self.index_model_family_comm: Dict[Tuple[str, str, str], Dict[str, list]] = {}
        self.index_model_family_fam: Dict[Tuple[str, str], Dict[str, list]] = {}
        self.index_numeric_specs: Dict[Tuple[str, str, float], Dict[str, list]] = {}
        self.index_comm_uom: Dict[Tuple[str, str], Dict[str, list]] = {}
        self.index_family_uom: Dict[Tuple[str, str], Dict[str, list]] = {}

        # Semantic partition stores: family -> {'dates': np.ndarray, 'row_ids': np.ndarray, 'word_svd': np.ndarray, 'char_svd': np.ndarray, 'emb_mat': np.ndarray, 'uoms': np.ndarray, 'comm_codes': np.ndarray, 'qtys': np.ndarray}
        self.family_semantic_store: Dict[str, Dict[str, Any]] = {}
        self.comm_semantic_store: Dict[str, Dict[str, Any]] = {}

        self.spec_engine = None
        self.all_embeddings = None
        self.all_record_dicts = None
        self.all_parsed_identities = None
        self.all_parsed_specs = None
        self._is_indexed = False

    def build_indices(
        self,
        df_full: pd.DataFrame,
        parsed_identities: List[Dict[str, Any]],
        parsed_specs: List[Dict[str, Any]],
        spec_engine: Any,
        all_embeddings: np.ndarray
    ) -> None:
        """
        Builds parallel retrieval indices across all 10 channels strictly from dataset.
        """
        self.spec_engine = spec_engine
        self.all_embeddings = all_embeddings
        self.all_parsed_identities = parsed_identities
        self.all_parsed_specs = parsed_specs

        # Clear existing indices
        self.index_mpn_comm.clear()
        self.index_mpn_global.clear()
        self.index_brand_model_comm.clear()
        self.index_brand_model_fam.clear()
        self.index_model_family_comm.clear()
        self.index_model_family_fam.clear()
        self.index_numeric_specs.clear()
        self.index_comm_uom.clear()
        self.index_family_uom.clear()
        self.family_semantic_store.clear()
        self.comm_semantic_store.clear()

        n_records = len(df_full)
        records = df_full.to_dict(orient="records")
        self.all_record_dicts = records

        # 1. Populate structured index channels (single linear pass, O(N))
        for i in range(n_records):
            r = records[i]
            r_id = r["row_id"]
            d_val = pd.to_datetime(r["award_date_parsed"])
            c_code = str(r["commodity_code"]).strip()
            fam = str(r.get("commodity_family", "")).strip()
            uom = str(r["uom_standardized"]).strip()
            p_val = float(r["target_unit_price"])
            q_val = float(r["quantity_numeric"] if pd.notnull(r["quantity_numeric"]) else 1.0)

            id_d = parsed_identities[i]
            sp_d = parsed_specs[i]

            mpn = id_d.get("parsed_mpn")
            brand = id_d.get("parsed_brand")
            model = id_d.get("parsed_model")
            fam_model = id_d.get("parsed_model_family")

            # Comm + UOM Index
            k_cu = (c_code, uom)
            if k_cu not in self.index_comm_uom:
                self.index_comm_uom[k_cu] = {"dates": [], "prices": [], "quantities": [], "row_ids": []}
            self.index_comm_uom[k_cu]["dates"].append(d_val)
            self.index_comm_uom[k_cu]["prices"].append(p_val)
            self.index_comm_uom[k_cu]["quantities"].append(q_val)
            self.index_comm_uom[k_cu]["row_ids"].append(r_id)

            # Family + UOM Index
            k_fu = (fam, uom)
            if k_fu not in self.index_family_uom:
                self.index_family_uom[k_fu] = {"dates": [], "prices": [], "quantities": [], "row_ids": []}
            self.index_family_uom[k_fu]["dates"].append(d_val)
            self.index_family_uom[k_fu]["prices"].append(p_val)
            self.index_family_uom[k_fu]["quantities"].append(q_val)
            self.index_family_uom[k_fu]["row_ids"].append(r_id)

            # Channel A: MPN Indices
            if mpn:
                k_mpn_c = (c_code, mpn, uom)
                if k_mpn_c not in self.index_mpn_comm:
                    self.index_mpn_comm[k_mpn_c] = {"dates": [], "prices": [], "quantities": [], "row_ids": []}
                self.index_mpn_comm[k_mpn_c]["dates"].append(d_val)
                self.index_mpn_comm[k_mpn_c]["prices"].append(p_val)
                self.index_mpn_comm[k_mpn_c]["quantities"].append(q_val)
                self.index_mpn_comm[k_mpn_c]["row_ids"].append(r_id)

                if mpn not in self.index_mpn_global:
                    self.index_mpn_global[mpn] = {"dates": [], "prices": [], "quantities": [], "row_ids": []}
                self.index_mpn_global[mpn]["dates"].append(d_val)
                self.index_mpn_global[mpn]["prices"].append(p_val)
                self.index_mpn_global[mpn]["quantities"].append(q_val)
                self.index_mpn_global[mpn]["row_ids"].append(r_id)

            # Channel B: Brand + Model Indices
            if brand and model:
                k_bm_c = (c_code, brand, model, uom)
                if k_bm_c not in self.index_brand_model_comm:
                    self.index_brand_model_comm[k_bm_c] = {"dates": [], "prices": [], "quantities": [], "row_ids": []}
                self.index_brand_model_comm[k_bm_c]["dates"].append(d_val)
                self.index_brand_model_comm[k_bm_c]["prices"].append(p_val)
                self.index_brand_model_comm[k_bm_c]["quantities"].append(q_val)
                self.index_brand_model_comm[k_bm_c]["row_ids"].append(r_id)

                k_bm_f = (fam, brand, model)
                if k_bm_f not in self.index_brand_model_fam:
                    self.index_brand_model_fam[k_bm_f] = {"dates": [], "prices": [], "quantities": [], "row_ids": []}
                self.index_brand_model_fam[k_bm_f]["dates"].append(d_val)
                self.index_brand_model_fam[k_bm_f]["prices"].append(p_val)
                self.index_brand_model_fam[k_bm_f]["quantities"].append(q_val)
                self.index_brand_model_fam[k_bm_f]["row_ids"].append(r_id)

            # Channel C: Model Family Indices
            if fam_model:
                k_mf_c = (c_code, fam_model, uom)
                if k_mf_c not in self.index_model_family_comm:
                    self.index_model_family_comm[k_mf_c] = {"dates": [], "prices": [], "quantities": [], "row_ids": []}
                self.index_model_family_comm[k_mf_c]["dates"].append(d_val)
                self.index_model_family_comm[k_mf_c]["prices"].append(p_val)
                self.index_model_family_comm[k_mf_c]["quantities"].append(q_val)
                self.index_model_family_comm[k_mf_c]["row_ids"].append(r_id)

                k_mf_f = (fam, fam_model)
                if k_mf_f not in self.index_model_family_fam:
                    self.index_model_family_fam[k_mf_f] = {"dates": [], "prices": [], "quantities": [], "row_ids": []}
                self.index_model_family_fam[k_mf_f]["dates"].append(d_val)
                self.index_model_family_fam[k_mf_f]["prices"].append(p_val)
                self.index_model_family_fam[k_mf_f]["quantities"].append(q_val)
                self.index_model_family_fam[k_mf_f]["row_ids"].append(r_id)

            # Channel D: Numeric Specifications Indices
            for spec_k in NUMERIC_SPEC_KEYS:
                spec_v = sp_d.get(spec_k)
                if spec_v is not None and not (isinstance(spec_v, float) and math.isnan(spec_v)):
                    rounded_v = round(float(spec_v), 1)
                    k_spec = (fam, spec_k, rounded_v)
                    if k_spec not in self.index_numeric_specs:
                        self.index_numeric_specs[k_spec] = {"dates": [], "prices": [], "quantities": [], "row_ids": []}
                    self.index_numeric_specs[k_spec]["dates"].append(d_val)
                    self.index_numeric_specs[k_spec]["prices"].append(p_val)
                    self.index_numeric_specs[k_spec]["quantities"].append(q_val)
                    self.index_numeric_specs[k_spec]["row_ids"].append(r_id)

        # 2. Build semantic matrix stores partitioned by commodity and family
        d_vals_all = pd.to_datetime(df_full["award_date_parsed"]).values
        sort_order = np.argsort(d_vals_all)
        sorted_dates = d_vals_all[sort_order]
        sorted_rids = df_full["row_id"].values[sort_order]
        sorted_fams = df_full["commodity_family"].values[sort_order]
        sorted_comms = df_full["commodity_code"].values[sort_order]
        sorted_uoms = df_full["uom_standardized"].values[sort_order]
        sorted_qtys = df_full["quantity_numeric"].fillna(1.0).values[sort_order]
        sorted_embs = all_embeddings[sort_order]

        # Extract Word SVD and Char SVD matrices if available
        raw_texts = df_full["product_text_normalized"].fillna("").values
        if spec_engine is not None and hasattr(spec_engine, "word_vec") and spec_engine.word_vec is not None:
            X_w = spec_engine.word_vec.transform(raw_texts)
            X_c = spec_engine.char_vec.transform(raw_texts)
            word_svd_all = normalize(spec_engine.word_svd.transform(X_w)).astype(np.float32)[sort_order]
            char_svd_all = normalize(spec_engine.char_svd.transform(X_c)).astype(np.float32)[sort_order]
        else:
            word_svd_all = sorted_embs
            char_svd_all = sorted_embs

        # Partition by family
        fam_series = pd.Series(sorted_fams)
        for fam_k in fam_series.dropna().unique():
            mask = (sorted_fams == fam_k)
            self.family_semantic_store[fam_k] = {
                "dates": sorted_dates[mask],
                "row_ids": sorted_rids[mask],
                "word_svd": word_svd_all[mask],
                "char_svd": char_svd_all[mask],
                "emb_mat": sorted_embs[mask],
                "uoms": sorted_uoms[mask],
                "comm_codes": sorted_comms[mask],
                "qtys": sorted_qtys[mask]
            }

        # Partition by commodity code
        comm_series = pd.Series(sorted_comms)
        for comm_k in comm_series.dropna().unique():
            mask = (sorted_comms == comm_k)
            self.comm_semantic_store[comm_k] = {
                "dates": sorted_dates[mask],
                "row_ids": sorted_rids[mask],
                "word_svd": word_svd_all[mask],
                "char_svd": char_svd_all[mask],
                "emb_mat": sorted_embs[mask],
                "uoms": sorted_uoms[mask],
                "comm_codes": sorted_comms[mask],
                "qtys": sorted_qtys[mask]
            }

        self._is_indexed = True

    def retrieve_channel_a_exact_identity(
        self,
        q_row: Dict[str, Any],
        q_id: Dict[str, Any],
        q_date: pd.Timestamp,
        q_row_id: int
    ) -> List[Dict[str, Any]]:
        """Channel A: Exact MPN / SKU / Part Number."""
        mpn = q_id.get("parsed_mpn")
        if not mpn:
            return []
        c_code = str(q_row["commodity_code"]).strip()
        uom = str(q_row["uom_standardized"]).strip()

        cands = []
        # Same commodity
        k_c = (c_code, mpn, uom)
        if k_c in self.index_mpn_comm:
            d_list = self.index_mpn_comm[k_c]["dates"]
            pos = bisect.bisect_left(d_list, q_date)
            start_p = max(0, pos - self.max_cands_per_channel)
            for r_id in reversed(self.index_mpn_comm[k_c]["row_ids"][start_p:pos]):
                if r_id != q_row_id:
                    c_dict = dict(self.all_record_dicts[r_id])
                    c_dict["channel"] = "A_EXACT_IDENTITY"
                    c_dict["channel_priority"] = 1
                    cands.append(c_dict)

        # Cross-commodity exact MPN if pool is small
        if len(cands) < 5 and mpn in self.index_mpn_global:
            d_list = self.index_mpn_global[mpn]["dates"]
            pos = bisect.bisect_left(d_list, q_date)
            start_p = max(0, pos - self.max_cands_per_channel)
            for r_id in reversed(self.index_mpn_global[mpn]["row_ids"][start_p:pos]):
                if r_id != q_row_id and r_id not in [c["row_id"] for c in cands]:
                    c_dict = dict(self.all_record_dicts[r_id])
                    c_dict["channel"] = "A_EXACT_IDENTITY"
                    c_dict["channel_priority"] = 1
                    cands.append(c_dict)

        return cands[:self.max_cands_per_channel]

    def retrieve_channel_b_brand_model(
        self,
        q_row: Dict[str, Any],
        q_id: Dict[str, Any],
        q_date: pd.Timestamp,
        q_row_id: int
    ) -> List[Dict[str, Any]]:
        """Channel B: Exact Brand + Model."""
        brand = q_id.get("parsed_brand")
        model = q_id.get("parsed_model")
        if not (brand and model):
            return []
        c_code = str(q_row["commodity_code"]).strip()
        fam = str(q_row.get("commodity_family", "")).strip()
        uom = str(q_row["uom_standardized"]).strip()

        cands = []
        # Same commodity
        k_bm_c = (c_code, brand, model, uom)
        if k_bm_c in self.index_brand_model_comm:
            d_list = self.index_brand_model_comm[k_bm_c]["dates"]
            pos = bisect.bisect_left(d_list, q_date)
            start_p = max(0, pos - self.max_cands_per_channel)
            for r_id in reversed(self.index_brand_model_comm[k_bm_c]["row_ids"][start_p:pos]):
                if r_id != q_row_id:
                    c_dict = dict(self.all_record_dicts[r_id])
                    c_dict["channel"] = "B_BRAND_MODEL"
                    c_dict["channel_priority"] = 2
                    cands.append(c_dict)

        # Same family
        if len(cands) < 10:
            k_bm_f = (fam, brand, model)
            if k_bm_f in self.index_brand_model_fam:
                d_list = self.index_brand_model_fam[k_bm_f]["dates"]
                pos = bisect.bisect_left(d_list, q_date)
                start_p = max(0, pos - self.max_cands_per_channel)
                for r_id in reversed(self.index_brand_model_fam[k_bm_f]["row_ids"][start_p:pos]):
                    if r_id != q_row_id and r_id not in [c["row_id"] for c in cands]:
                        c_dict = dict(self.all_record_dicts[r_id])
                        c_dict["channel"] = "B_BRAND_MODEL"
                        c_dict["channel_priority"] = 2
                        cands.append(c_dict)

        return cands[:self.max_cands_per_channel]

    def retrieve_channel_c_model_family(
        self,
        q_row: Dict[str, Any],
        q_id: Dict[str, Any],
        q_date: pd.Timestamp,
        q_row_id: int
    ) -> List[Dict[str, Any]]:
        """Channel C: Model Family Series Matching."""
        fam_model = q_id.get("parsed_model_family")
        if not fam_model:
            return []
        c_code = str(q_row["commodity_code"]).strip()
        fam = str(q_row.get("commodity_family", "")).strip()
        uom = str(q_row["uom_standardized"]).strip()

        cands = []
        k_mf_c = (c_code, fam_model, uom)
        if k_mf_c in self.index_model_family_comm:
            d_list = self.index_model_family_comm[k_mf_c]["dates"]
            pos = bisect.bisect_left(d_list, q_date)
            start_p = max(0, pos - self.max_cands_per_channel)
            for r_id in reversed(self.index_model_family_comm[k_mf_c]["row_ids"][start_p:pos]):
                if r_id != q_row_id:
                    c_dict = dict(self.all_record_dicts[r_id])
                    c_dict["channel"] = "C_MODEL_FAMILY"
                    c_dict["channel_priority"] = 3
                    cands.append(c_dict)

        if len(cands) < 10:
            k_mf_f = (fam, fam_model)
            if k_mf_f in self.index_model_family_fam:
                d_list = self.index_model_family_fam[k_mf_f]["dates"]
                pos = bisect.bisect_left(d_list, q_date)
                start_p = max(0, pos - self.max_cands_per_channel)
                for r_id in reversed(self.index_model_family_fam[k_mf_f]["row_ids"][start_p:pos]):
                    if r_id != q_row_id and r_id not in [c["row_id"] for c in cands]:
                        c_dict = dict(self.all_record_dicts[r_id])
                        c_dict["channel"] = "C_MODEL_FAMILY"
                        c_dict["channel_priority"] = 3
                        cands.append(c_dict)

        return cands[:self.max_cands_per_channel]

    def retrieve_channel_d_numeric_specs(
        self,
        q_row: Dict[str, Any],
        q_spec: Dict[str, Any],
        q_date: pd.Timestamp,
        q_row_id: int
    ) -> List[Dict[str, Any]]:
        """Channel D: Numeric Specification Matching across hardware & capacity keys."""
        fam = str(q_row.get("commodity_family", "")).strip()
        cands = []

        for spec_k in NUMERIC_SPEC_KEYS:
            spec_v = q_spec.get(spec_k)
            if spec_v is not None and not (isinstance(spec_v, float) and math.isnan(spec_v)):
                rounded_v = round(float(spec_v), 1)
                k_spec = (fam, spec_k, rounded_v)
                if k_spec in self.index_numeric_specs:
                    d_list = self.index_numeric_specs[k_spec]["dates"]
                    pos = bisect.bisect_left(d_list, q_date)
                    start_p = max(0, pos - 15)
                    for r_id in reversed(self.index_numeric_specs[k_spec]["row_ids"][start_p:pos]):
                        if r_id != q_row_id and r_id not in [c["row_id"] for c in cands]:
                            c_dict = dict(self.all_record_dicts[r_id])
                            c_dict["channel"] = "D_NUMERIC_SPECS"
                            c_dict["channel_priority"] = 4
                            cands.append(c_dict)

        return cands[:self.max_cands_per_channel]

    def retrieve_channel_e_word_tfidf(
        self,
        q_row: Dict[str, Any],
        q_word_vec: np.ndarray,
        q_date: pd.Timestamp,
        q_row_id: int
    ) -> List[Dict[str, Any]]:
        """Channel E: Word-Level TF-IDF / SVD Cosine Retrieval."""
        c_code = str(q_row["commodity_code"]).strip()
        fam = str(q_row.get("commodity_family", "")).strip()
        store = self.comm_semantic_store.get(c_code) or self.family_semantic_store.get(fam)
        if store is None or q_word_vec is None:
            return []

        d_list = store["dates"]
        pos = bisect.bisect_left(d_list, q_date)
        if pos <= 0:
            return []

        w_mat = store["word_svd"][:pos]
        rids = store["row_ids"][:pos]
        sims = np.dot(w_mat, q_word_vec)
        top_idx = np.argsort(sims)[::-1][:self.max_cands_per_channel]

        cands = []
        for idx in top_idx:
            sim = float(sims[idx])
            if sim < 0.10:
                continue
            r_id = rids[idx]
            if r_id != q_row_id:
                c_dict = dict(self.all_record_dicts[r_id])
                c_dict["channel"] = "E_WORD_TFIDF"
                c_dict["channel_priority"] = 5
                c_dict["word_sim"] = sim
                cands.append(c_dict)
        return cands

    def retrieve_channel_f_char_ngrams(
        self,
        q_row: Dict[str, Any],
        q_char_vec: np.ndarray,
        q_date: pd.Timestamp,
        q_row_id: int
    ) -> List[Dict[str, Any]]:
        """Channel F: Character N-Gram Subword & Model String Retrieval."""
        c_code = str(q_row["commodity_code"]).strip()
        fam = str(q_row.get("commodity_family", "")).strip()
        store = self.comm_semantic_store.get(c_code) or self.family_semantic_store.get(fam)
        if store is None or q_char_vec is None:
            return []

        d_list = store["dates"]
        pos = bisect.bisect_left(d_list, q_date)
        if pos <= 0:
            return []

        c_mat = store["char_svd"][:pos]
        rids = store["row_ids"][:pos]
        sims = np.dot(c_mat, q_char_vec)
        top_idx = np.argsort(sims)[::-1][:self.max_cands_per_channel]

        cands = []
        for idx in top_idx:
            sim = float(sims[idx])
            if sim < 0.10:
                continue
            r_id = rids[idx]
            if r_id != q_row_id:
                c_dict = dict(self.all_record_dicts[r_id])
                c_dict["channel"] = "F_CHAR_NGRAMS"
                c_dict["channel_priority"] = 5
                c_dict["char_sim"] = sim
                cands.append(c_dict)
        return cands

    def retrieve_channel_g_lsi_svd(
        self,
        q_row: Dict[str, Any],
        q_emb: np.ndarray,
        q_date: pd.Timestamp,
        q_row_id: int
    ) -> List[Dict[str, Any]]:
        """Channel G: LSI / SVD Latent Semantic Retrieval."""
        c_code = str(q_row["commodity_code"]).strip()
        fam = str(q_row.get("commodity_family", "")).strip()
        store = self.comm_semantic_store.get(c_code) or self.family_semantic_store.get(fam)
        if store is None or q_emb is None:
            return []

        d_list = store["dates"]
        pos = bisect.bisect_left(d_list, q_date)
        if pos <= 0:
            return []

        emb_mat = store["emb_mat"][:pos]
        rids = store["row_ids"][:pos]
        sims = np.dot(emb_mat, q_emb)
        top_idx = np.argsort(sims)[::-1][:self.max_cands_per_channel]

        cands = []
        for idx in top_idx:
            sim = float(sims[idx])
            if sim < 0.10:
                continue
            r_id = rids[idx]
            if r_id != q_row_id:
                c_dict = dict(self.all_record_dicts[r_id])
                c_dict["channel"] = "G_LSI_SVD"
                c_dict["channel_priority"] = 6
                c_dict["lsi_sim"] = sim
                cands.append(c_dict)
        return cands

    def retrieve_channel_h_semantic_embedding(
        self,
        q_row: Dict[str, Any],
        q_emb: np.ndarray,
        q_date: pd.Timestamp,
        q_row_id: int
    ) -> List[Dict[str, Any]]:
        """Channel H: Dense Composite Semantic Embedding Retrieval (Family-Wide)."""
        fam = str(q_row.get("commodity_family", "")).strip()
        store = self.family_semantic_store.get(fam)
        if store is None or q_emb is None:
            return []

        d_list = store["dates"]
        pos = bisect.bisect_left(d_list, q_date)
        if pos <= 0:
            return []

        emb_mat = store["emb_mat"][:pos]
        rids = store["row_ids"][:pos]
        uoms = store["uoms"][:pos]
        q_uom = str(q_row["uom_standardized"]).strip()

        cos_sims = np.dot(emb_mat, q_emb)
        uom_match = np.array([1.0 if u == q_uom else 0.0 for u in uoms])
        composite_scores = cos_sims * 0.75 + uom_match * 0.25
        top_idx = np.argsort(composite_scores)[::-1][:self.max_cands_per_channel]

        cands = []
        for idx in top_idx:
            sim = float(cos_sims[idx])
            if sim < 0.10:
                continue
            r_id = rids[idx]
            if r_id != q_row_id:
                c_dict = dict(self.all_record_dicts[r_id])
                c_dict["channel"] = "H_SEMANTIC_EMB"
                c_dict["channel_priority"] = 6
                c_dict["emb_sim"] = sim
                cands.append(c_dict)
        return cands

    def retrieve_channel_i_commodity_uom(
        self,
        q_row: Dict[str, Any],
        q_date: pd.Timestamp,
        q_row_id: int
    ) -> List[Dict[str, Any]]:
        """Channel I: Commodity + UOM Broad Fallback."""
        c_code = str(q_row["commodity_code"]).strip()
        uom = str(q_row["uom_standardized"]).strip()
        fam = str(q_row.get("commodity_family", "")).strip()

        cands = []
        k_cu = (c_code, uom)
        if k_cu in self.index_comm_uom:
            d_list = self.index_comm_uom[k_cu]["dates"]
            pos = bisect.bisect_left(d_list, q_date)
            start_p = max(0, pos - self.max_cands_per_channel)
            for r_id in reversed(self.index_comm_uom[k_cu]["row_ids"][start_p:pos]):
                if r_id != q_row_id:
                    c_dict = dict(self.all_record_dicts[r_id])
                    c_dict["channel"] = "I_COMM_UOM"
                    c_dict["channel_priority"] = 7
                    cands.append(c_dict)

        if len(cands) < 5:
            k_fu = (fam, uom)
            if k_fu in self.index_family_uom:
                d_list = self.index_family_uom[k_fu]["dates"]
                pos = bisect.bisect_left(d_list, q_date)
                start_p = max(0, pos - self.max_cands_per_channel)
                for r_id in reversed(self.index_family_uom[k_fu]["row_ids"][start_p:pos]):
                    if r_id != q_row_id and r_id not in [c["row_id"] for c in cands]:
                        c_dict = dict(self.all_record_dicts[r_id])
                        c_dict["channel"] = "I_COMM_UOM"
                        c_dict["channel_priority"] = 7
                        cands.append(c_dict)

        return cands[:self.max_cands_per_channel]

    def retrieve_channel_j_quantity_aware(
        self,
        q_row: Dict[str, Any],
        q_date: pd.Timestamp,
        q_row_id: int
    ) -> List[Dict[str, Any]]:
        """Channel J: Quantity-Aware Preference Retrieval."""
        c_code = str(q_row["commodity_code"]).strip()
        uom = str(q_row["uom_standardized"]).strip()
        fam = str(q_row.get("commodity_family", "")).strip()
        q_qty = max(float(q_row.get("quantity_numeric", 1.0)), 1.0)
        log_q_qty = math.log10(q_qty)

        store = self.comm_semantic_store.get(c_code) or self.family_semantic_store.get(fam)
        if store is None:
            return []

        d_list = store["dates"]
        pos = bisect.bisect_left(d_list, q_date)
        if pos <= 0:
            return []

        rids = store["row_ids"][:pos]
        qtys = store["qtys"][:pos]
        log_c_qtys = np.log10(np.maximum(qtys, 1.0))
        qty_diffs = np.abs(log_c_qtys - log_q_qty)

        # Prefer small quantity gap (< 0.50) with recent dates
        good_qty_mask = np.where(qty_diffs <= 0.50)[0]
        cands = []
        if len(good_qty_mask) > 0:
            recent_good = good_qty_mask[-self.max_cands_per_channel:]
            for idx in reversed(recent_good):
                r_id = rids[idx]
                if r_id != q_row_id:
                    c_dict = dict(self.all_record_dicts[r_id])
                    c_dict["channel"] = "J_QUANTITY_AWARE"
                    c_dict["channel_priority"] = 5
                    cands.append(c_dict)

        return cands[:self.max_cands_per_channel]

    def retrieve_multi_view_candidates(
        self,
        q_row: Dict[str, Any],
        q_id: Dict[str, Any],
        q_spec: Dict[str, Any],
        q_emb: np.ndarray,
        q_word_vec: Optional[np.ndarray],
        q_char_vec: Optional[np.ndarray],
        q_row_id: int,
        enabled_channels: Optional[Set[str]] = None,
        max_total_k: int = 100,
        filter_mode: str = "CONSERVATIVE"
    ) -> List[Dict[str, Any]]:
        """
        Executes parallel candidate retrieval across enabled channels and computes their union.
        100% causal: candidate.award_date < query.award_date enforced for every item.
        """
        q_date = pd.to_datetime(q_row["award_date_parsed"])
        if q_date >= pd.to_datetime("2025-01-01"):
            raise RuntimeError("FROZEN TEST ACCESS FORBIDDEN: Date >= 2025-01-01")

        if enabled_channels is None:
            enabled_channels = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}

        channel_cands: Dict[str, List[Dict[str, Any]]] = {}

        if "A" in enabled_channels:
            channel_cands["A"] = self.retrieve_channel_a_exact_identity(q_row, q_id, q_date, q_row_id)
        if "B" in enabled_channels:
            channel_cands["B"] = self.retrieve_channel_b_brand_model(q_row, q_id, q_date, q_row_id)
        if "C" in enabled_channels:
            channel_cands["C"] = self.retrieve_channel_c_model_family(q_row, q_id, q_date, q_row_id)
        if "D" in enabled_channels:
            channel_cands["D"] = self.retrieve_channel_d_numeric_specs(q_row, q_spec, q_date, q_row_id)
        if "E" in enabled_channels and q_word_vec is not None:
            channel_cands["E"] = self.retrieve_channel_e_word_tfidf(q_row, q_word_vec, q_date, q_row_id)
        if "F" in enabled_channels and q_char_vec is not None:
            channel_cands["F"] = self.retrieve_channel_f_char_ngrams(q_row, q_char_vec, q_date, q_row_id)
        if "G" in enabled_channels and q_emb is not None:
            channel_cands["G"] = self.retrieve_channel_g_lsi_svd(q_row, q_emb, q_date, q_row_id)
        if "H" in enabled_channels and q_emb is not None:
            channel_cands["H"] = self.retrieve_channel_h_semantic_embedding(q_row, q_emb, q_date, q_row_id)
        if "I" in enabled_channels:
            channel_cands["I"] = self.retrieve_channel_i_commodity_uom(q_row, q_date, q_row_id)
        if "J" in enabled_channels:
            channel_cands["J"] = self.retrieve_channel_j_quantity_aware(q_row, q_date, q_row_id)

        # Multi-View Union & Deduplication
        union_pool: Dict[int, Dict[str, Any]] = {}

        for ch_key, cand_list in channel_cands.items():
            for c in cand_list:
                r_id = c["row_id"]
                c_date = pd.to_datetime(c["award_date_parsed"])
                if c_date >= q_date:
                    raise RuntimeError(f"TEMPORAL CAUSALITY VIOLATION: Candidate {c_date} >= Query {q_date}")

                if r_id not in union_pool:
                    c_copy = dict(c)
                    c_copy["retrieval_channels"] = [ch_key]
                    c_copy["channel_count"] = 1
                    union_pool[r_id] = c_copy
                else:
                    if ch_key not in union_pool[r_id]["retrieval_channels"]:
                        union_pool[r_id]["retrieval_channels"].append(ch_key)
                        union_pool[r_id]["channel_count"] += 1
                        # Update channel priority to best (lowest numeric value)
                        union_pool[r_id]["channel_priority"] = min(
                            union_pool[r_id].get("channel_priority", 99),
                            c.get("channel_priority", 99)
                        )

        # Conservative Compatibility Filtering
        final_cands = list(union_pool.values())
        if filter_mode != "NONE" and len(final_cands) > 10:
            filtered_cands = []
            for c in final_cands:
                c_id = self.all_parsed_identities[c["row_id"]]
                c_spec = self.all_parsed_specs[c["row_id"]]

                # Check critical conflict
                has_crit_conflict = False
                for k in ["ports", "pipe_diameter_in", "horsepower_hp", "ram_gb"]:
                    qv = q_spec.get(k)
                    cv = c_spec.get(k)
                    if qv is not None and cv is not None:
                        try:
                            if abs(float(qv) - float(cv)) / max(float(qv), 1e-4) > 0.35:
                                has_crit_conflict = True
                                break
                        except (ValueError, TypeError):
                            pass

                if filter_mode == "CONSERVATIVE":
                    # Reject only if critical conflict AND not exact MPN/model
                    if has_crit_conflict and ("A" not in c["retrieval_channels"]) and ("B" not in c["retrieval_channels"]):
                        continue
                elif filter_mode == "MODERATE":
                    if has_crit_conflict and ("A" not in c["retrieval_channels"]):
                        continue
                elif filter_mode == "STRICT":
                    if has_crit_conflict:
                        continue

                filtered_cands.append(c)

            if len(filtered_cands) >= 5:
                final_cands = filtered_cands

        # Sort order: multi-channel consensus (channel_count desc, channel_priority asc, recency desc)
        def sort_key(x):
            prio = x.get("channel_priority", 99)
            ch_cnt = x.get("channel_count", 1)
            d_val = pd.to_datetime(x["award_date_parsed"]).timestamp()
            return (-ch_cnt, prio, -d_val)

        final_cands.sort(key=sort_key)
        return final_cands[:max_total_k]
