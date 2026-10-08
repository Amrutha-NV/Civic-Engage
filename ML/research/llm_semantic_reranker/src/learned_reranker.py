"""
Experiment 7 - Learned, Rank-1-aware Historical Price Reranker

What changed relative to Experiment 6 (the gaps this closes)
------------------------------------------------------------
 1. Hand-set weights and gate margins  ->  LightGBM trained with grouped
    cross-validation (out-of-fold predictions only).
 2. Fixed override margin               ->  margin chosen by NESTED CV on
    training folds only, never on the fold being scored.
 3. No candidate-level labels           ->  full (query, candidate) table with an
    evaluation-only is_within_10pct column and OOF scores is saved.
 4. Price evidence was raw counts       ->  semantic-weighted support, log-price
    KDE density, weighted median, top-3 semantic median.
 5. No variant handling                 ->  duplicate-description groups,
    near-duplicate groups, pack-size parsing, price-scale-factor mismatch.
 6. Target quantity / UOM ignored       ->  used as target-side context.
 7. Rank-1 reliability not modelled     ->  Rank-1 context and candidate-minus-Rank-1
    difference features, so the model learns WHEN to trust Rank-1.
 8. No significance testing             ->  bootstrap CIs and McNemar test.

Leakage rules
-------------
 * actual_unit_price / APE / is_within_10pct are used ONLY as training labels on
   training folds and for evaluation. They are never features.
 * Target price strings ($xx.xx) are stripped before embedding / token features,
   and results are also reported on a clean subset with no price strings.

Pre-specified primary policy: "LGBM binary + nested-CV gate".
All other rows are reported for comparison, not for selection.

Usage
-----
    python learned_rank1_reranker_exp7.py --debug 300         # quick check
    python learned_rank1_reranker_exp7.py                     # full run
    python learned_rank1_reranker_exp7.py --resume            # reuse embeddings
    python learned_rank1_reranker_exp7.py --embedder tfidf    # offline smoke test
"""

from pathlib import Path
import argparse
import json
import re
import time
import warnings

import numpy as np
import pandas as pd

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
except IndexError:
    PROJECT_ROOT = Path.cwd()

DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "ML" / "research" / "procurement_benchmark_topk"
    / "frozen_evaluation_dataset.parquet"
)
BASE_OUTPUT_DIR = (
    PROJECT_ROOT / "ML" / "research" / "llm_semantic_reranker"
    / "results" / "learned_reranker_exp7"
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64
K = 10
GATE_GRID = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]

LGBM_COMMON = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=-1,
    verbose=-1,
)

REQUIRED_COLUMNS = [
    "query_index", "target_description", "target_quantity",
    "target_unit_of_measure", "target_brand", "target_model",
    "target_commodity_code", "target_commodity_family",
    "candidate_rank", "candidate_description", "candidate_unit_price",
    "candidate_score", "actual_unit_price",
]

PRICE_RE = re.compile(r"[$\u20ac\u00a3]\s?\d|\b(?:usd|inr|eur|gbp)\s*\d", re.IGNORECASE)


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(text):
    """Strip explicit currency strings, lowercase, collapse whitespace."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"[$\u20ac\u00a3]\s?\d[\d,]*(?:\.\d+)?", " ", text)
    text = re.sub(
        r"\b(?:USD|INR|EUR|GBP)\s*\d[\d,]*(?:\.\d+)?\b", " ",
        text, flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", text).strip().lower()


def normalize_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def boundary_match(target_value, candidate_text):
    """1.0 if target value appears as a whole token/phrase in candidate text."""
    t = normalize_value(target_value)
    if not t or t == "nan":
        return 0.0
    c = normalize_value(candidate_text)
    pattern = r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])"
    return 1.0 if re.search(pattern, c) else 0.0


def numeric_tokens(text):
    text = normalize_value(text)
    nums = re.findall(
        r"\b\d+(?:\.\d+)?(?:gb|tb|mb|inch|in|mm|cm|w|v|hz)?\b",
        text, flags=re.IGNORECASE,
    )
    alnum = re.findall(r"\b[a-z]+\d+[a-z0-9-]*\b", text, flags=re.IGNORECASE)
    return set(nums + alnum)


_PACK_PATTERNS = [
    re.compile(r"\b(?:pack|pkg|box|case|carton|bag|set|bundle|roll|ream)s?\s+of\s+(\d+)\b"),
    re.compile(r"\b(\d+)\s*/\s*(?:pk|pack|case|cs|box|bx|ct)\b"),
    re.compile(r"\b(\d+)\s*-?\s*(?:pk|pack|packs|pcs|pc|pieces|piece|ct|count|ea|each|units|unit)\b"),
    re.compile(r"\bper\s+(\d+)\b"),
]


def parse_pack_size(text):
    """Best-effort pack size from cleaned lowercase text. NaN if unknown."""
    if not text:
        return np.nan
    if re.search(r"\bdozen\b", text):
        return 12.0
    if re.search(r"\bgross\b", text):
        return 144.0
    for pat in _PACK_PATTERNS:
        m = pat.search(text)
        if m:
            v = float(m.group(1))
            if 1 <= v <= 100000:
                return v
    return np.nan


# ============================================================
# DATA
# ============================================================

def validate_dataset(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    g = df.groupby("query_index")["candidate_rank"].agg(["size", "min", "max", "nunique"])
    bad = g[(g["size"] != K) | (g["min"] != 1) | (g["max"] != K) | (g["nunique"] != K)]
    if len(bad):
        raise ValueError(
            f"{len(bad)} queries do not have exactly ranks 1..{K}. "
            f"Examples: {bad.head().index.tolist()}"
        )
    print("[OK] Dataset validation passed.")
    print(f"[INFO] Rows: {len(df):,} | Queries: {df['query_index'].nunique():,}")


def load_dataset(path, debug_n):
    df = pd.read_parquet(path)
    validate_dataset(df)
    if debug_n > 0:
        keep = df["query_index"].drop_duplicates().iloc[:debug_n]
        df = df[df["query_index"].isin(keep)].copy()
        print(f"[DEBUG] Using first {len(keep)} queries ({len(df):,} rows).")
    df = df.sort_values(["query_index", "candidate_rank"], kind="mergesort")
    return df.reset_index(drop=True)


# ============================================================
# EMBEDDINGS
# ============================================================

class Embedder:
    def __init__(self, kind, fit_texts, seed):
        self.kind = kind
        if kind == "minilm":
            from sentence_transformers import SentenceTransformer
            print("[INFO] Loading semantic model...")
            self.model = SentenceTransformer(EMBEDDING_MODEL)
        elif kind == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            m = self.vec.fit_transform(fit_texts)
            nc = int(max(2, min(128, m.shape[1] - 1, m.shape[0] - 1)))
            self.svd = TruncatedSVD(nc, random_state=seed).fit(m)
        else:
            raise ValueError(f"Unknown embedder: {kind}")

    def encode(self, texts):
        if self.kind == "minilm":
            return self.model.encode(
                texts, batch_size=BATCH_SIZE, show_progress_bar=True,
                normalize_embeddings=True,
            ).astype(np.float32)
        z = self.svd.transform(self.vec.transform(texts))
        z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-9)
        return z.astype(np.float32)


def compute_semantic(df, tgt_clean, cand_clean, kind, cache_path, resume, seed):
    """Returns sem_tc (Q,K) target-candidate cosine and sem_cc (Q,K,K)."""
    Q = len(df) // K
    qids = df["query_index"].to_numpy()[::K]

    if resume and cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        if (
            z["sem_tc"].shape == (Q, K)
            and str(z["embedder"]) == kind
            and np.array_equal(z["query_ids"], qids)
        ):
            print("[RESUME] Loaded cached semantic arrays.")
            return z["sem_tc"], z["sem_cc"]
        print("[RESUME] Cache does not match this run; recomputing.")

    t_codes, t_unique = pd.factorize(pd.Series(tgt_clean))
    c_codes, c_unique = pd.factorize(pd.Series(cand_clean))
    embedder = Embedder(kind, list(t_unique) + list(c_unique), seed)

    print(f"[INFO] Encoding {len(t_unique):,} unique target descriptions...")
    t_emb = embedder.encode(list(t_unique))[t_codes]                 # (Q, D)
    print(f"[INFO] Encoding {len(c_unique):,} unique candidate descriptions...")
    c_emb = embedder.encode(list(c_unique))[c_codes].reshape(Q, K, -1)

    sem_tc = np.einsum("qd,qkd->qk", t_emb, c_emb).astype(np.float32)
    sem_cc = np.einsum("qid,qjd->qij", c_emb, c_emb).astype(np.float32)

    np.savez_compressed(
        cache_path, sem_tc=sem_tc, sem_cc=sem_cc,
        embedder=np.array(kind), query_ids=qids,
    )
    return sem_tc, sem_cc


# ============================================================
# FEATURES
# ============================================================

def row_rank_desc(a):
    """0 = largest value in row."""
    return np.argsort(np.argsort(-a, axis=1, kind="stable"), axis=1, kind="stable")


def softmax_w(s, tau):
    z = (s - s.max(axis=1, keepdims=True)) / tau
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def build_features(df, sem_tc, sem_cc, tgt_clean, cand_clean):
    """
    All features are computed from the frozen artifact only, per query,
    WITHOUT actual_unit_price. Returns (feature DataFrame, context dict).
    """
    Q = len(df) // K
    F = {}

    def put(name, arr):
        a = np.asarray(arr, dtype=np.float64)
        if a.shape != (Q, K):
            a = np.broadcast_to(a.reshape(Q, -1) if a.ndim == 1 else a, (Q, K))
        F[name] = np.array(a, dtype=np.float64).reshape(-1)

    ar = np.arange(Q)

    # ---------------- raw columns ----------------
    rank = df["candidate_rank"].to_numpy(float).reshape(Q, K)
    score = pd.to_numeric(df["candidate_score"], errors="coerce").fillna(0.0)
    score = score.to_numpy(float).reshape(Q, K)

    price = pd.to_numeric(df["candidate_unit_price"], errors="coerce").to_numpy(float)
    price = price.reshape(Q, K)
    bad = ~np.isfinite(price)
    if bad.any():
        print(f"[WARN] {int(bad.sum())} non-finite candidate prices; filling with pool median.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med_fill = np.nanmedian(np.where(bad, np.nan, price), axis=1, keepdims=True)
        med_fill = np.where(np.isfinite(med_fill), med_fill, 1.0)
        price = np.where(bad, med_fill, price)

    lp = np.log(np.maximum(price, 1e-9))
    med_lp = np.median(lp, axis=1, keepdims=True)
    lr = lp - med_lp

    # ---------------- retrieval evidence ----------------
    put("rank", rank)
    put("log_rank", np.log(rank))
    put("cand_score", score)
    put("score_gap_rank1", score - score[:, :1])
    put("score_z", (score - score.mean(1, keepdims=True))
        / np.maximum(score.std(1, keepdims=True), 1e-9))
    put("score_rel_max", score / np.maximum(np.abs(score).max(1, keepdims=True), 1e-9))
    put("score_range", (score.max(1) - score.min(1))[:, None])

    # ---------------- semantic evidence ----------------
    sem = sem_tc.astype(np.float64)
    put("sem", sem)
    put("sem_rank", row_rank_desc(sem))
    put("sem_gap_best", sem - sem.max(1, keepdims=True))
    put("sem_gap_rank1", sem - sem[:, :1])
    put("sem_z", (sem - sem.mean(1, keepdims=True)) / np.maximum(sem.std(1, keepdims=True), 1e-9))
    rng_s = sem.max(1, keepdims=True) - sem.min(1, keepdims=True)
    put("sem_minmax", np.where(rng_s < 1e-9, 1.0, (sem - sem.min(1, keepdims=True)) / np.maximum(rng_s, 1e-9)))
    put("sem_pool_std", sem.std(1)[:, None])
    put("cand_centrality", (sem_cc.sum(2) - 1.0) / (K - 1))

    # ---------------- structured / specification evidence ----------------
    n_rows = len(df)
    tb = df["target_brand"].to_numpy()
    tm = df["target_model"].to_numpy()
    tc = df["target_commodity_code"].to_numpy()
    raw_c = df["candidate_description"].to_numpy()

    brand = np.array([boundary_match(a, b) for a, b in zip(tb, raw_c)]).reshape(Q, K)
    model = np.array([boundary_match(a, b) for a, b in zip(tm, raw_c)]).reshape(Q, K)
    code = np.array([boundary_match(a, b) for a, b in zip(tc, raw_c)]).reshape(Q, K)
    put("brand_match", brand)
    put("model_match", model)
    put("code_match", code)

    num_cov = np.zeros((Q, K))
    num_jac = np.zeros((Q, K))
    tok_jac = np.zeros((Q, K))
    exact = np.zeros((Q, K))
    pack_t = np.full((Q, K), np.nan)
    pack_c = np.full((Q, K), np.nan)
    tgt_ntok = np.zeros(Q)
    tgt_len = np.zeros(Q)
    cand_len = np.zeros((Q, K))

    for q in range(Q):
        t = tgt_clean[q]
        tn = numeric_tokens(t)
        tw = set(t.split())
        tp = parse_pack_size(t)
        tgt_ntok[q] = len(tn)
        tgt_len[q] = len(tw)
        for k in range(K):
            c = cand_clean[q * K + k]
            cn = numeric_tokens(c)
            cw = set(c.split())
            inter = len(tn & cn)
            union = len(tn | cn)
            num_cov[q, k] = inter / len(tn) if tn else 0.0
            num_jac[q, k] = inter / union if union else 0.0
            tok_union = len(tw | cw)
            tok_jac[q, k] = len(tw & cw) / tok_union if tok_union else 0.0
            exact[q, k] = float(t == c and t != "")
            pack_t[q, k] = tp
            pack_c[q, k] = parse_pack_size(c)
            cand_len[q, k] = len(cw)

    put("num_cov", num_cov)
    put("num_jaccard", num_jac)
    put("tok_jaccard", tok_jac)
    put("exact_desc_match", exact)
    put("tgt_n_numeric_tokens", tgt_ntok[:, None])
    put("tgt_len_tokens", tgt_len[:, None])
    put("cand_len_tokens", cand_len)

    both = np.isfinite(pack_t) & np.isfinite(pack_c)
    with np.errstate(divide="ignore", invalid="ignore"):
        pack_lr = np.where(both, np.log(pack_c / pack_t), np.nan)
    put("pack_target", pack_t)
    put("pack_cand", pack_c)
    put("pack_both_known", both.astype(float))
    put("pack_log_ratio", pack_lr)
    put("pack_mismatch", (both & (pack_t != pack_c)).astype(float))

    # ---------------- historical price evidence ----------------
    put("log_price", lp)
    put("lr_median", lr)
    put("abs_lr_median", np.abs(lr))
    asc_rank = np.argsort(np.argsort(lp, axis=1, kind="stable"), axis=1, kind="stable")
    put("price_pct_rank", asc_rank / (K - 1))

    rel = np.abs(price[:, :, None] - price[:, None, :]) / np.maximum(np.abs(price[:, :, None]), 1e-9)
    for t in (0.05, 0.10, 0.20):
        put(f"support_{int(round(t * 100))}", (rel <= t).sum(2))

    q25, q50, q75 = np.percentile(price, [25, 50, 75], axis=1)
    put("pool_iqr_rel", ((q75 - q25) / np.maximum(np.abs(q50), 1e-9))[:, None])
    put("pool_lp_std", lp.std(1)[:, None])
    put("pool_lp_range", (lp.max(1) - lp.min(1))[:, None])

    # semantic-weighted support and KDE in log-price space
    diff = lp[:, :, None] - lp[:, None, :]
    for tau in (0.05, 0.10):
        w = softmax_w(sem, tau)
        tag = f"t{int(round(tau * 100)):02d}"
        put(f"wsupport10_{tag}", (w[:, None, :] * (rel <= 0.10)).sum(2))

    w10 = softmax_w(sem, 0.10)
    for h in (0.05, 0.10):
        kern = np.exp(-0.5 * (diff / h) ** 2)
        dens_w = (w10[:, None, :] * kern).sum(2)
        dens_u = kern.mean(2)
        htag = f"h{int(round(h * 100)):02d}"
        put(f"dens_w_{htag}", dens_w)
        put(f"dens_u_{htag}", dens_u)
        put(f"dens_w_rank_{htag}", row_rank_desc(dens_w))
        put(f"dens_w_rel_{htag}", dens_w / np.maximum(dens_w.max(1, keepdims=True), 1e-12))
        if h == 0.05:
            dens_w_main = dens_w

    order = np.argsort(lp, axis=1, kind="stable")
    lps = np.take_along_axis(lp, order, 1)
    ws = np.take_along_axis(w10, order, 1)
    wmed = lps[ar, (np.cumsum(ws, axis=1) >= 0.5).argmax(1)][:, None]
    put("dist_wmedian", np.abs(lp - wmed))

    top3 = np.argsort(-sem, axis=1, kind="stable")[:, :3]
    top3_med = np.median(np.take_along_axis(lp, top3, 1), axis=1, keepdims=True)
    put("dist_top3_sem_median", np.abs(lp - top3_med))

    # ---------------- duplicate / variant structure ----------------
    codes = pd.factorize(pd.Series(cand_clean))[0].reshape(Q, K)
    eq = codes[:, :, None] == codes[:, None, :]
    dup_count = eq.sum(2)
    lp_b = np.broadcast_to(lp[:, None, :], (Q, K, K))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gmed = np.nanmedian(np.where(eq, lp_b, np.nan), axis=2)
        gstd = np.nanstd(np.where(eq, lp_b, np.nan), axis=2)
        near = sem_cc >= 0.97
        nmed = np.nanmedian(np.where(near, lp_b, np.nan), axis=2)
    put("dup_count", dup_count)
    put("dup_dist_group_median", np.abs(lp - gmed))
    put("dup_group_lp_std", gstd)
    put("n_distinct_desc", (1.0 / dup_count).sum(1)[:, None])
    put("near_dup_count", near.sum(2))
    put("near_dist_group_median", np.abs(lp - nmed))

    # ---------------- unit-scale mismatch ----------------
    facs = np.log(np.array([2, 3, 4, 5, 6, 8, 10, 12, 20, 24, 25, 50, 100, 1000], float))
    d_scale = np.abs(np.abs(lr)[:, :, None] - facs[None, None, :]).min(2)
    put("scale_factor_dist", d_scale)
    put("near_scale_factor", ((d_scale < 0.06) & (np.abs(lr) > 0.5)).astype(float))

    # ---------------- Rank-1 context and differences ----------------
    sup10 = F["support_10"].reshape(Q, K)
    put("model_diff_r1", model - model[:, :1])
    put("brand_diff_r1", brand - brand[:, :1])
    put("numcov_diff_r1", num_cov - num_cov[:, :1])
    put("support10_diff_r1", sup10 - sup10[:, :1])
    put("dens_diff_r1", dens_w_main - dens_w_main[:, :1])
    put("lp_vs_rank1", lp - lp[:, :1])
    put("abs_lp_vs_rank1", np.abs(lp - lp[:, :1]))
    put("r1_model_match", model[:, :1])
    put("r1_brand_match", brand[:, :1])
    put("r1_support10", sup10[:, :1])
    put("r1_sem", sem[:, :1])
    put("r1_dens_rel", (dens_w_main / np.maximum(dens_w_main.max(1, keepdims=True), 1e-12))[:, :1])
    put("r1_dup_count", dup_count[:, :1])

    # ---------------- target-side context ----------------
    qty = pd.to_numeric(df["target_quantity"], errors="coerce").to_numpy(float).reshape(Q, K)[:, 0]
    put("log_target_qty", np.log1p(np.where(qty >= 0, qty, np.nan))[:, None])
    uom = df["target_unit_of_measure"].map(normalize_value).to_numpy().reshape(Q, K)[:, 0]
    uom_freq = pd.Series(uom).map(pd.Series(uom).value_counts(normalize=True)).to_numpy(float)
    put("target_uom_freq", uom_freq[:, None])
    put("tgt_has_brand", np.array([float(bool(normalize_value(x))) for x in tb]).reshape(Q, K)[:, :1])
    put("tgt_has_model", np.array([float(bool(normalize_value(x))) for x in tm]).reshape(Q, K)[:, :1])

    feats = pd.DataFrame(F)

    forbidden = ("actual", "ape", "within", "label", "error")
    leaks = [c for c in feats.columns if any(f in c.lower() for f in forbidden)]
    if leaks:
        raise RuntimeError(f"Feature names look like leakage: {leaks}")

    ctx = {
        "price": price, "score": score, "sem": sem, "brand": brand,
        "model": model, "num_cov": num_cov, "support10": sup10,
        "pool_median": np.median(price, axis=1, keepdims=True),
    }
    return feats, ctx


# ============================================================
# BASELINE POLICIES
# ============================================================

def exp6_rule_policy(ctx):
    """Vectorised re-implementation of the Experiment 6 hand-built gate."""
    sem, score, price = ctx["sem"], ctx["score"], ctx["price"]

    def mm(a):
        lo, hi = a.min(1, keepdims=True), a.max(1, keepdims=True)
        return np.where(hi - lo < 1e-9, 1.0, (a - lo) / np.maximum(hi - lo, 1e-9))

    sem_n, retr_n = mm(sem), mm(score)
    ident = 0.5 * sem_n + 0.2 * ctx["brand"] + 0.2 * ctx["model"] + 0.1 * ctx["num_cov"]
    med = ctx["pool_median"]
    prox = 1.0 / (1.0 + np.abs(price - med) / np.maximum(np.abs(med), 1e-9))
    hist = 0.5 * ctx["support10"] / 10.0 + 0.5 * prox
    alt = 0.55 * ident + 0.30 * hist + 0.15 * retr_n

    Q = alt.shape[0]
    ar = np.arange(Q)
    bi = alt[:, 1:].argmax(1) + 1
    margin = alt[ar, bi] - alt[:, 0]
    id_gain = ident[ar, bi] - ident[:, 0]
    price_gain = hist[ar, bi] - hist[:, 0]
    sem_gain = sem_n[ar, bi] - sem_n[:, 0]

    need = np.where(ident[:, 0] >= 0.75, 0.08 + 0.10, 0.08)
    ok = (alt[ar, bi] >= 0.55) & (margin >= need) & (
        (id_gain >= 0.08) | (price_gain >= 0.12) | (sem_gain >= 0.08)
    )
    return np.where(ok, bi, 0)


def gated_selection(P, margin):
    """Override Rank-1 only if best alternative beats it by MORE than margin."""
    ar = np.arange(P.shape[0])
    bi = P[:, 1:].argmax(1) + 1
    gap = P[ar, bi] - P[:, 0]
    return np.where(gap > margin, bi, 0)


# ============================================================
# MODELS AND CROSS-VALIDATION
# ============================================================

def rows_of(qidx):
    return (np.asarray(qidx)[:, None] * K + np.arange(K)[None, :]).ravel()


def make_binary(seed):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(objective="binary", random_state=seed, **LGBM_COMMON)


def make_ranker(seed):
    from lightgbm import LGBMRanker
    return LGBMRanker(objective="lambdarank", random_state=seed, **LGBM_COMMON)


def make_folds(df, Q, n_splits, seed):
    from sklearn.model_selection import GroupKFold, KFold

    fam = df["target_commodity_family"].to_numpy()[::K]
    fam_s = pd.Series(fam).astype("object")
    null = fam_s.isna().to_numpy()
    codes = pd.factorize(fam_s.fillna("__nan__"))[0]
    groups = np.where(null, -1 - np.arange(Q), codes)

    n_groups = len(np.unique(groups))
    max_share = pd.Series(groups).value_counts(normalize=True).iloc[0]
    if n_groups >= 3 * n_splits and max_share <= 0.5:
        print(f"[CV] GroupKFold by commodity family ({n_groups} groups).")
        return list(GroupKFold(n_splits=n_splits).split(np.arange(Q), groups=groups))

    print("[CV] Family grouping unsuitable; using shuffled query-level KFold.")
    return list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.arange(Q)))


def choose_gate_margin(X, y, train_q, seed, inner_splits=3):
    """Pick margin using ONLY training queries (inner out-of-fold probabilities)."""
    from sklearn.model_selection import KFold

    P = np.zeros((len(train_q), K))
    kf = KFold(n_splits=inner_splits, shuffle=True, random_state=seed + 1)
    for a, b in kf.split(np.arange(len(train_q))):
        qa, qb = train_q[a], train_q[b]
        m = make_binary(seed)
        m.fit(X[rows_of(qa)], y[rows_of(qa)])
        P[b] = m.predict_proba(X[rows_of(qb)])[:, 1].reshape(-1, K)

    y_mat = y.reshape(-1, K)[train_q]
    ar = np.arange(len(train_q))
    best_m, best_acc = 0.0, -1.0
    for m_ in GATE_GRID:
        acc = y_mat[ar, gated_selection(P, m_)].mean()
        if acc >= best_acc:          # ties -> larger (more conservative) margin
            best_m, best_acc = m_, acc
    return best_m


def run_cv(X, y, folds, Q, seed, tune_gate):
    oof_bin = np.zeros((Q, K))
    oof_rank = np.zeros((Q, K))
    sel_gated = np.zeros(Q, dtype=int)
    margins = []
    gains = []

    for f, (tr, te) in enumerate(folds, 1):
        t0 = time.time()
        tr, te = np.asarray(tr), np.asarray(te)
        Xtr, ytr = X[rows_of(tr)], y[rows_of(tr)]
        Xte = X[rows_of(te)]

        clf = make_binary(seed)
        clf.fit(Xtr, ytr)
        oof_bin[te] = clf.predict_proba(Xte)[:, 1].reshape(-1, K)
        gains.append(clf.booster_.feature_importance(importance_type="gain"))

        rk = make_ranker(seed)
        rk.fit(Xtr, ytr.astype(int), group=[K] * len(tr))
        oof_rank[te] = rk.predict(Xte).reshape(-1, K)

        if tune_gate:
            m_ = choose_gate_margin(X, y, tr, seed)
        else:
            m_ = 0.0
        margins.append(m_)
        sel_gated[te] = gated_selection(oof_bin[te], m_)

        print(f"[CV] fold {f}/{len(folds)}  train={len(tr):,}  test={len(te):,}  "
              f"gate_margin={m_:.3f}  ({time.time() - t0:.1f}s)")

    return oof_bin, oof_rank, sel_gated, margins, np.mean(gains, axis=0)


# ============================================================
# EVALUATION
# ============================================================

def mcnemar_p(b, c):
    if b + c == 0:
        return 1.0
    from scipy.stats import binomtest
    return float(binomtest(min(b, c), b + c, 0.5).pvalue)


def summarize(name, sel, ape, price, actual, mask, boot_idx):
    Q = ape.shape[0]
    ar = np.arange(Q)
    a = ape[ar, sel][mask]
    r1 = ape[:, 0][mask]
    c10, b10 = a <= 10, r1 <= 10

    rec = int((~b10 & c10).sum())
    dam = int((b10 & ~c10).sum())

    acc = c10.mean() * 100
    acc_boot = c10[boot_idx].mean(1) * 100
    d_boot = (c10.astype(float) - b10.astype(float))[boot_idx].mean(1) * 100

    sel_price = price[ar, sel][mask]
    return {
        "policy": name,
        "n_queries": int(mask.sum()),
        "acc5": (a <= 5).mean() * 100,
        "acc10": acc,
        "acc10_ci_lo": np.percentile(acc_boot, 2.5),
        "acc10_ci_hi": np.percentile(acc_boot, 97.5),
        "acc20": (a <= 20).mean() * 100,
        "delta_vs_rank1_pp": acc - b10.mean() * 100,
        "delta_ci_lo": np.percentile(d_boot, 2.5),
        "delta_ci_hi": np.percentile(d_boot, 97.5),
        "recovered": rec,
        "damaged": dam,
        "net": rec - dam,
        "mcnemar_p": mcnemar_p(rec, dam),
        "override_rate_pct": (sel[mask] != 0).mean() * 100,
        "mdape": float(np.median(a)),
        "capped_mean_ape": float(np.minimum(a, 100).mean()),
        "mae": float(np.mean(np.abs(sel_price - actual[mask]))),
    }


def evaluate_all(policies, ape, price, actual, mask, rng, n_boot):
    idx = np.flatnonzero(mask)
    boot_idx = rng.integers(0, len(idx), size=(n_boot, len(idx)))
    rows = [summarize(n, s, ape, price, actual, mask, boot_idx) for n, s in policies.items()]
    return pd.DataFrame(rows)


def print_table(title, table):
    cols = ["policy", "acc10", "acc10_ci_lo", "acc10_ci_hi", "delta_vs_rank1_pp",
            "recovered", "damaged", "net", "mcnemar_p", "override_rate_pct", "mdape"]
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def build_ledger(df, sel, ape, price, actual):
    Q = len(sel)
    ar = np.arange(Q)
    b_ape, s_ape = ape[:, 0], ape[ar, sel]
    return pd.DataFrame({
        "query_index": df["query_index"].to_numpy()[::K],
        "baseline_unit_price": price[:, 0],
        "selected_candidate_rank": sel + 1,
        "selected_unit_price": price[ar, sel],
        "override": sel != 0,
        "actual_unit_price": actual,
        "baseline_ape": b_ape,
        "selected_ape": s_ape,
        "is_baseline_within_10pct": b_ape <= 10,
        "is_selected_within_10pct": s_ape <= 10,
        "recovered": (b_ape > 10) & (s_ape <= 10),
        "damaged": (b_ape <= 10) & (s_ape > 10),
    })


# ============================================================
# MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Experiment 7 - Learned Rank-1-aware reranker")
    p.add_argument("--debug", type=int, default=0, help="Use only the first N queries (0 = full).")
    p.add_argument("--resume", action="store_true", help="Reuse cached semantic arrays.")
    p.add_argument("--embedder", choices=["minilm", "tfidf"], default="minilm")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-gate", action="store_true", help="Skip nested-CV gate tuning.")
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH))
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.time()
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out) if args.out else (
        BASE_OUTPUT_DIR / (f"debug_{args.debug}" if args.debug > 0 else "full")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 7 - LEARNED RANK-1-AWARE HISTORICAL PRICE RERANKER")
    print("=" * 70)
    print(f"[INFO] Output dir: {out_dir}")

    df = load_dataset(Path(args.data), args.debug)
    Q = len(df) // K

    tgt_clean = [clean_text(x) for x in df["target_description"].to_numpy()[::K]]
    cand_clean = [clean_text(x) for x in df["candidate_description"].to_numpy()]

    sem_tc, sem_cc = compute_semantic(
        df, tgt_clean, cand_clean, args.embedder,
        out_dir / "semantic_cache.npz", args.resume, args.seed,
    )

    print("[INFO] Building features...")
    feats, ctx = build_features(df, sem_tc, sem_cc, tgt_clean, cand_clean)
    feature_names = list(feats.columns)
    X = feats.to_numpy(dtype=np.float32)
    print(f"[OK] Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features")

    # ---------------- evaluation-only quantities ----------------
    price = ctx["price"]
    actual = pd.to_numeric(df["actual_unit_price"], errors="coerce").to_numpy(float).reshape(Q, K)[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        ape = np.abs(price - actual[:, None]) / np.maximum(np.abs(actual[:, None]), 1e-9) * 100
    ape = np.where(np.isfinite(ape), ape, np.inf)
    y = (ape <= 10).astype(np.float32).reshape(-1)

    has_price_string = np.array([bool(PRICE_RE.search(str(t))) for t in df["target_description"].to_numpy()[::K]])
    oracle = (ape.min(1) <= 10)
    b10 = ape[:, 0] <= 10
    recoverable = (~b10) & oracle
    print(f"[INFO] Rank-1 Acc@10%: {b10.mean() * 100:.2f}% | Oracle@10: {oracle.mean() * 100:.2f}% | "
          f"recoverable queries: {int(recoverable.sum()):,} | "
          f"target text with price strings: {has_price_string.mean() * 100:.1f}%")

    # ---------------- cross-validated models ----------------
    folds = make_folds(df, Q, args.folds, args.seed)
    oof_bin, oof_rank, sel_gated, margins, gain = run_cv(
        X, y, folds, Q, args.seed, tune_gate=not args.no_gate
    )

    policies = {
        "Rank-1 baseline": np.zeros(Q, dtype=int),
        "Semantic argmax (MiniLM)": ctx["sem"].argmax(1),
        "Exp6 hand-built gate (re-impl.)": exp6_rule_policy(ctx),
        "LGBM binary argmax": oof_bin.argmax(1),
        "LGBM lambdarank argmax": oof_rank.argmax(1),
        "LGBM binary + nested-CV gate [PRIMARY]": sel_gated,
    }

    all_mask = np.ones(Q, dtype=bool)
    res_all = evaluate_all(policies, ape, price, actual, all_mask, rng, args.boot)
    print_table("RESULTS - ALL QUERIES (out-of-fold; CIs are 95% bootstrap)", res_all)

    clean_mask = ~has_price_string
    res_clean = None
    if clean_mask.sum() >= 30:
        res_clean = evaluate_all(policies, ape, price, actual, clean_mask, rng, args.boot)
        print_table(f"RESULTS - CLEAN SUBSET (no price string in target; n={int(clean_mask.sum()):,})", res_clean)

    # picker diagnostics on the recoverable population
    print("\n" + "-" * 100)
    print(f"Recoverable population: {int(recoverable.sum()):,} queries "
          f"(Rank-1 wrong, a correct candidate exists)")
    ar = np.arange(Q)
    for n, s in policies.items():
        got = int((ape[ar, s] <= 10)[recoverable].sum())
        print(f"  {n:<44s} captured {got:>6,} / {int(recoverable.sum()):,}")
    print(f"\nNested-CV gate margins per fold: {[round(m, 3) for m in margins]}")

    # ---------------- save artifacts ----------------
    imp = pd.DataFrame({"feature": feature_names, "gain": gain})
    imp["gain_pct"] = imp["gain"] / max(imp["gain"].sum(), 1e-9) * 100
    imp = imp.sort_values("gain", ascending=False)
    print("\nTop 15 features by gain:")
    print(imp.head(15).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    cand = feats.copy()
    cand.insert(0, "candidate_rank", df["candidate_rank"].to_numpy())
    cand.insert(0, "query_index", df["query_index"].to_numpy())
    cand["candidate_unit_price"] = price.reshape(-1)
    cand["EVAL_ONLY_ape"] = ape.reshape(-1)
    cand["EVAL_ONLY_is_within_10pct"] = y
    cand["oof_binary_prob"] = oof_bin.reshape(-1)
    cand["oof_rank_score"] = oof_rank.reshape(-1)
    cand.to_parquet(out_dir / "exp7_candidate_table.parquet", index=False)

    ledger = build_ledger(df, sel_gated, ape, price, actual)
    ledger.to_csv(out_dir / "exp7_ledger_primary.csv", index=False)
    res_all.to_csv(out_dir / "exp7_results_all.csv", index=False)
    if res_clean is not None:
        res_clean.to_csv(out_dir / "exp7_results_clean_subset.csv", index=False)
    imp.to_csv(out_dir / "exp7_feature_importance.csv", index=False)

    summary = {
        "experiment": "Experiment 7 - Learned Rank-1-aware reranker",
        "queries": int(Q),
        "rank1_acc10": float(b10.mean() * 100),
        "oracle_acc10": float(oracle.mean() * 100),
        "recoverable_queries": int(recoverable.sum()),
        "primary_policy": "LGBM binary + nested-CV gate",
        "gate_margins_per_fold": margins,
        "n_features": len(feature_names),
        "folds": args.folds,
        "embedder": args.embedder,
        "seed": args.seed,
        "lgbm_params": LGBM_COMMON,
        "selection_uses_actual_price_as_feature": False,
        "results_all": res_all.to_dict(orient="records"),
    }
    with open(out_dir / "exp7_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print(f"\n[OK] Artifacts saved to {out_dir}")
    print(f"[DONE] {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()