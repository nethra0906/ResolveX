"""Pairwise feature engineering for (Source 1, candidate) record pairs.

Given a candidate-pairs table and the two source record tables it references, builds
a numeric feature matrix for the matching model. All features are computed with
vectorized pandas/numpy operations or ``rapidfuzz.process`` batch APIs (implemented
in C) since candidate tables can run into the tens of millions of rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from resolvex import normalization as norm

FEATURE_COLUMNS = [
    "name_exact_raw",
    "name_exact_norm",
    "name_levenshtein",
    "name_jaro_winkler",
    "name_token_sort",
    "name_token_set",
    "name_jaccard",
    "name_char_ngram_jaccard",
    "name_tfidf_cosine",
    "name_len_ratio",
    "name_len_diff",
    "name_token_count_diff",
    "name_shared_rare_token",
    "addr_exact_norm",
    "addr_levenshtein",
    "addr_token_sort",
    "addr_jaccard",
    "addr_tfidf_cosine",
    "addr_shared_digit_token",
    "addr_shared_rare_token",
    "addr_len_ratio",
    "addr_len_diff",
    "country_exact",
    "name_x_addr_jaccard",
    "country_x_name_jaccard",
    "country_x_addr_jaccard",
]


def _prep_side(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["entity_id", "business_name", "business_address", "country"]].copy()
    out["name_norm"] = norm.name_normalized_string(df["business_name"])
    out["addr_norm"] = norm.address_normalized_string(df["business_address"])
    out["name_tokens"] = out["name_norm"].str.split().apply(lambda x: x or [])
    out["addr_tokens"] = out["addr_norm"].str.split().apply(lambda x: x or [])
    out["name_ngrams"] = norm.char_ngrams(out["name_norm"], 3)
    out["digit_tokens"] = norm.extract_digit_tokens(df["business_address"]).apply(set)
    out["country_norm"] = norm.country_normalized(df["country"])
    return out.set_index("entity_id")


def _jaccard(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _set_jaccard(sa: set, sb: set) -> float:
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _token_rarity_weights(token_lists: pd.Series) -> dict[str, float]:
    """Inverse-document-frequency-style weight per token across a token-list series."""
    from collections import Counter

    counts: Counter[str] = Counter()
    for toks in token_lists:
        counts.update(set(toks))
    n = len(token_lists)
    return {tok: np.log(n / c) for tok, c in counts.items()}


def _shared_rare_token(tokens_a: list, tokens_b: list, weights: dict[str, float]) -> float:
    shared = set(tokens_a) & set(tokens_b)
    if not shared:
        return 0.0
    return max(weights.get(t, 0.0) for t in shared)


def _build_tfidf_cosine(
    left_texts: pd.Series, right_texts: pd.Series, corpus: pd.Series
) -> np.ndarray:
    """Row-wise cosine similarity between paired left/right texts, fit on ``corpus``.

    ``min_df=1`` (rather than 2+) is deliberate: a small candidate batch (e.g. one
    S1 entity with a single low-country-volume candidate) can have so few unique
    normalized strings that a higher min_df prunes the vocabulary to nothing,
    which previously crashed inference on exactly the small-batch case that's
    common once candidates are split and scored per source.
    """
    non_empty_corpus = corpus[corpus.str.len() > 0]
    if non_empty_corpus.nunique() < 1:
        return np.zeros(len(left_texts), dtype=np.float64)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=1, max_features=200_000)
    vectorizer.fit(non_empty_corpus)
    left_mat = vectorizer.transform(left_texts)
    right_mat = vectorizer.transform(right_texts)
    # row-wise dot product of two L2-normalized sparse matrices = cosine similarity
    left_mat = left_mat.multiply(1.0 / (np.sqrt(left_mat.multiply(left_mat).sum(axis=1)) + 1e-9))
    right_mat = right_mat.multiply(1.0 / (np.sqrt(right_mat.multiply(right_mat).sum(axis=1)) + 1e-9))
    left_mat = csr_matrix(left_mat)
    right_mat = csr_matrix(right_mat)
    dots = left_mat.multiply(right_mat).sum(axis=1)
    return np.asarray(dots).ravel()


def compute_features(
    pairs: pd.DataFrame,
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the full feature matrix for ``pairs`` (source1_entity_id, candidate_entity_id, ...).

    ``other_df`` must contain every candidate_entity_id referenced in ``pairs``
    (typically the concatenation of Source 2 and Source 3 records relevant to the
    slice being scored). Both ``s1_df`` and ``other_df`` are filtered down to the
    ids ``pairs`` actually references before normalization — at inference time
    ``other_df`` is commonly a full 5M-row source table while ``pairs`` only
    touches a small fraction of it, so normalizing the full table on every call
    would waste the majority of the work.
    """
    needed_s1 = set(pairs["source1_entity_id"])
    needed_other = set(pairs["candidate_entity_id"])
    s1_idx = _prep_side(s1_df[s1_df["entity_id"].isin(needed_s1)])
    other_idx = _prep_side(other_df[other_df["entity_id"].isin(needed_other)])

    left = s1_idx.loc[pairs["source1_entity_id"]].reset_index(drop=True)
    right = other_idx.loc[pairs["candidate_entity_id"]].reset_index(drop=True)

    feat = pd.DataFrame(index=pairs.index)

    feat["name_exact_raw"] = (left["business_name"].values == right["business_name"].values).astype(np.float32)
    feat["name_exact_norm"] = (left["name_norm"].values == right["name_norm"].values).astype(np.float32)

    name_pairs = list(zip(left["name_norm"], right["name_norm"]))
    feat["name_levenshtein"] = [fuzz.ratio(a, b) / 100.0 for a, b in name_pairs]
    feat["name_jaro_winkler"] = [fuzz.WRatio(a, b) / 100.0 for a, b in name_pairs]
    feat["name_token_sort"] = [fuzz.token_sort_ratio(a, b) / 100.0 for a, b in name_pairs]
    feat["name_token_set"] = [fuzz.token_set_ratio(a, b) / 100.0 for a, b in name_pairs]

    feat["name_jaccard"] = [
        _jaccard(a, b) for a, b in zip(left["name_tokens"], right["name_tokens"])
    ]
    feat["name_char_ngram_jaccard"] = [
        _set_jaccard(a, b) for a, b in zip(left["name_ngrams"], right["name_ngrams"])
    ]

    name_len_l = left["name_norm"].str.len().replace(0, 1)
    name_len_r = right["name_norm"].str.len().replace(0, 1)
    feat["name_len_ratio"] = (np.minimum(name_len_l, name_len_r) / np.maximum(name_len_l, name_len_r)).astype(np.float32)
    feat["name_len_diff"] = (name_len_l - name_len_r).abs().astype(np.float32)
    feat["name_token_count_diff"] = (
        left["name_tokens"].apply(len) - right["name_tokens"].apply(len)
    ).abs().astype(np.float32)

    name_weights = _token_rarity_weights(pd.concat([left["name_tokens"], right["name_tokens"]]))
    feat["name_shared_rare_token"] = [
        _shared_rare_token(a, b, name_weights) for a, b in zip(left["name_tokens"], right["name_tokens"])
    ]

    feat["addr_exact_norm"] = (left["addr_norm"].values == right["addr_norm"].values).astype(np.float32)
    addr_pairs = list(zip(left["addr_norm"], right["addr_norm"]))
    feat["addr_levenshtein"] = [fuzz.ratio(a, b) / 100.0 for a, b in addr_pairs]
    feat["addr_token_sort"] = [fuzz.token_sort_ratio(a, b) / 100.0 for a, b in addr_pairs]
    feat["addr_jaccard"] = [
        _jaccard(a, b) for a, b in zip(left["addr_tokens"], right["addr_tokens"])
    ]

    feat["addr_shared_digit_token"] = [
        1.0 if (a & b) else 0.0 for a, b in zip(left["digit_tokens"], right["digit_tokens"])
    ]
    addr_weights = _token_rarity_weights(pd.concat([left["addr_tokens"], right["addr_tokens"]]))
    feat["addr_shared_rare_token"] = [
        _shared_rare_token(a, b, addr_weights) for a, b in zip(left["addr_tokens"], right["addr_tokens"])
    ]

    addr_len_l = left["addr_norm"].str.len().replace(0, 1)
    addr_len_r = right["addr_norm"].str.len().replace(0, 1)
    feat["addr_len_ratio"] = (np.minimum(addr_len_l, addr_len_r) / np.maximum(addr_len_l, addr_len_r)).astype(np.float32)
    feat["addr_len_diff"] = (addr_len_l - addr_len_r).abs().astype(np.float32)

    feat["country_exact"] = (left["country_norm"].values == right["country_norm"].values).astype(np.float32)

    # TF-IDF cosine, fit on the union of names/addresses present in this call's pairs
    # (keeps the vocabulary relevant to the slice being scored rather than a fixed
    # global vocabulary that would need retraining whenever the corpus shifts).
    name_corpus = pd.concat([left["name_norm"], right["name_norm"]]).drop_duplicates()
    feat["name_tfidf_cosine"] = _build_tfidf_cosine(left["name_norm"], right["name_norm"], name_corpus)
    addr_corpus = pd.concat([left["addr_norm"], right["addr_norm"]]).drop_duplicates()
    feat["addr_tfidf_cosine"] = _build_tfidf_cosine(left["addr_norm"], right["addr_norm"], addr_corpus)

    feat["name_x_addr_jaccard"] = feat["name_jaccard"] * feat["addr_jaccard"]
    feat["country_x_name_jaccard"] = feat["country_exact"] * feat["name_jaccard"]
    feat["country_x_addr_jaccard"] = feat["country_exact"] * feat["addr_jaccard"]

    feat = feat[FEATURE_COLUMNS].astype(np.float32)
    feat.index = pairs.index
    return feat


# Default chunk size for compute_features_batched. Each row of the intermediate
# left/right working tables carries several Python object columns (token lists,
# n-gram sets) whose per-row overhead is much larger than the final float32
# feature row — at full competition scale (up to ~100M candidate pairs) building
# those intermediates for everything at once can need tens of GB of RAM. Chunking
# bounds peak memory to roughly this many rows' worth of intermediates at a time,
# keeping only the compact float32 results across chunks.
DEFAULT_FEATURE_BATCH_SIZE = 200_000


def compute_features_batched(
    pairs: pd.DataFrame,
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    batch_size: int = DEFAULT_FEATURE_BATCH_SIZE,
) -> pd.DataFrame:
    """Memory-bounded version of ``compute_features``: processes ``pairs`` in chunks
    of ``batch_size`` rows and concatenates only the resulting float32 feature rows.

    Use this instead of ``compute_features`` directly whenever ``pairs`` can be
    large (candidate tables at full dataset scale routinely run into the tens of
    millions of rows) — see the module-level note on ``DEFAULT_FEATURE_BATCH_SIZE``
    for why the unbatched version can exhaust memory well before that scale.
    """
    if len(pairs) <= batch_size:
        return compute_features(pairs, s1_df, other_df)

    chunks = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs.iloc[start : start + batch_size]
        chunks.append(compute_features(chunk, s1_df, other_df))
    return pd.concat(chunks)
