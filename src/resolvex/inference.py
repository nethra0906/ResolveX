"""Batch inference: score a candidate-pairs table with a trained model.

Splits candidates by S2/S3 source before feature computation because
``features.compute_features`` needs a single ``other_df`` table to index into.
Uses the batched feature computation so full-test-set scoring (potentially tens of
millions of candidate rows) doesn't exhaust memory building feature intermediates
all at once.
"""

from __future__ import annotations

import pandas as pd

from resolvex.features import compute_features_batched


def score_candidates(
    candidates: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    model,
) -> pd.DataFrame:
    """Return ``candidates`` with an added ``score`` column (model's positive-class probability)."""
    is_s2 = candidates["candidate_entity_id"].str.startswith("S2-")
    parts = []
    for mask, other_df in ((is_s2, s2_df), (~is_s2, s3_df)):
        subset = candidates[mask]
        if subset.empty:
            continue
        feats = compute_features_batched(subset, s1_df, other_df)
        scores = model.predict_proba(feats)[:, 1]
        scored = subset.copy()
        scored["score"] = scores
        parts.append(scored)
    if not parts:
        return candidates.assign(score=pd.Series(dtype=float))
    return pd.concat(parts).sort_index()
