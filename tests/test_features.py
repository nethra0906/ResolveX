"""Robustness tests for pairwise feature computation: missing values, unicode,
identical records, completely different records, very long addresses."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resolvex.features import FEATURE_COLUMNS, compute_features


def _df(rows):
    return pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])


def test_identical_records_score_near_max_on_key_features():
    s1 = _df([("S1-1", "Acme Corp", "1 Main St, Springfield", "US")])
    other = _df([("S2-1", "Acme Corp", "1 Main St, Springfield", "US")])
    pairs = pd.DataFrame({"source1_entity_id": ["S1-1"], "candidate_entity_id": ["S2-1"]})
    feats = compute_features(pairs, s1, other)
    assert feats.loc[0, "name_exact_raw"] == 1.0
    assert feats.loc[0, "addr_exact_norm"] == 1.0
    assert feats.loc[0, "country_exact"] == 1.0
    assert set(FEATURE_COLUMNS) == set(feats.columns)
    assert not feats.isna().any().any()


def test_completely_different_records_score_low():
    s1 = _df([("S1-1", "Acme Corp", "1 Main St, Springfield", "US")])
    other = _df([("S2-1", "Zephyr Consulting", "99 Distant Rd, Faraway", "India")])
    pairs = pd.DataFrame({"source1_entity_id": ["S1-1"], "candidate_entity_id": ["S2-1"]})
    feats = compute_features(pairs, s1, other)
    assert feats.loc[0, "name_exact_raw"] == 0.0
    assert feats.loc[0, "country_exact"] == 0.0
    assert feats.loc[0, "name_jaccard"] == 0.0


def test_missing_address_does_not_crash():
    s1 = _df([("S1-1", "Acme Corp", "", "US")])
    other = _df([("S2-1", "Acme Corp", "", "US")])
    pairs = pd.DataFrame({"source1_entity_id": ["S1-1"], "candidate_entity_id": ["S2-1"]})
    feats = compute_features(pairs, s1, other)
    assert not feats.isna().any().any()
    assert np.isfinite(feats.to_numpy()).all()


def test_unicode_names_do_not_crash():
    s1 = _df([("S1-1", "राम मार्केटिंग", "Delhi", "India")])
    other = _df([("S2-1", "राम मार्केटिंग प्राइवेट लिमिटेड", "New Delhi", "India")])
    pairs = pd.DataFrame({"source1_entity_id": ["S1-1"], "candidate_entity_id": ["S2-1"]})
    feats = compute_features(pairs, s1, other)
    assert not feats.isna().any().any()
    assert feats.loc[0, "name_jaccard"] > 0


def test_multiple_pairs_against_shared_other_df():
    s1 = _df([("S1-1", "Acme Corp", "1 Main St", "US"), ("S1-2", "Other Biz", "2 Second St", "US")])
    other = _df([("S2-1", "Acme Corporation", "1 Main Street", "US")])
    pairs = pd.DataFrame(
        {"source1_entity_id": ["S1-1", "S1-2"], "candidate_entity_id": ["S2-1", "S2-1"]}
    )
    feats = compute_features(pairs, s1, other)
    assert len(feats) == 2
    assert feats.loc[0, "name_levenshtein"] > feats.loc[1, "name_levenshtein"]
