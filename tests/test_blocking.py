"""Small end-to-end blocking tests on synthetic data — correctness, not scale.

Covers: recall on an obvious match, country partitioning (no cross-country
candidates), robustness to empty/missing fields, and the max-candidates cap.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resolvex.blocking import generate_candidates


def _df(rows):
    return pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])


def test_obvious_match_is_found():
    s1 = _df([("S1-1", "Acme Pvt Ltd", "105 Elm St, Springfield", "US")])
    s2 = _df([("S2-1", "Acme Private Limited", "105 Elm Street, Springfield", "US")])
    s3 = _df([("S3-1", "Totally Different Co", "999 Nowhere Ave", "India")])
    cands = generate_candidates(s1, s2, s3)
    got = set(cands[cands["source1_entity_id"] == "S1-1"]["candidate_entity_id"])
    assert "S2-1" in got
    assert "S3-1" not in got


def test_country_partitioning_blocks_cross_country_pairs():
    # Same name/address tokens, different country -> should not match (country
    # partitioning applies to every key type).
    s1 = _df([("S1-1", "Global Traders Inc", "12 Market Road", "US")])
    s2 = _df([("S2-1", "Global Traders Inc", "12 Market Road", "India")])
    cands = generate_candidates(s1, s2, _df([]))
    got = set(cands[cands["source1_entity_id"] == "S1-1"]["candidate_entity_id"])
    assert "S2-1" not in got


def test_empty_fields_do_not_crash():
    s1 = _df([("S1-1", "", "", "US"), ("S1-2", "Normal Co", "1 Main St", "US")])
    s2 = _df([("S2-1", "", "", "US"), ("S2-2", "Normal Co", "1 Main Street", "US")])
    cands = generate_candidates(s1, s2, _df([]))
    got = set(cands[cands["source1_entity_id"] == "S1-2"]["candidate_entity_id"])
    assert "S2-2" in got


def test_max_candidates_cap_respected():
    s1 = _df([("S1-1", "Common Name Inc", "1 Main St", "US")])
    s2_rows = [(f"S2-{i}", "Common Name Inc", "1 Main St", "US") for i in range(100)]
    s2 = _df(s2_rows)
    cands = generate_candidates(s1, s2, _df([]), max_candidates=10)
    got = cands[cands["source1_entity_id"] == "S1-1"]
    assert len(got) <= 10
