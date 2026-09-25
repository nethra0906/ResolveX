"""Unit tests for output-rule enforcement: subset-of-candidates, valid-id filtering,
one-row-per-required-entity, duplicate-free ID lists."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resolvex.postprocessing import enforce_valid_matches, to_output_frame


def test_drops_non_candidate_matches():
    predictions = {"S1-1": {"S2-1", "S2-2"}}
    candidates = {"S1-1": {"S2-1"}}
    cleaned = enforce_valid_matches(predictions, candidates)
    assert cleaned["S1-1"] == {"S2-1"}


def test_drops_invalid_prefix_and_unknown_ids():
    predictions = {"S1-1": {"S1-2", "S2-9", "S3-9"}}
    candidates = {"S1-1": {"S1-2", "S2-9", "S3-9"}}
    cleaned = enforce_valid_matches(predictions, candidates, valid_ids={"S2-9"})
    assert cleaned["S1-1"] == {"S2-9"}


def test_output_frame_has_one_row_per_required_id_including_missing():
    predictions = {"S1-1": {"S2-1"}}
    required = ["S1-1", "S1-2"]
    df = to_output_frame(predictions, required, "source1_entity_id", "matched_entity_ids")
    assert len(df) == 2
    assert df.set_index("source1_entity_id").loc["S1-2", "matched_entity_ids"] == ""
    assert df["source1_entity_id"].duplicated().sum() == 0


def test_output_frame_ids_sorted_and_deduplicated():
    predictions = {"S1-1": {"S2-2", "S2-1"}}
    df = to_output_frame(predictions, ["S1-1"], "source1_entity_id", "matched_entity_ids")
    assert df.iloc[0]["matched_entity_ids"] == "S2-1,S2-2"
