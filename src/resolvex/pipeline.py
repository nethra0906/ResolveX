"""End-to-end orchestration: raw data -> blocking -> features -> model -> thresholds
-> singleton/multi-match decision -> postprocessing -> output files.

This module wires the other resolvex modules together; it's called from
notebooks/01_train_and_infer_kaggle.ipynb's test-inference section. Every step
from raw data to the final output files is reproducible by running the
notebooks in notebooks/ — see the repo README for which ones run locally vs. on
Kaggle.
"""

from __future__ import annotations

import pandas as pd

from resolvex.blocking import generate_candidates
from resolvex.inference import score_candidates
from resolvex.postprocessing import enforce_valid_matches, to_output_frame
from resolvex.thresholding import ThresholdConfig, apply_thresholds


def run_full_pipeline(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    model,
    threshold_config: ThresholdConfig,
) -> dict:
    """Run blocking -> scoring -> thresholding -> postprocessing for one split.

    Returns a dict with the candidate table, scored table, and the two output
    DataFrames (matching_results, candidate_pairs) ready to write to TSV.
    """
    required_ids = s1_df["entity_id"].tolist()
    valid_ids = set(s2_df["entity_id"]) | set(s3_df["entity_id"])

    candidates = generate_candidates(s1_df, s2_df, s3_df)
    candidate_map = (
        candidates.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict()
        if not candidates.empty
        else {}
    )

    scored = score_candidates(candidates, s1_df, s2_df, s3_df, model) if not candidates.empty else candidates.assign(score=[])

    predictions = apply_thresholds(scored, threshold_config)
    predictions = enforce_valid_matches(predictions, candidate_map, valid_ids)

    matching_df = to_output_frame(predictions, required_ids, "source1_entity_id", "matched_entity_ids")
    candidate_df = to_output_frame(candidate_map, required_ids, "source1_entity_id", "candidate_entity_ids")

    return {
        "candidates": candidates,
        "scored": scored,
        "matching_results": matching_df,
        "candidate_pairs": candidate_df,
    }
