"""Loading competition TSV files into memory-efficient pandas DataFrames.

All source files are read with an explicit ``sep="\\t"`` (per the problem statement,
addresses and ID lists contain commas) and a fixed dtype map so millions of rows of
short strings don't balloon into Python-object overhead any more than necessary.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from resolvex.config import GROUND_TRUTH_COLUMNS, SOURCE_COLUMNS, SOURCE_DTYPES


def load_source(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Load one *_source{1,2,3}.tsv file.

    Missing name/address values become empty strings (not NaN) so downstream
    normalization code never has to special-case NaN.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=SOURCE_DTYPES,
        na_filter=False,
        nrows=nrows,
    )
    missing = set(SOURCE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    return df[SOURCE_COLUMNS]


def load_ground_truth(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Load train_ground_truth.tsv as (source1_entity_id, matched_entity_ids)."""
    df = pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        na_filter=False,
        nrows=nrows,
    )
    missing = set(GROUND_TRUTH_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    return df[GROUND_TRUTH_COLUMNS]


def ground_truth_to_sets(gt: pd.DataFrame) -> dict[str, frozenset[str]]:
    """Turn the ground-truth DataFrame into {source1_entity_id: frozenset(matched_ids)}."""
    out: dict[str, frozenset[str]] = {}
    for s1, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        out[s1] = frozenset(ids.split(",")) if ids else frozenset()
    return out
