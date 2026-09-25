"""Final output assembly: enforce every competition output rule before writing TSVs.

Rules enforced here (see resources/README.md / problem statement):
- exactly one row per required Source 1 entity (missing ones get an empty prediction)
- no duplicate source1_entity_id rows
- no duplicate IDs within an ID list
- only S2-/S3- IDs, and only ones that exist in the given valid-id set
- final matches are a subset of the candidates offered for that entity
"""

from __future__ import annotations

import pandas as pd


def enforce_valid_matches(
    predictions: dict[str, set[str]],
    candidates: dict[str, set[str]],
    valid_ids: set[str] | None = None,
) -> dict[str, set[str]]:
    """Drop any predicted id that isn't S2-/S3-, isn't in ``valid_ids`` (if given), or
    isn't among that entity's candidates (matches must be a subset of candidates).
    """
    cleaned: dict[str, set[str]] = {}
    for s1_id, ids in predictions.items():
        allowed = candidates.get(s1_id, set())
        kept = {
            i
            for i in ids
            if i.startswith(("S2-", "S3-")) and i in allowed and (valid_ids is None or i in valid_ids)
        }
        cleaned[s1_id] = kept
    return cleaned


def to_output_frame(
    predictions: dict[str, set[str]], required_ids: list[str], id_col: str, list_col: str
) -> pd.DataFrame:
    """Build the two-column output DataFrame with exactly one row per required id."""
    rows = []
    for eid in required_ids:
        ids = predictions.get(eid, set())
        rows.append((eid, ",".join(sorted(ids))))
    return pd.DataFrame(rows, columns=[id_col, list_col])


def write_tsv(df: pd.DataFrame, path) -> None:
    df.to_csv(path, sep="\t", index=False)
