"""Official-metric evaluation: per-Source-1 precision/recall/F_0.5, macro-averaged.

Implements the exact metric definition from the problem statement: F_0.5 computed
per Source 1 entity (a singleton with a correct empty prediction scores 1.0; any
false merge on a singleton scores 0.0), then macro-averaged across all Source 1
entities in the evaluation set — including singletons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BETA = 0.5
BETA_SQ = BETA * BETA


def f_beta_per_entity(predicted: set[str], truth: set[str]) -> tuple[float, float, float]:
    """Return (precision, recall, f_0.5) for one Source 1 entity.

    Matches the competition's edge cases exactly:
    - truth empty, predicted empty -> (1.0, 1.0, 1.0)
    - truth empty, predicted non-empty -> (0.0, 1.0, 0.0)  [recall is vacuously 1,
      but F_0.5 is defined as 0 here per the problem statement's singleton rule]
    - truth non-empty, predicted empty -> (1.0 if no predictions is correct... ) ->
      precision undefined (no predictions) -> treated as 0 recall, 0 f0.5 with
      precision reported as 0 for the undefined case (standard convention).
    """
    if not truth and not predicted:
        return 1.0, 1.0, 1.0
    if not truth and predicted:
        # Any prediction on a true singleton is a false merge -> F_0.5 = 0.
        return 0.0, 1.0, 0.0
    if truth and not predicted:
        return 0.0, 0.0, 0.0

    tp = len(predicted & truth)
    precision = tp / len(predicted)
    recall = tp / len(truth)
    if precision == 0.0 and recall == 0.0:
        return 0.0, 0.0, 0.0
    f_beta = (1 + BETA_SQ) * precision * recall / (BETA_SQ * precision + recall)
    return precision, recall, f_beta


def macro_f_beta(
    predictions: dict[str, set[str]], truth: dict[str, set[str]], entity_ids: list[str] | None = None
) -> dict:
    """Macro-average precision/recall/F_0.5 over ``entity_ids`` (default: union of both dicts' keys).

    Every entity id must be scored, even if it's missing from ``predictions`` (treated
    as an empty prediction) or from ``truth`` (treated as a true singleton) — this
    matches "every Source 1 entity must appear" from the competition rules.
    """
    if entity_ids is None:
        entity_ids = sorted(set(predictions) | set(truth))

    precisions, recalls, f_betas = [], [], []
    for eid in entity_ids:
        pred = predictions.get(eid, set())
        true = truth.get(eid, set())
        p, r, f = f_beta_per_entity(pred, true)
        precisions.append(p)
        recalls.append(r)
        f_betas.append(f)

    return {
        "n_entities": len(entity_ids),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f0.5": float(np.mean(f_betas)),
    }


def predictions_from_scored_pairs(
    scored: pd.DataFrame,
    threshold: float,
    s1_col: str = "source1_entity_id",
    cand_col: str = "candidate_entity_id",
    score_col: str = "score",
) -> dict[str, set[str]]:
    """Convert a scored-pairs table into {source1_id: {predicted matches}} at a fixed threshold."""
    kept = scored[scored[score_col] >= threshold]
    return kept.groupby(s1_col)[cand_col].apply(set).to_dict()
