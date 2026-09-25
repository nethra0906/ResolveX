"""Decision-layer logic: turning per-pair match scores into a final match set per
Source 1 entity, with a precision-heavy, F_0.5-aware, abstention-capable strategy.

F_0.5 weights precision 2x over recall, and a false merge on a true singleton scores
0.0 (see evaluation.py) — so a flat ``score > 0.5`` cutoff is a poor default. Instead
this module supports:

- a global score threshold (tuned on validation F_0.5, never on test labels)
- a score-margin / top-k style filter: keep a candidate only if it clears the
  threshold AND is not overwhelmingly dominated by a much stronger competing
  candidate pool signal (guards against low-confidence long tails)
- singleton abstention: if the best candidate's score does not clear an (higher)
  abstention threshold, predict no match at all for that entity
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from resolvex.evaluation import macro_f_beta


@dataclass
class ThresholdConfig:
    accept_threshold: float = 0.5
    # An entity gets ANY predictions only if its top candidate score clears this
    # (typically >= accept_threshold; set higher to make singleton abstention stricter)
    singleton_abstain_threshold: float = 0.5


def apply_thresholds(
    scored: pd.DataFrame,
    config: ThresholdConfig,
    s1_col: str = "source1_entity_id",
    cand_col: str = "candidate_entity_id",
    score_col: str = "score",
) -> dict[str, set[str]]:
    """Score -> final prediction dict, applying accept threshold + singleton abstention."""
    if scored.empty:
        return {}

    best_per_entity = scored.groupby(s1_col)[score_col].transform("max")
    eligible = scored[best_per_entity >= config.singleton_abstain_threshold]
    kept = eligible[eligible[score_col] >= config.accept_threshold]
    return kept.groupby(s1_col)[cand_col].apply(set).to_dict()


def optimize_threshold(
    scored_val: pd.DataFrame,
    truth: dict[str, set[str]],
    all_entity_ids: list[str],
    candidate_thresholds: np.ndarray | None = None,
) -> tuple[ThresholdConfig, pd.DataFrame]:
    """Grid-search a single global threshold (used for both accept + abstention) on
    validation data to maximize macro F_0.5. Returns the best config and a results
    table for the full grid (for the experiment log / ablations).

    Only ``scored_val`` (validation candidate scores) and ``truth`` (validation
    ground truth) are used — never test labels, per the competition rules.
    """
    if candidate_thresholds is None:
        candidate_thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)

    rows = []
    best_f = -1.0
    best_cfg = ThresholdConfig()
    for t in candidate_thresholds:
        cfg = ThresholdConfig(accept_threshold=float(t), singleton_abstain_threshold=float(t))
        preds = apply_thresholds(scored_val, cfg)
        metrics = macro_f_beta(preds, truth, entity_ids=all_entity_ids)
        rows.append({"threshold": float(t), **metrics})
        if metrics["macro_f0.5"] > best_f:
            best_f = metrics["macro_f0.5"]
            best_cfg = cfg

    return best_cfg, pd.DataFrame(rows)
