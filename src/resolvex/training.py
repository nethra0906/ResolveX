"""Train/validation split construction and the supervised training loop.

Validation is entity-aware (grouped by source1_entity_id): every ground-truth row
for a given Source 1 entity stays entirely in train or entirely in validation, so no
positive/negative pair from the same S1 entity leaks across the split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from resolvex.config import RANDOM_SEED, VAL_FRACTION


def grouped_train_val_split(
    s1_ids: list[str], val_fraction: float = VAL_FRACTION, seed: int = RANDOM_SEED
) -> tuple[set[str], set[str]]:
    """Deterministically split Source 1 entity ids into train/val sets."""
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(s1_ids))
    rng.shuffle(ids)
    n_val = int(len(ids) * val_fraction)
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return train_ids, val_ids


def fit_lgbm(model, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
    import lightgbm as lgb

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    return model
