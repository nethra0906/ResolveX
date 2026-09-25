"""Model definitions: baselines and the supervised LightGBM matcher.

LightGBM (MIT license, github.com/microsoft/LightGBM) is used as the primary
supervised model — a classical gradient-boosted tree ensemble, far under the
competition's 8B-parameter ceiling, and a strong fit for a dense tabular pairwise
feature matrix at multi-million-row scale. Logistic regression is kept as an
interpretable baseline for comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from resolvex.config import LGBM_PARAMS


class RuleBasedBaseline:
    """Weighted fuzzy-score baseline: no training, just a hand-set linear combination.

    Used as Phase-9 baseline #4/#5 (weighted fuzzy similarity / rule-based scoring).
    """

    def __init__(self, name_weight: float = 0.5, addr_weight: float = 0.3, country_weight: float = 0.2):
        self.name_weight = name_weight
        self.addr_weight = addr_weight
        self.country_weight = country_weight

    def score(self, features: pd.DataFrame) -> np.ndarray:
        name_sim = features[["name_levenshtein", "name_jaro_winkler", "name_token_set", "name_jaccard"]].mean(axis=1)
        addr_sim = features[["addr_levenshtein", "addr_token_sort", "addr_jaccard"]].mean(axis=1)
        country_sim = features["country_exact"]
        return (
            self.name_weight * name_sim + self.addr_weight * addr_sim + self.country_weight * country_sim
        ).to_numpy()


class ExactNameBaseline:
    """Phase-9 baseline #1/#2: exact normalized name (+ optional country) match."""

    def __init__(self, require_country: bool = False):
        self.require_country = require_country

    def score(self, features: pd.DataFrame) -> np.ndarray:
        s = features["name_exact_norm"].to_numpy(dtype=np.float64)
        if self.require_country:
            s = s * features["country_exact"].to_numpy(dtype=np.float64)
        return s


class TfidfRetrievalBaseline:
    """Phase-9 baseline #4: name TF-IDF cosine similarity alone."""

    def score(self, features: pd.DataFrame) -> np.ndarray:
        return features["name_tfidf_cosine"].to_numpy(dtype=np.float64)


def make_lgbm_model():
    import lightgbm as lgb

    return lgb.LGBMClassifier(**LGBM_PARAMS)


def make_logreg_model():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=1000, class_weight="balanced")
