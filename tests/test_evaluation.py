"""Unit tests for the official F_0.5 metric implementation, including the
competition's explicit singleton edge cases."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resolvex.evaluation import f_beta_per_entity, macro_f_beta


def test_true_singleton_correctly_predicted_scores_one():
    p, r, f = f_beta_per_entity(set(), set())
    assert (p, r, f) == (1.0, 1.0, 1.0)


def test_true_singleton_false_merge_scores_zero():
    p, r, f = f_beta_per_entity({"S2-1"}, set())
    assert f == 0.0


def test_missed_all_matches_scores_zero():
    p, r, f = f_beta_per_entity(set(), {"S2-1", "S2-2"})
    assert f == 0.0


def test_worked_example_from_problem_statement():
    predicted = {"S2-00047", "S2-00193", "S3-00812"}
    truth = {"S2-00047", "S3-00812"}
    p, r, f = f_beta_per_entity(predicted, truth)
    assert math.isclose(p, 2 / 3, abs_tol=1e-6)
    assert r == 1.0
    assert math.isclose(f, 0.714, abs_tol=1e-3)


def test_macro_average_includes_missing_entities_as_empty_predictions():
    predictions = {"S1-1": {"S2-1"}}
    truth = {"S1-1": {"S2-1"}, "S1-2": set()}
    metrics = macro_f_beta(predictions, truth, entity_ids=["S1-1", "S1-2"])
    # S1-1 perfect (f=1.0), S1-2 correctly predicted empty (f=1.0) -> macro 1.0
    assert metrics["macro_f0.5"] == 1.0
    assert metrics["n_entities"] == 2
