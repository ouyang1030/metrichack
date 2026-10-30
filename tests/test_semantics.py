"""Metrica's possession rule on synthetic label sequences (harness/semantics.py)."""

import numpy as np

from harness import semantics
from evaluation import possession_metrics as pm

H, A, D = pm.H, pm.A, pm.D


def _labels(*runs):
    return np.concatenate([np.full(n, code, dtype=np.int8) for code, n in runs])


def _metrica(labels, halves=None):
    return semantics.metrica(labels, np.array(halves or [[0, len(labels)]]), 1.0)


def test_a_touch_without_control_does_not_switch():
    labels = _labels((H, 100), (A, 10), (H, 100))
    np.testing.assert_array_equal(_metrica(labels), np.full(210, H))


def test_a_spell_as_long_as_the_control_time_switches():
    labels = _labels((H, 100), (A, 25), (H, 100))
    np.testing.assert_array_equal(_metrica(labels), labels)


def test_a_ball_touched_back_and_forth_stays_with_the_team_that_had_it():
    labels = _labels((H, 100), (A, 10), (H, 5), (A, 10), (H, 100))
    np.testing.assert_array_equal(_metrica(labels), np.full(225, H))


def test_spells_next_to_a_stoppage_are_kept():
    labels = _labels((H, 100), (A, 10), (D, 50), (A, 5), (H, 100))
    np.testing.assert_array_equal(_metrica(labels), labels)


def test_halves_are_relabelled_separately():
    labels = _labels((H, 100), (A, 10), (H, 100))
    np.testing.assert_array_equal(_metrica(labels, [[0, 105], [105, 210]]), labels)
