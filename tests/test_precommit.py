import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts", "hooks"))
from check_staged import FRAME_ROWS, MAX_BYTES, blocked_reason  # noqa: E402


@pytest.mark.parametrize("path, size, n_lines", [
    ("data/tactical/BAR-ATH_tactical_atd.txt", 10, 0),
    ("harness/cache/x.json", 10, 0),
    ("debug/overlay.png", 10, 0),
    ("clip.mp4", 10, 0),
    ("models/features.npz", 10, 0),
    ("evaluation/results/TEAM_NAME.csv", 10, 3),
    ("evaluation/reports/baseline_raw_transitions.csv", 10, 3),
    ("models/weights.bin", MAX_BYTES + 1, 0),
    ("notes/predictions.csv", 10, FRAME_ROWS + 1),
])
def test_blocked(path, size, n_lines):
    assert blocked_reason(path, size, n_lines)


@pytest.mark.parametrize("path, size, n_lines", [
    ("evaluation/results/baseline_smooth50.csv", 1682773, FRAME_ROWS + 1),
    ("evaluation/gt/BAR-ATH_possession_gt_public.csv", 1682773, FRAME_ROWS + 1),
    ("mtah/cache.py", 10000, 0),
    ("harness/labels_summary.csv", 1000, 20),
])
def test_allowed(path, size, n_lines):
    assert blocked_reason(path, size, n_lines) is None
