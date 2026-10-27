import os
import subprocess
import sys

import numpy as np

from mtah import frames
from mtah.submission import prediction_summary, validation_error, write_submission

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
(H1_START, _), (H2_START, _) = frames.graded_runs()


def _labels(*spans, fill="D"):
    """Full-length labels: X outside the graded halves, `fill` inside, then the given spans."""
    labels = np.full(frames.N_FRAMES, "X", dtype="<U1")
    labels[frames.graded_mask()] = fill
    for (start, stop), label in spans:
        labels[start:stop] = label
    return labels


def test_written_file_passes_the_official_checks(tmp_path):
    path = write_submission(str(tmp_path / "team.csv"), _labels(((H1_START, H1_START + 500), "H")))
    assert validation_error(path) is None
    done = subprocess.run([sys.executable, "evaluation/validate_submission.py", path],
                          cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0 and done.stdout.startswith("OK")


def test_wrong_length_is_rejected(tmp_path):
    path = write_submission(str(tmp_path / "short.csv"), np.array(["D"] * 10))
    assert "row count" in validation_error(path)


def test_dead_ball_splits_a_possession():
    labels = _labels(((H1_START, H1_START + 100), "H"), ((H1_START + 150, H1_START + 250), "H"))
    assert prediction_summary(labels)["n_possessions"] == 2  # H H D D H H is two


def test_abstention_inside_a_possession_is_bridged_like_the_scorer():
    labels = _labels(((H2_START, H2_START + 300), "A"), ((H2_START + 100, H2_START + 120), "X"))
    summary = prediction_summary(labels)
    assert summary["n_possessions"] == 1
    assert summary["x_on_graded"] == 20
