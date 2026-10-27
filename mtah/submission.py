"""Writing, validating and summarising a submission file. Validation is the official code."""

import os

import numpy as np

from mtah import config, frames  # noqa: F401  (config puts the repository root on sys.path)
from evaluation import possession_metrics as pm


def write_submission(path, labels):
    """frame,label CSV, one row per video frame, LF line endings -- the starter's exact format."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    return path


def validation_error(path, n_rows=frames.N_FRAMES):
    """None if the official scorer would accept the file, else its one-line reason."""
    return pm.validate_submission(pm.read_label_csv(path), n_rows)


def prediction_summary(labels):
    """What our file claims over the graded halves, the way the scorer will read it.

    Abstentions are filled exactly as the scorer fills them before counting possessions.
    """
    codes = pm.label_codes(np.asarray(labels))
    runs = frames.graded_runs()
    spells = pm.possession_spells(pm.fill_abstentions(codes, runs), runs)
    seconds = np.array([stop - start for start, stop, _ in spells], dtype=float) / frames.FPS
    graded = frames.graded_mask(len(codes))
    return {
        "n_possessions": len(spells),
        "duration_p10_s": float(np.percentile(seconds, 10)),
        "duration_p50_s": float(np.percentile(seconds, 50)),
        "duration_p90_s": float(np.percentile(seconds, 90)),
        "short_share": float(np.mean(seconds < 1.0)),
        "dead_share": float(np.mean(codes[graded] == pm.D)),
        "x_on_graded": int(np.sum(codes[graded] == pm.X)),
    }
