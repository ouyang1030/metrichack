"""Calibrate the simulated phase tagger of harness/fuse.py to Smart Tagging, behind the firewall.

    python scripts/calibrate_tagger.py      # after harness.train_v2 (its out-of-fold outputs)

Measures Smart Tagging's vote statistics (harness/tagger.py) on BAR-ATH's graded frames after
each kickoff, the agreement against the pipeline with Smart Tagging switched off, then fits the
simulated tagger's knobs on the DFL matches until it shows the same. All go to cache/aggregates/
(tagging_stats.json; tagger_params.json, read by tagger.calibrated(); tagger_fitted.json, the
simulated tagger's statistics, for the team to compare); under STRICT none is shown. The
terminal says, per knob, only whether it lies inside its grid (or at 0 or inf, its least or
most), and per statistic whether the simulated tagger reproduces it to within 10%.
"""

import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah import cache, frames, pipeline, safe_run, tagging  # noqa: E402
from harness import fuse, tagger  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402
from scripts.make_submission import team_sides  # noqa: E402


def main(report):
    cache.build(report)
    atd, events = cache.load_atd(), cache.load_smart_tagging()
    sides = team_sides(report, events)
    path = pm.label_codes(pipeline.predict(atd, events, sides, tagging_weight=0.0))
    vote = tagging.votes(tagging.phases(events, sides), len(path))
    live = frames.graded_mask(len(path)) & ~pipeline.before_kickoff(events, len(path))
    target = tagger.statistics(vote, path, tagger.runs_of(live))
    report.aggregates("tagging_stats", target)

    report.status("fitting the simulated tagger on the harness matches")
    cases = fuse.cases()
    knobs, inside = tagger.fit(cases, target)
    for knob, ok in inside.items():
        report.check("{} inside its grid".format(knob), ok)
    fitted = tagger.mean_statistics(cases, knobs)
    report.aggregates("tagger_fitted", fitted)
    for name in target:
        report.check("simulated tagger reproduces {} to within 10%".format(name),
                     np.isclose(fitted[name], target[name], rtol=0.1))
    report.aggregates(os.path.splitext(os.path.basename(tagger.CALIBRATED))[0],
                      dataclasses.asdict(knobs))


if __name__ == "__main__":
    sys.exit(safe_run.run(main, name="calibrate_tagger"))
