"""Calibrate the ATD noise simulator to the ATD (AGENTS.md 5.2), behind the firewall.

    python scripts/calibrate_noise.py

Measures the noise statistics of the ATD's graded frames (harness/calibrate.py), then fits the
simulator's knobs on the harness matches until their degraded copies show the same. All go to
cache/aggregates/ (atd_noise.json; noise_params.json, read by degrade.calibrated(); and
fitted_noise.json, the fitted copies' statistics, for the team to compare); under STRICT none is
shown. The terminal says, per knob, only whether the ATD's statistic lay within
the knob's search range, and per statistic whether the fitted copies reproduce it to within 10%.
"""

import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah import cache, frames, safe_run  # noqa: E402
from harness import calibrate, degrade, idsse  # noqa: E402
import games  # noqa: E402


def main(report):
    cache.build(report)
    atd = cache.load_atd()
    target = calibrate.statistics(atd.players, atd.ball, games.GAMES[games.GAME]["pitch"],
                                  frames.graded_runs())
    report.aggregates("atd_noise", target)

    report.status("fitting the simulator on the harness matches")
    matches = idsse.load_all()
    params, reached = calibrate.fit(matches, target)
    for knob, ok in reached.items():
        report.check("{} within its search range".format(knob), ok)
    fitted = calibrate.mean_statistics(matches, params)
    report.aggregates("fitted_noise", fitted)  # for the team to set beside atd_noise
    for name in target:
        report.check("fitted copies reproduce {} to within 10%".format(name),
                     np.isclose(fitted[name], target[name], rtol=0.1, atol=1e-4))
    report.aggregates(os.path.splitext(os.path.basename(degrade.CALIBRATED))[0],
                      dataclasses.asdict(params))


if __name__ == "__main__":
    sys.exit(safe_run.run(main, name="calibrate_noise"))
