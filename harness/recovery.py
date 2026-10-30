"""Does the calibration recover known knobs? (harness/calibrate.py, EXPERIMENTS.md)

    python -m harness.recovery

Copies of the DFL matches are made with knobs set by hand (TRUE), their statistics taken as the
target, and the simulator fitted to them from another seed, as scripts/calibrate_noise.py fits
it to the ATD. Prints each fitted knob beside its true value and each statistic's relative
error. Nothing here touches BAR-ATH.
"""

import dataclasses
import time

from harness import calibrate, degrade, idsse

# About what the calibration with lost stoppages found plausible (EXPERIMENTS.md, 10-29).
TRUE = dataclasses.replace(degrade.Params(), person_missing=0.1, gap_mean_s=2.0, jitter_m=0.1,
                           ball_dead_lost=0.85, ball_dead_seen_per_min=2.0, ball_missing=0.01,
                           ball_dead_on_s=0.4, ball_dead_off_s=-0.6, ball_false=0.01,
                           ball_jitter_m=0.1)
TARGET_SEED = 7

if __name__ == "__main__":
    start = time.time()
    matches = idsse.load_all()
    target = calibrate.mean_statistics(matches, TRUE, seed=TARGET_SEED)
    params, reached = calibrate.fit(matches, target)
    fitted = calibrate.mean_statistics(matches, params)
    for knob in sorted(set(calibrate.DROPOUT_KNOBS) | {"person_missing", "gap_mean_s",
                                                       "jitter_m", "ball_false",
                                                       "ball_jitter_m"}):
        print("%-24s true %8.3f  fitted %8.3f  %s" % (
            knob, getattr(TRUE, knob), getattr(params, knob),
            "" if reached.get(knob, True) else "at a bound"))
    for name in target:
        print("%-18s relative error %+.3f" % (name, fitted[name] / target[name] - 1))
    print("%.0f s" % (time.time() - start))
