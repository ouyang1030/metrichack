# Technical document — draft

Skeleton for the deliverable SUBMISSION.md describes. Filled in as versions land; final by
2026-11-14.

## Approach

The task is treated as segmentation. Per frame, a gradient-boosted classifier gives
P(H, A, D) from 85 tracking features: how fast the players move together, where the ball was
last located and how it moves (towards which goal, how close to a line, how still), how close
each team is to it, the ball's changes of velocity next to a player of either team (touches)
and with both teams close (duels), how long each team has been nearest, and how each team's
block is placed and moving towards the goal it attacks -- in windows of up to 2 s before and
after the frame. A second classifier on the same features gives the probability that
possession switches, play stops or play restarts within 0.2 s. A second stage sees how both
classifiers' outputs move over up to 8 s before and after (stacked sequential learning, trained
on out-of-fold outputs) and re-estimates P(H, A, D). Each team quantity is measured from that
team's point of view and comes as a home / away pair, and every classifier is trained on every
frame twice, once with the teams swapped, so it cannot favour either side. An input-output HMM
over H, A and D turns these into one most likely, piecewise-constant path (Viterbi): the
evidence is tempered (kappa 0.6), and the probability of a switch, stop or restart at each
frame follows the change detector (gamma 1.5) instead of a constant rate. The trees are stored
as JSON and evaluated in numpy, so the submission needs nothing beyond numpy and pandas. v1, the
reference it replaced, used two logistic models on two features. Everything is fitted and
selected on other matches (IDSSE / DFL, made ATD-like by a noise simulator), never on BAR-ATH.
Methods are compared on DFL's labels and on the same labels under Metrica's rule that a touch
without control does not change possession (EXPERIMENTS.md).
The parsed cache is verified by rebuilding the four published baselines byte for byte.

## Assets used and dropped

| asset | status |
|---|---|
| tactical ATD | used (v0) |
| Smart Tagging | parsed and cached; not used in v0 |
| tactical homography | parsed and cached; not used |
| tactical video, broadcast feed | not used |

## Training data

IDSSE / DFL open data: 7 Bundesliga matches, labels (`BallStatus`, `BallPossession`) by DFL,
~1.0M frames (DATASETS.md). None of it comes from the provided match.

## Possession rules implemented

v2 learns the frame-level evidence from DFL's labels instead of hand-written rules (v1 used
collective deceleration and balls leaving the pitch for dead ball, the closer team for
possession). The touch and duel features give it what tells control from a deflection. The
decoder adds: no switch unless the evidence persists (the switch cost, lower where a change is
likely), so single touches and deflections do not change possession; while nothing speaks for
a change, possession stays with the last team; a stoppage always separates two possessions,
even of the same team.

## Limitations

- The noise simulator is calibrated on the ATD without our seeing its statistics (STRICT): a
  tracker that loses the ball in most stoppages, a little in play, with false detections. When
  the ball is lost and how the players slow down is reproduced; when it is found again and how
  crowded it is then, not (EXPERIMENTS.md). The decoder setting was chosen on the calibrated
  copies and a much noisier stress copy together.
- DFL's notion of possession and dead ball is not Metrica's; the harness compares methods, it
  does not predict BAR-ATH scores.
- Short possessions remain hard: on the calibrated copies v2 finds 46% of those under 4 s, 86%
  of the longer ones, and predicts about 0.8 times as many possessions as there are (DFL
  labels).
- The touch features need well-located balls; under the partly calibrated noise (with the ATD's
  false-detection rate) they still gave +0.03.
- Half of the training copies are drawn between the calibrated noise and the stress copy's.
  This scores the same on the calibrated copies and +0.09 on the stress copy, whose dropout
  differs exactly where the calibration is uncertain (when the ball is found again). Adopted by
  the team as an exception to our decision rule, which asks for a gain on the calibrated
  copies.
- Features that look across the ball's flight to the next contact (who receives it, how firmly)
  added nothing beyond the touch features and the second stage.

## Reproduction

See README.md: one command, `python scripts/make_submission.py`; two clean-room runs must give
byte-identical files (`scripts/check_reproducibility.sh`).
