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

## Ablations

On the off-game harness (leave one match out, DFL labels, the noise calibrated to the ATD),
each change against the first stage as accepted, spellF1 on the calibrated / stress copies
(EXPERIMENTS.md, 10-31; the same holds under Metrica's rule):

| change | spellF1 |
|---|---|
| trained on clean tracking instead of degraded copies | -0.106 / -0.268 |
| feature windows that do not look ahead | -0.017 / -0.031 |
| no team-swapped training copies | -0.006 / -0.011 |
| second stage over ±8 s added (v2 as submitted) | +0.026 / +0.029 |

Training under the simulated ATD noise matters most: a model fitted on clean tracking is the
best on clean tracking (+0.04) and the worst under noise.

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

Two more providers' matches check the exported models and are never trained or tuned on
(`harness/external.py`): Metrica's sample games 1-3 (labels derived from Metrica's human event
data by a written rule, `harness/metrica.py`), as provided, degraded and under stress; and
SkillCorner's 20 open broadcast matches (automatic labels from its phases of play and
possession group, `harness/skillcorner.py`), as provided. On Metrica v2 keeps the lead it has
on DFL (spellF1 0.805 degraded against v1's 0.498); on SkillCorner, where the broadcast camera
sees about 16 players, it falls slightly behind v1 on spellF1 (0.378 against 0.392) and stays
ahead on macroF1 (EXPERIMENTS.md, 10-31).

## Acknowledgements

IDSSE / DFL open data: Bassek, Rein, Weber & Memmert (2025), *Scientific Data* 12:195,
CC BY 4.0. Sample tracking and event data by Metrica Sports
(github.com/metrica-sports/sample-data). Open broadcast tracking data by SkillCorner
(github.com/SkillCorner/opendata, MIT).

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
- v2's features were learnt with every player tracked. With players missing, as on
  SkillCorner's broadcast tracking, it loses its lead over v1 on spellF1. The ATD is tracked
  from the tactical camera, so this matters only where it loses players.
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
