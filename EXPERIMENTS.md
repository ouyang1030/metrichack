# Experiments

One entry per hypothesis (AGENTS.md section 9): date, hypothesis, change, off-game spellF1 /
macroF1 (clean and degraded) with intervals, BAR-ATH diagnostics, decision.

## 2026-10-27 — v0: scaffold, baseline reproduction

- **Hypothesis.** The parsed cache plus a vectorised nearest-player rule reproduces the four
  published baselines byte for byte, and the one-command pipeline emits `baseline_smooth50`.
- **Change.** `mtah/` (ATD parser, cache, firewall, vectorised baseline, submission helpers),
  `scripts/make_submission.py`, `scripts/reproduce_baselines.py`, clean-room script, tests.
- **Off-game harness.** Not built yet (v1).
- **Synthetic bundle (random positions, 199,319 frames).** Vectorised codes and all four files
  are byte-identical to the official `starter/baseline.py` run on the same files, edge cases
  included (ties, NaN distances). Two clean-room runs give byte-identical files. Full pipeline
  including cache build: about 2 s, 0.65 GB peak RSS.
- **BAR-ATH.** Pending: the data bundle was not on the machine. Expected on first run: the four
  "identical to the committed file" checks pass, and `make_submission` writes a file
  byte-identical to `evaluation/results/baseline_smooth50.csv`.
- **Decision.** Accept as v0 once the BAR-ATH checks pass.

## 2026-10-27 — v1 step 1: off-game harness and noise simulator

- **Hypothesis.** IDSSE / DFL tracking, degraded to look like the ATD, reproduces the way the
  published baselines behave on BAR-ATH, so it can rank methods.
- **Change.** `harness/idsse.py` (7 matches, 14 halves, labels from `BallStatus` /
  `BallPossession`), `harness/degrade.py` (default parameters from DATA.md numbers only: 41% ball
  missing on in-play frames, 20.5 tracked objects, 46 slots), `harness/score.py`.
- **Off-game harness**, mean over 14 halves, 5–95% bootstrap interval, degraded = mean of seeds
  0–2:

  | method | clean spellF1 | clean macroF1 | degraded spellF1 | degraded macroF1 |
  |---|---|---|---|---|
  | baseline_raw | 0.098 [0.093, 0.105] | 0.363 | 0.017 [0.014, 0.019] | 0.500 |
  | baseline_smooth25 | 0.297 [0.282, 0.314] | 0.385 | 0.210 [0.202, 0.218] | 0.461 |
  | baseline_smooth50 | 0.309 [0.294, 0.325] | 0.391 | 0.254 [0.244, 0.265] | 0.424 |
  | baseline_smooth125 | 0.219 [0.206, 0.231] | 0.387 | 0.207 [0.191, 0.222] | 0.385 |

- **Reading.** Degraded spellF1 ranks the baselines like the published full-match scores
  (smooth50 best, raw far behind; published 0.049 / 0.200 / 0.243 / 0.224). macroF1 does not
  match (published 0.64–0.68): the simulated dropouts are short (median gap 0.12 s), so the
  baselines almost never say D, while the real ATD presumably loses the ball for long stretches
  around dead balls. The published scores are a comparison point only, never a calibration
  target (COMPLIANCE_LOG.md). Calibration waits for ATD aggregates under STRICT.
- **Decision.** Accept the harness as the decision tool for v1; calibrate the simulator once the
  data bundle is in place.

## 2026-10-27 — v1 step 2: evidence + HMM decoder (accepted)

- **Hypothesis.** Collective player speed and where the ball was last seen separate dead from
  alive; the closer team to the ball separates H from A; an HMM over {H, A, D} turns that into
  piecewise-constant spells and beats smoothing.
- **Exploration (off-game, degraded seed 0, 4 matches).** P(dead) is 0.89–1.00 when the median
  player speed is below 1 m/s and under 0.05 above 2 m/s; a ball seen beyond a line is dead
  99.6% of the time; the nearest player's team owns the ball 92.5% of the time within 1 m,
  ~60% beyond 3 m.
- **Change.** `mtah/features.py` (ATD fields only: slot positions and speeds, ball x/y),
  `mtah/emissions.py` (logistic P(dead | x): speed, speed², ball seen, last seen out, log time
  since seen; logistic P(home | alive, x) on exp(-d_home/1.5 m) - exp(-d_away/1.5 m)),
  `mtah/decoder.py` (Viterbi; transitions from off-game label counts; evidence tempered by
  kappa), `harness/train_v1.py`, `models/v1.json`.
- **Off-game harness**, leave one match out (fit on degraded seed 0 of six matches, score the
  seventh clean and on seeds 0–2), 14 halves:

  | method | clean spellF1 | clean macroF1 | degraded spellF1 | degraded macroF1 |
  |---|---|---|---|---|
  | baseline_smooth50 | 0.309 [0.294, 0.325] | 0.391 | 0.254 [0.244, 0.265] | 0.424 |
  | v1 kappa=0.2 | 0.387 [0.377, 0.399] | 0.782 | 0.318 [0.303, 0.333] | 0.763 |
  | **v1 kappa=0.4** | **0.410 [0.400, 0.420]** | 0.776 | **0.334 [0.324, 0.345]** | 0.755 |
  | v1 kappa=0.7 | 0.377 [0.367, 0.386] | 0.769 | 0.327 [0.315, 0.340] | 0.746 |
  | v1 kappa=1.0 | 0.336 | 0.762 | 0.304 | 0.737 |

  Paired difference to smooth50, degraded: spellF1 **+0.080 [+0.070, +0.091]**, macroF1
  +0.331 [+0.316, +0.345]. kappa was picked on the same 14 halves (one parameter; neighbours
  within 0.01).
- **Where it still loses** (3 matches, seed 1): alive/dead F1 0.85–0.93 and the number of D
  segments is about right, but it predicts 130–150 possession spells per half where the labels
  have 175–275, with H/A accuracy 0.72–0.87. Team switches are too few and too late.
- **BAR-ATH.** Not run (no data). Synthetic full-length bundle: valid, two runs byte-identical.
- **Decision.** Accept; `mtah/pipeline.py` is now v1 (kappa 0.4).

## 2026-10-27 — separate temperatures for dead and team evidence (rejected)

- **Hypothesis.** Play stops rarely but possession changes often, so team evidence should be
  trusted more (kappa_team > kappa_dead) to recover the missing switches.
- **Result** (same protocol): best pair 0.3 / 0.4 at +0.084 [+0.075, +0.094] vs +0.080 for one
  kappa — within noise; kappa_team 0.8 no better, 1.6 clearly worse (+0.023). Trusting the
  nearest-player signal more adds wrong switches as fast as right ones.
- **Decision.** Keep one temperature (the code keeps the factorised form). The fix has to be
  better possession evidence, not a different weight on the current one.

## 2026-10-27 — where v1 loses: short possessions

- **Diagnosis** (all 14 halves, degraded seed 1, model fitted on all matches — for understanding
  only). Recall at IoU 0.5 by ground-truth spell length: < 1 s 0.02 (12.5% of spells), 1–2 s
  0.05 (17.5%), 2–4 s 0.09 (22.9%), 4–8 s 0.36 (20.3%), 8–16 s 0.57 (15.2%), > 16 s 0.74–0.80
  (11.6%). Half the DFL spells are shorter than 4 s and v1 finds almost none of them; it predicts
  1,979 spells where the labels have 3,061. DFL's notion of possession may be finer-grained than
  Metrica's; the harness compares methods, it does not say how many spells BAR-ATH has.

## 2026-10-27 — simulator: ball positional noise

- **Change.** `degrade.py` adds 0.3 m ball jitter (correlation 0.2 s), a guess until calibrated:
  without it, ball velocities on the harness would be exact. The harness break between halves is
  now 15 min (was 1 min), so that windows of a few minutes never straddle two halves.
- **Effect.** v1 (near, kappa 0.4) degraded spellF1 0.334 -> 0.336; smooth50 0.254 -> 0.249.
  New reference numbers below are on this simulator.

## 2026-10-27 — control evidence: ball and player moving together (rejected)

- **Hypothesis.** A nearby player whose velocity matches the ball's controls it; that sharpens
  team evidence enough to catch quick turnovers.
- **Change.** `features.py`: ball and slot velocities by central differences (±2 frames), relative
  speed to the nearest slot of each team; team term `control` = closeness × exp(-rel / 2 m/s).
- **Result.** Weight 2.0 next to closeness 5.0, but within 1.5 m the nearest team owns the ball
  83–95% of the time whatever the relative speed, and with 0.3 m ball jitter ±2 m/s is normal
  even when dribbling. Paired difference to v1: spellF1 +0.000 [-0.001, +0.001].
- **Decision.** Reject. The features stay in `features.py` (cheap), unused by the model.

## 2026-10-27 — team movement while the ball is lost (accepted)

- **Hypothesis.** The whole block moves towards the goal of the team without the ball; that
  says who has it on the 41% of frames without a ball.
- **Exploration.** Sign of the mean player velocity towards the away goal (direction of attack
  from where the teams sit over ±2 min): right 64% of alive frames, 68% when the ball is lost,
  72.5% when the movement is clear. Spread (home more stretched along the pitch): 60%. Relative
  height of the teams: no signal (47%).
- **Change.** `features.py`: `flow`, `spread`; the team model now trains on every alive frame.
- **Off-game harness** (leave one match out, degraded seeds 0–2, 14 halves):

  | variant (kappa 0.4) | clean spellF1 | degraded spellF1 | degraded macroF1 |
  |---|---|---|---|
  | baseline_smooth50 | 0.309 | 0.249 [0.236, 0.263] | 0.422 |
  | near (v1) | 0.407 | 0.336 [0.324, 0.349] | 0.761 |
  | **near+flow** | 0.411 | **0.358 [0.347, 0.370]** | 0.771 |
  | near+flow+spread | 0.404 | 0.345 [0.333, 0.359] | 0.772 |

  near+flow minus near: spellF1 **+0.022 [+0.009, +0.035]**, macroF1 +0.011 [+0.003, +0.018].
  Adding spread: +0.009 [-0.004, +0.022] over near, i.e. worse than flow alone.
- **Decision.** Accept near+flow; reject spread.

## 2026-10-27 — duration chains and the temperature (chains rejected, kappa 0.2)

- **Hypothesis.** Possessions have a typical length, not a geometric one; k sub-states per state
  (minimum k frames, duration peaking above zero) cut implausible spells.
- **Result** (near+flow, same protocol): chain 2 at kappa 0.4 0.372, chain 3 at 0.55 0.372, vs
  chain 1 at 0.25 0.371 — paired +0.002 [+0.001, +0.003] and +0.001. The chain and the
  temperature trade off against each other; neither adds anything. Extending the temperature
  grid for chain 1: 0.1 0.347, 0.15 0.364, **0.2 0.371**, 0.25 0.371 (clean 0.420 / 0.427).
- **Decision.** Reject chains (within noise, keep the simpler decoder); kappa 0.2.

## 2026-10-27 — boundary refinement (rejected)

- **Diagnosis** (matched spells, IoU >= 0.3, degraded seed 1): starts unbiased (median 0
  frames), ends late — median +20 frames before a stoppage, +22 before a turnover.
- **Hypothesis.** Moving each switch back to its physical onset (the new team's first located
  touch; the ball first seen beyond a line), at most 2 s and never past the previous switch,
  removes that lag.
- **Result.** Degraded spellF1 0.374 vs 0.371, paired +0.003 [+0.001, +0.006]; clean 0.415 vs
  0.420.
- **Decision.** Reject (within noise, clean worse). `mtah/refine.py` stays, switched off.

## v1 as it stands (2026-10-27)

near+flow, one temperature 0.2, plain HMM, no refinement: degraded spellF1 **0.371
[0.359, 0.384]**, macroF1 0.782; clean 0.420 / 0.793. Against baseline_smooth50 on the same
copies: spellF1 **+0.122 [+0.108, +0.138]**, macroF1 +0.360 [+0.343, +0.377]. Not run on BAR-ATH
(no data). Main open problem: possessions shorter than ~4 s.

## 2026-10-27 — v2: one gradient-boosted frame classifier (accepted)

- **Hypothesis.** v1 sees two numbers per frame. A classifier over a wide table of
  permutation-invariant tracking features, with context before *and after* each frame (the
  pipeline is offline), gives evidence sharp enough for the same HMM to find the short
  possessions v1 misses (v1 found 5.6% of those under 4 s).
- **Change.** `features.matrix`: 66 columns — collective and per-team speed, ball located / time
  since and until located / last and next location beyond a line, ball speed, stillness and
  place (distance to the touchline, the goal line, each team's target goal), ball velocity
  towards each team's goal, per team: nearest distance, closeness in 0.5 / 1 / 2 s centred
  windows and 2 s before / after, players within 2 / 5 / 10 m, depth, push towards goal (same
  windows), length, width. Every team column comes as a home / away pair measured towards the
  goal that team attacks, so swapping the teams permutes the columns (tested). Classifier:
  scikit-learn `HistGradientBoostingClassifier`, 100 iterations, 31 leaves, at least 100 frames
  per leaf, learning rate 0.1, no early stopping — defaults apart from the leaf size, not tuned.
  Trained on every 5th in-play frame of degraded seed 0 of the training matches plus the same
  frames with the teams swapped. P(H, A, D | frame) feeds the v1 HMM (`decode_probs`). The
  trees are exported to JSON (811 kB) and evaluated in numpy (`mtah/trees.py`; equal to
  scikit-learn to 1e-12, tested), so the pipeline needs no scikit-learn.
- **Stress copy.** New in the protocol: one more copy of each held-out match, much noisier than
  the simulator's defaults (ball missing 55%, no extra dropout off the pitch, players missing
  15%, team errors 5%, swaps 20% within 2 m, jitter 0.4 m / ball 0.6 m). Never trained on. It
  asks whether a gain rests on the particular noise the classifier saw in training.
- **Result** (leave one match out, 14 halves, spellF1 [5-95%] / macroF1):

  | | clean | degraded | stress |
  |---|---|---|---|
  | baseline_smooth50 | 0.309 / 0.391 | 0.249 / 0.422 | 0.162 / 0.331 |
  | v1 (kappa 0.2) | 0.420 / 0.793 | 0.371 [0.359, 0.384] / 0.782 | 0.296 / 0.729 |
  | v2, kappa 0.2 | 0.593 / 0.920 | 0.578 / 0.909 | 0.485 / 0.845 |
  | **v2, kappa 0.3** | 0.611 / 0.922 | **0.589 [0.569, 0.609]** / 0.910 | **0.486** / 0.842 |
  | v2, kappa 0.6 | 0.632 / 0.924 | 0.595 / 0.910 | 0.464 / 0.837 |
  | v2, kappa 1.0 | 0.636 / 0.925 | 0.591 / 0.910 | 0.436 / 0.834 |

  v2 (kappa 0.3) minus v1, paired: degraded spellF1 **+0.218 [+0.201, +0.235]**, macroF1 +0.128
  [+0.118, +0.137]; stress +0.190 [+0.170, +0.209] / +0.113. Minus smooth50: degraded +0.340
  [+0.319, +0.363].
  Possessions found (IoU 0.5, degraded, 3 seeds): under 4 s 5.6% (v1) -> 23.1%; 4 s and longer
  55.7% -> 78.6%. Predicted / true possessions 0.57 -> 0.65: still too few switches.
- **Temperature.** The degraded score is flat from 0.3 to 1.0 (0.589-0.595) while the stress score
  falls (0.486 -> 0.436): sharper evidence pays only when the noise is the noise trained on. The
  ATD's noise level is not known yet, so kappa is chosen on the mean of degraded and stress
  spellF1 -> 0.3. Re-choose after calibration.
- **Too good to be true?** Checked before accepting. No input is derived from the labels
  (`harness/idsse.py` takes only positions and speeds; possession and status go to the labels).
  Folds are whole matches. The gain holds on the stress copy. Permutation importance (log loss
  0.18-0.21 on a held-out degraded copy, three folds; increase when a group is shuffled): ball
  motion and place +0.53-0.54, players' speed +0.35-0.43, team depth / push / shape
  +0.16-0.19, closeness windows +0.12-0.15, ball located / out +0.03-0.04, closeness of this
  frame alone ~0.01 (the windows carry it). Nothing points at an artefact; the gain looks like
  context plus non-linearity over a v1 that saw two numbers. What the harness cannot test:
  whether the ATD's noise and Metrica's label semantics are close enough to the simulator and
  DFL's for the gain to carry over.
- **Decision.** Accept v2 at kappa 0.3 (`models/v2.json`, refitted on all seven matches); the
  pipeline now runs v2 (7 s and 1.2 GB on a full-length synthetic bundle, two runs
  byte-identical). Cross-validation reruns give identical numbers.

## 2026-10-27 — diagnosis: where v2 loses possessions (no change)

Folds 0, 2, 4, 6 held out, degraded seed 1, kappa 0.3; 1,704 true possessions.

| true length | share | found | missed outright | merged into a longer spell | split | shifted |
|---|---|---|---|---|---|---|
| < 2 s | 30% | 0.08 | 0.59 | 0.26 | 0.00 | 0.07 |
| 2-4 s | 22% | 0.38 | 0.26 | 0.23 | 0.00 | 0.12 |
| 4-10 s | 26% | 0.69 | 0.04 | 0.21 | 0.00 | 0.06 |
| >= 10 s | 22% | 0.94 | 0.00 | 0.05 | 0.01 | 0.00 |

- v2 under-segments and never over-segments. Matched spells: start median 0 frames (q10 -34,
  q90 +21), end median +4 (q10 -25, q90 +52).
- What separates the two halves of a merge: true segments lying between two spells of one team,
  reproduced by the decoder — the other team's spell: under 1 s 33% (131), 1-2 s 45% (163),
  2-4 s 68% (225), 4-10 s 95%; a stoppage: only 22 such stoppages, almost all found. All
  stoppages: 94-100% found from 2 s up; none shorter than 1 s exists.
- Oracle boundaries (the true segments, each labelled by v2's mean probabilities): spellF1 0.696
  against 0.593 decoded on the same folds. The headroom is in the segmentation, not the labels.
- Reading: the misses are short spells of the other team inside one team's possession. Under
  Metrica's written rule (a touch without control does not change possession) many of these
  would not be spells at all, so chasing them on DFL labels risks flicker on BAR-ATH. Plan:
  docs/PLAN_v2.md, step 0 first.

## 2026-10-28 — plan step 0: Metrica's possession rule as a second label set (harness change; training on it rejected)

- **Hypothesis.** Some of DFL's sub-second spells of the other team are touches without control,
  which Metrica's ground truth does not count (AGENTS.md section 3). Scoring only on DFL's
  labels would reward chasing them.
- **Change.** `harness/semantics.py`: a spell of one team shorter than a control time, between
  two spells of the other team, goes to the other team (left to right; spells next to a stoppage
  kept; tested). Every prediction is now scored against three label sets: DFL (as is), M1
  (control time 1 s) and M2 (2 s, a check). On the seven matches M1 removes 12% of the
  possessions (3,061 -> 2,705; under 1 s 13% -> 6%; median 3.7 -> 4.4 s) but only 0.2% of the
  frames; M2 removes 26%. The temperature is now chosen on the mean of degraded and stress
  spellF1 over DFL and M1. Organiser question 6 asks which rule the ground truth follows.
- **Result** (spellF1, degraded [5-95%]; stress):

  | | DFL | M1 | M2 |
  |---|---|---|---|
  | v1 | 0.371 / 0.296 | 0.405 / 0.325 | 0.461 / 0.362 |
  | v2, kappa 0.2 | 0.578 / 0.485 | 0.629 / 0.524 | **0.690** / 0.572 |
  | **v2, kappa 0.3** | 0.589 [0.569, 0.609] / 0.486 | 0.638 [0.625, 0.651] / 0.521 | 0.686 / 0.556 |
  | v2, kappa 0.6 | 0.595 / 0.464 | 0.638 / 0.492 | 0.667 / 0.508 |
  | v2, kappa 1.0 | 0.591 / 0.436 | 0.630 / 0.458 | 0.645 / 0.465 |
  | v2 trained on M1, kappa 0.3 | 0.586 / 0.484 | 0.635 / 0.521 | 0.685 / 0.560 |

  Trained on M1 minus trained on DFL (kappa 0.3 each): DFL -0.003 [-0.007, +0.000], M1 -0.003
  [-0.007, +0.000], M2 -0.001 [-0.004, +0.002]; stress within +-0.004. v2 minus v1 holds on
  every label set (+0.218 / +0.233 / +0.225).
- **Reading.** The label rule moves the scores by 0.05-0.10 but not the ranking of v1 and v2.
  The temperature is where it matters: the stricter the rule, the more a high kappa (more
  switches) costs; under M2 the best is 0.2 and kappa 1.0 loses 0.045. Chasing DFL's
  sub-second spells by sharpening the evidence would be a bet on DFL's semantics. Predicted /
  true possessions at kappa 0.3: 0.65 (DFL), 0.74 (M1), 0.88 (M2).
- **Decision.** Keep training on DFL labels (equal within noise; no change). Keep kappa 0.3 (best
  on the new criterion too). Every later step is judged on DFL and M1 and must not lose on M2.
  `models/v2.json` unchanged.

## 2026-10-28 — plan step 1: more and randomised training noise (three default copies accepted; randomised noise deferred)

- **Hypothesis** (domain randomisation, Tobin 2017). Training on one default-noise copy fits the
  classifier to one guess of the ATD's noise; noise drawn over a range should hold up better
  when the real noise differs.
- **Change.** `degrade.drawn()`: every simulator parameter that differs between the defaults and
  the stress copy is drawn uniformly between the two. Training copies per match (each gives every
  5th in-play frame, the n-th copy from the n-th frame on): one default copy (as before), three
  default copies (seeds 0-2), one or three randomised copies (seeds 10-12). Note: the stress copy
  now lies at the edge of the randomised training range, so for those variants it is no longer
  noise never trained on.
- **Result** (spellF1 degraded / stress; each variant at its best kappa, chosen as in step 0):

  | training copies | kappa | DFL | M1 | M2 |
  |---|---|---|---|---|
  | one default (before) | 0.3 | 0.589 / 0.486 | 0.638 / 0.521 | 0.686 / 0.556 |
  | **three default** | 0.3 | **0.595** / 0.492 | **0.644** / 0.528 | **0.692** / 0.566 |
  | one randomised | 0.6 | 0.567 / 0.538 | 0.611 / 0.576 | 0.656 / 0.609 |
  | three randomised | 0.45 | 0.584 / 0.544 | 0.630 / 0.585 | 0.676 / 0.621 |

  Minus one default copy, paired, degraded: three default +0.006 [+0.002, +0.010] (DFL),
  +0.006 [+0.001, +0.010] (M1), +0.006 [+0.001, +0.011] (M2), stress +0.006 to +0.011, macroF1
  +0.001 to +0.003. Three randomised: degraded -0.005 [-0.008, -0.003] / -0.008 / -0.010,
  stress +0.058 / +0.064 / +0.065, macroF1 -0.005 degraded, +0.044 stress.
- **Reading.** More copies of the same noise help a little, on every label set. Randomised noise
  buys a lot of robustness to heavy noise at a small cost under the default noise: which one
  wins on BAR-ATH depends on how noisy the ATD really is, which only the calibration can say.
- **Decision.** Accept three default copies (beyond the interval on all three label sets). Keep
  randomised-only training for the re-choice after calibration. Next: default + randomised
  copies together, run with step 2.

## 2026-10-28 — plan step 2: switch-aware decoding (change detector accepted; switch factor rejected) and mixed training noise (deferred)

Reference: v2 trained on three default copies (step 1), constant-transition HMM, kappa 0.3.

- **Mixed noise** (three default + three randomised copies per match; best kappa 0.45). Minus
  the reference, degraded: DFL -0.000 [-0.007, +0.007], M1 -0.002 [-0.009, +0.006], M2 -0.007
  [-0.013, -0.000]; stress +0.046 / +0.051 / +0.053, macroF1 stress +0.038. Equal under the
  default noise (slightly worse on M2), much better under heavy noise; the stress copy is inside
  its training range. **Deferred** to the calibration with the randomised-only variant: if the ATD
  is noisier than the simulator's defaults, this is the one to take.
- **2a. Switch factor** (possession switches H <-> A made 2x or 0.5x as likely per frame, kappa
  re-chosen): every difference within +-0.003. A constant cost per switch mostly duplicates what
  kappa does (kappa scales the evidence against all transition costs at once). **Rejected**,
  code removed.
- **2b. Change detector, input-output HMM** (Bengio & Frasconi; PathCRF). A second classifier
  on the same 66 features and copies gives P(no change, possession switch, stop, restart within
  +-0.2 s | frame) (`harness/train_v2.py: change_kinds, fit_first`). Its P(kind) / 11 is that
  kind's per-frame hazard, blended with the HMM's constant one in log space by gamma
  (`decoder.input_transitions`; gamma 0 is the plain HMM). Result, spellF1 degraded / stress:

  | | DFL | M1 | M2 |
  |---|---|---|---|
  | constant HMM, kappa 0.3 | 0.595 / 0.492 | 0.644 / 0.528 | 0.692 / 0.566 |
  | detector, gamma 0.5, kappa 0.3 | 0.598 / 0.493 | 0.646 / 0.528 | 0.691 / 0.565 |
  | detector, gamma 1, kappa 0.3 | 0.604 / 0.496 | 0.651 / 0.530 | 0.693 / 0.563 |
  | **detector, gamma 1, kappa 0.2** (chosen) | 0.598 / 0.501 | 0.647 / 0.540 | 0.699 / 0.583 |
  | detector, gamma 1, kappa 0.45 | 0.610 / 0.486 | 0.655 / 0.517 | 0.684 / 0.538 |

  Chosen minus reference, paired, degraded: DFL +0.002 [+0.000, +0.005], M1 +0.003 [+0.001,
  +0.006], M2 +0.006 [+0.004, +0.008]; stress +0.009 [+0.003, +0.014] / +0.012 / +0.017; macroF1
  +0.002 degraded, +0.004 stress. Clean: 0.618 -> 0.625 (0.636 at kappa 0.3). gamma 1 beats 0.5
  everywhere.
- **Reading.** A real but small gain, on every label set and noise level; the decoder now knows
  *where* a switch is plausible, not only how often. It does not fix the merges by itself:
  predicted / true possessions stay at 0.63-0.71.
- **Decision.** Accept the change detector, gamma 1, kappa 0.2. Next run tries gamma 1.5 too.
- **Run note.** This run took 4.6 h instead of ~25 min: a second training job ran alongside it
  (two OpenMP jobs on the same cores), and the machine was short of memory (swapping) and slept.
  Now one job at a time, under `caffeinate`; randomised copies are only built when a variant
  uses them, and second-stage tables are built one at a time.

## 2026-10-28 — plan step 3: stacked second stage over ±8 s (accepted)

Reference: three default copies, change detector (step 2), now also tried at gamma 1.5.

- **Hypothesis** (stacked sequential learning, Cohen & Carvalho 2005; auto-context, Tu 2008;
  MS-TCN). The first stage sees +-2 s; a second stage that sees how the first stage's
  probabilities and change scores move over up to +-8 s can tell a real switch from a blip and
  find the spells the decoder merges.
- **Change.** `features.context` / `second_stage`: the 66 features plus, for P(H), P(A), P(D) and
  the detector's P(switch), P(stop), P(restart), the value and its means over 0.5, 1, 2, 4 and
  8 s before and after the frame and after minus before (96 columns). The second classifier
  (same settings) learns from out-of-fold first-stage outputs: for each training match the first
  stage is refitted on the other training matches (`fit_second`). The detector still drives the
  transitions. Cost: seven first-stage fits per fold instead of one (6 min per fold).
- **Result** (spellF1, degraded [5-95%] / stress):

  | | DFL | M1 | M2 |
  |---|---|---|---|
  | constant HMM, kappa 0.3 | 0.595 / 0.492 | 0.644 / 0.528 | 0.692 / 0.566 |
  | detector, kappa 0.2, gamma 1 (step 2) | 0.598 / 0.501 | 0.647 / 0.540 | 0.699 / 0.583 |
  | detector, kappa 0.3, gamma 1.5 | 0.609 / 0.498 | 0.654 / 0.532 | 0.689 / 0.559 |
  | **stacked, kappa 0.3, gamma 1.5** | **0.621** [0.602, 0.640] / 0.526 | **0.670** [0.657, 0.681] / 0.564 | **0.704** / 0.593 |
  | stacked, kappa 0.2, gamma 1 | 0.608 / 0.517 | 0.660 / 0.556 | 0.713 / 0.604 |

  Stacked minus detector alone (both kappa 0.3, gamma 1.5), paired, degraded: DFL +0.012
  [+0.008, +0.015], M1 +0.015 [+0.011, +0.019], M2 +0.015 [+0.012, +0.019]; stress +0.029 /
  +0.032 / +0.034; macroF1 +0.006 degraded, +0.010 stress. Clean 0.647 -> 0.656. Short
  possessions found (DFL, degraded): 27.2% -> 27.7%; 4 s and longer 79.8% -> 80.9%.
- **Reading.** The longer view helps everywhere, most under heavy noise, where +-2 s says least.
  Gamma 1.5 over 1 helps on DFL and M1, costs a little on M2 -- the label-semantics trade-off of
  step 0 again. Since the start of the plan: DFL 0.589 -> 0.621, M1 0.638 -> 0.670, M2 0.686 ->
  0.704, stress 0.486 -> 0.526.
- **Decision.** Accept the second stage, kappa 0.3, gamma 1.5.

## 2026-10-28 — plan step 4: touch and duel features (accepted)

Reference: the stacked model of step 3 (kappa 0.3, gamma 1.5), refitted in the same run with the
new columns left out; it reproduces step 3's numbers exactly.

- **Hypothesis** (Vidal-Codina et al. 2022; Link & Hoernig 2017). Control and a deflection both
  change the ball's velocity next to a player; who touches it, how often, and whether an opponent
  is as close tell one from the other -- which is Metrica's rule.
- **Change** (`features.matrix`, 19 new columns, 85 in all): `impulse`, how much the ball's
  velocity changes from the 0.2 s before a frame to the 0.2 s after; per team, that impulse
  where the team's nearest player is within 1.5 m of the ball and no opponent is nearer, in
  windows of 0.5, 1 and 2 s before and after; `duel`, both teams within 1.5 m (0.5 and 1 s
  windows); per team, how long it has been and will stay nearest to where the ball was last
  seen. 1.5 m rather than the papers' 1 m because the simulator's ball jitters by 0.3 m.
- **Result** (spellF1, degraded [5-95%] / stress; clean in brackets):

  | | DFL | M1 | M2 |
  |---|---|---|---|
  | stacked, no touch features | 0.621 [0.602, 0.640] / 0.526 (0.656) | 0.670 / 0.564 (0.701) | 0.704 / 0.593 (0.737) |
  | **stacked, touch features** | **0.635** [0.619, 0.652] / 0.526 (0.707) | **0.683** / 0.564 (0.751) | **0.714** / 0.587 (0.762) |

  Paired, degraded: DFL +0.014 [+0.008, +0.020], M1 +0.013 [+0.007, +0.020], M2 +0.010
  [+0.005, +0.016]; macroF1 +0.003 [+0.002, +0.004]. Stress: -0.000 [-0.006, +0.005], -0.000,
  -0.006 [-0.013, +0.001]. Clean +0.051, macroF1 0.932 -> 0.944. Short possessions found (DFL,
  degraded) 27.7% -> 29.5%; 4 s and longer 80.9% -> 82.5%.
- **Reading.** The gain grows with how precisely the ball is located: large on clean data,
  +0.01 under the default noise, nothing under the stress copy's 0.6 m ball jitter and 55%
  dropout. So on BAR-ATH it depends on how good the ATD's ball positions are, and the simulator
  does not model false ball detections yet. No leakage: the columns come from positions only.
- **Decision.** Accept (beyond the interval on all three label sets, no loss under stress).
  Re-check after the calibration, with false detections added to the simulator.
- **Run note.** The two stacked models per fold took 11.5 min per fold when the machine was
  awake and had memory; the first two folds took hours (sleep, swapping).

## v2 as it stands (2026-10-28, after plan steps 0-4)

Three default-noise training copies per match; first-stage classifier (85 features) and change
detector; second stage over +-8 s of both, trained on out-of-fold outputs; input-output HMM,
kappa 0.3, gamma 1.5. `models/v2.json` refitted on all seven matches (2.7 MB). Leave one match
out, spellF1 degraded [5-95%] / stress / clean:

| | DFL | M1 | M2 |
|---|---|---|---|
| baseline_smooth50 | 0.249 / 0.162 / 0.309 | 0.261 / 0.166 / 0.329 | 0.264 / 0.161 / 0.343 |
| v1 | 0.371 / 0.296 / 0.420 | 0.405 / 0.325 / 0.460 | 0.461 / 0.362 / 0.513 |
| v2 as accepted on 10-27 | 0.589 / 0.486 / 0.611 | 0.638 / 0.521 / 0.661 | 0.686 / 0.556 / 0.712 |
| **v2 now** | **0.635** [0.619, 0.652] / 0.526 / 0.707 | **0.683** / 0.564 / 0.751 | **0.714** / 0.587 / 0.762 |

macroF1 degraded 0.910 -> 0.923. Steps, each paired against the previous (DFL, degraded): three
copies +0.006, change detector +0.002 (+0.009 stress), second stage +0.012, touch features
+0.014. Pipeline on a full-length synthetic bundle: 19.5 s, 1.6 GB, validator passes, two runs
byte-identical. Still open: the merges (predicted / true possessions 0.69 on DFL); the choice
between default and randomised training noise, which waits for the calibration.

## 2026-10-29 — diagnosis of v2 after plan steps 0-4 (no change)

Same protocol as the diagnosis of 10-27 (folds 0, 2, 4, 6, degraded seed 1), current model
(stacked, touch features, change detector, kappa 0.3, gamma 1.5). In brackets: 10-27.

| true length (DFL) | n | found | missed outright | merged into a longer spell | split | shifted |
|---|---|---|---|---|---|---|
| < 2 s | 519 | 0.15 (0.08) | 0.53 (0.59) | 0.23 (0.26) | 0.00 | 0.08 |
| 2-4 s | 375 | 0.47 (0.38) | 0.23 (0.26) | 0.21 (0.23) | 0.00 | 0.09 |
| 4-10 s | 442 | 0.73 (0.69) | 0.04 (0.04) | 0.17 (0.21) | 0.00 | 0.07 |
| >= 10 s | 368 | 0.95 (0.94) | 0.00 | 0.04 (0.05) | 0.01 | 0.00 |

- Matched spells: start median -1 frame (q10 -30, q90 +18; was -34, +21), end +1 (q10 -21, q90
  +39; was +4, -25, +52). Predicted / true possessions 0.68 (0.65); on M1 0.78.
- Decoded 0.646, oracle boundaries 0.739 (0.593 / 0.696): both up ~0.05, the gap unchanged.
  On M1: 0.701 / 0.800.
- **The decoder is no longer what loses the short spells.** For the other team's segments
  between two spells of one team, the share the decoder reproduces equals the share where the
  evidence (second-stage probabilities averaged over the segment) favours the true team: under
  1 s 0.40 / 0.39 (n 131), 1-2 s 0.52 / 0.54 (163), 2-4 s 0.73 / 0.70 (225), 4-10 s 0.95 / 0.92.
  Where the evidence points the right way, the decoder finds it; where it does not, no decoder
  can.
- **Learning curve** (first stage + change detector, no second stage, trained on 2 / 4 / 6
  matches): DFL 0.614 / 0.620 / 0.628, M1 0.662 / 0.673 / 0.679 -- about +0.008 per doubling.
  More matches of the same kind would help this model little.
- **Reading for plan step 5.** What is missing is information in the evidence about short
  spells -- who receives or intercepts the ball -- not decoding and not data volume for the
  trees. A neural sequence model over the raw player set could in principle find that
  information, but it needs far more than seven matches (the trees gain little from more), CPU
  training on this machine, and an export to numpy. Decision: not now; first targeted features
  for receptions and interceptions (where a ball flight ends and with whom), which test the same
  hypothesis for a fraction of the cost.

## 2026-10-29 — simulator: false ball detections; the touch features under them (kept)

- **Simulator change.** `degrade.false_detections`: in runs of 0.5 s on average, the reported
  ball is something else -- half the runs a player (the "ball" sits at his feet and follows
  him), the rest a fixed spot on the pitch or up to 3 m beyond its lines. `Params.ball_false`
  is the share of in-play frames; 0 by default (DATA.md says nothing about it, every earlier
  copy stays bit-identical; tested), to be set by the calibration (docs/PLAN_v2.md).
- **Experiment.** The step-4 comparison (stacked, with vs without the touch features) rerun with
  false detections in every training and scored copy: 5% of in-play frames in the default
  copies, 15% in the stress copy (guesses).
- **Result** (spellF1, degraded / stress; both variants at kappa 0.3, gamma 1.5):

  | | DFL | M1 | M2 |
  |---|---|---|---|
  | no touch features | 0.602 / 0.506 | 0.652 / 0.546 | 0.697 / 0.577 |
  | touch features | 0.613 / 0.505 | 0.661 / 0.544 | 0.702 / 0.573 |
  | gain with false detections | +0.011 / -0.001 | +0.009 / -0.002 | +0.005 / -0.004 |
  | gain without them (step 4) | +0.014 / -0.000 | +0.013 / -0.000 | +0.010 / -0.006 |

  At each variant's chosen setting (touch: kappa 0.45, gamma 1), paired, degraded: DFL +0.015
  [+0.012, +0.019], M1 +0.013 [+0.010, +0.017], M2 +0.005 [-0.002, +0.012]; stress +0.002,
  -0.003, -0.010 [-0.018, -0.002]. Clean +0.053. False detections cost both variants about
  0.02 (no touch 0.621 -> 0.602, touch 0.635 -> 0.613 on DFL); v1 0.371 -> 0.350.
- **Reading.** About two thirds of the touch features' gain survives 5% false detections; on
  M2 it halves and is no longer certain; under heavy noise (stress, 15%) it is gone and slightly
  negative on M2. The trees learn to discount touches that a false ball fakes, but only so far.
- **Decision.** Keep the touch features. Re-check once the calibration gives the ATD's
  false-detection rate; above ~15% they would no longer pay. `models/v2.json` unchanged
  (trained without false detections, as the default simulator).

## 2026-10-29 — the data bundle on the machine: checks, diagnostics, the pre-kickoff stretch

- **Checks** (all behind `mtah/safe_run.py`, all PASS): the parser rebuilds the four published
  baselines byte for byte, against the committed files and against a fresh run of
  `starter/baseline.py`; two runs of the one command give byte-identical files; the official
  validator accepts the file; Smart Tagging's kickoff windows contain the published kickoffs.
- **Public window, alarm only** (COMPLIANCE_LOG.md): v2 spellF1 0.563, macroF1 0.847;
  `baseline_smooth50` 0.326 / 0.621, as published. No sign of misalignment or a label swap.
- **Unsupervised diagnostics** (`scripts/diagnose.py`, our predictions only): 292 possessions,
  duration p10/p50/p90 1.9 / 6.7 / 32.2 s, 1.4% under 1 s, 44.1% dead. Agreement with Smart
  Tagging team tags: 0.72 in attacking phases (the tagged team), 0.86 in defending phases (the
  other team). Stability with the ball removed on about 10% more frames: 1.1% of graded frames
  change, possessions x0.98, spellF1 of the perturbed run against the original 0.97.
- **Bug found:** of the graded stretch before the second-half kickoff -- dead in the ground
  truth (DATA.md) -- only 31% was predicted dead; the harness cannot show this (its halves start
  at kickoff). **Fix** (v2.1, `pipeline.before_kickoff`): graded frames before a half's Smart
  Tagging "Kick Off (Start)" window are dead; inside the window the decoder places the restart.
  After: 97.5% of the stretch dead, 284 possessions (8 spurious ones gone), the other
  diagnostics unchanged. No frame number of ours in the code; not scored on the public window.

## 2026-10-29 — calibrating the simulator to the ATD (STRICT; partly done)

- **Method** (`harness/calibrate.py`, `scripts/calibrate_noise.py`). Eleven noise statistics
  that need no labels, computed alike on the ATD's graded frames and on degraded DFL matches:
  ball missing share; mean gap; share of gaps of 1-2 frames (flicker); share of the missing time
  in gaps of 5 s or more (long losses); crowd within 3 m of the ball on missing vs seen frames
  (the ball put on the line between its sightings); ball speed before a gap vs overall; share of
  frame-to-frame moves over 1.5 m (jumps: false detections); ball and player jitter (robust
  spread around a centred mean); tracked players per frame; mean tracklet length. The ATD's go
  to `cache/aggregates/atd_noise.json`, unseen.
- **Simulator changes** (default behaviour as before in distribution; realisations differ from
  earlier entries, which are therefore not bit-reproducible with today's code): dropout noise is
  a mix of rough (exponential kernel, flickers at the threshold) and smooth (Gaussian kernel)
  noise, `ball_rough`; extra dropout while the ball is dead, `ball_weight_dead`; each part of the
  ball's noise draws from its own stream and false detections come from a fixed pool (common
  random numbers); the missing share is exact after false detections. Why: the first fit showed
  the old model could not reach the ATD's gap statistics -- the exponential-kernel noise
  crosses its threshold in bursts, which caps the mean gap at about 2 s whatever its
  correlation time -- and that fitting all ball knobs at once ended in poor local minima.
- **Fit.** Player knobs one at a time by bisection; the missing share directly; the five
  dropout knobs (smooth/rough mix, correlation time, dead, crowd and speed weights) jointly by
  bounded least squares; false detections and ball jitter by bisection on their own statistics.
- **Recovery check on DFL** (copies made with known knobs, fitted from another seed): all
  eleven statistics reproduced within 0.3%; person_missing, gap_mean_s, jitter, ball_false and
  ball_jitter recovered; the dropout knobs are not individually identifiable (another
  combination gives the same statistics), which is all the classifier sees.
- **On the ATD:** tracked players, tracklet length, player jitter, missing share, jumps and
  ball jitter reproduced within 10% (player jitter at the bottom of its range); the five dropout
  statistics (mean gap, flicker, long losses, crowd and speed lifts) are not, with no knob at a
  bound: the ATD's combination lies outside what the dropout model can make. The fitted knobs
  are at least as close to it as the defaults (least squares from the defaults).
- **Decision.** Use the partial calibration for now; ask the team, who may look at
  `cache/aggregates/atd_noise.json` beside `fitted_noise.json`, how the dropout differs, in
  generic terms. A cross-validation under this noise is running (v2; trained on mixed noise;
  without the touch features).

## 2026-10-29 — v2 under the partly calibrated noise (no change)

- **Setup.** Leave one match out as in plan steps 0-4, but every training and scored copy now
  comes from `degrade.calibrated()` (three copies per match) and the stress copy from
  `degrade.stressed()` of it. Variants: v2 as it stands; v2 trained on mixed copies; v2 without
  the touch features. Decoder settings chosen as before (mean over degraded and stress): v2
  kappa 0.3, gamma 1.5, unchanged.
- **Result** (spellF1 degraded [5-95%] / stress / clean):

  | | DFL | M1 | M2 |
  |---|---|---|---|
  | baseline_smooth50 | 0.169 / 0.103 / 0.309 | 0.174 / 0.106 / 0.329 | 0.166 / 0.098 / 0.343 |
  | v1 | 0.287 / 0.256 / 0.382 | 0.317 / 0.284 / 0.420 | 0.362 / 0.323 / 0.479 |
  | v2 | 0.677 [0.659, 0.694] / 0.573 / 0.706 | 0.716 [0.703, 0.728] / 0.611 / 0.738 | 0.714 [0.700, 0.728] / 0.627 / 0.713 |
  | v2, default noise (10-28) | 0.635 / 0.526 / 0.707 | 0.683 / 0.564 / 0.751 | 0.714 / 0.587 / 0.762 |

  macroF1 degraded 0.927. Predicted / true possessions on DFL 0.75 (0.69 under default noise).
  Paired against v2, degraded / stress:
  - without the touch features: DFL -0.029 [-0.037, -0.021] / +0.004, M1 -0.026
    [-0.034, -0.018] / +0.007, M2 -0.009 [-0.018, +0.000] / +0.005 (stress intervals contain 0);
  - trained on mixed copies: DFL -0.003 [-0.007, +0.002] / +0.015 [+0.009, +0.021], M1 -0.004 /
    +0.017, M2 -0.005 / +0.011 [+0.001, +0.020]; macroF1 +0.002 / +0.010.
- **Reading.** The calibrated copies are harder than the default ones for smooth50 and v1 and
  easier for v2 trained on them; a different test noise, so the rise over the 10-28 numbers is
  not a gain of the method. Which part of the noise does it is not shown under STRICT. v1 and v2
  fit parts of themselves on the training copies, hence their clean scores move too (v2 on M2
  clean -0.049: trained on noisier copies, it does a little worse on clean ones). The touch
  features now pay about twice what they did under the default noise; the question left open on
  10-29 (would they still pay at the ATD's false-detection rate) is answered yes. Mixed copies
  buy robustness under stress at no cost on the degraded copies, but no gain there either.
- **Decision.** Keep v2 as it is (touch features in; mixed copies not accepted by the rule of
  AGENTS.md 5.5, within noise on the degraded copies). Mixed copies stay the candidate if the
  ATD's dropout turns out far from what the simulator makes. `models/v2.json` not refitted yet:
  the refit on calibrated noise waits for the dropout calibration (the team's description), so
  that the model changes once.

## 2026-10-29 — a tracker that loses the ball in stoppages (simulator and calibration, STRICT)

- **Prompt.** The team compared the ATD's dropout statistics with the fitted copies' and shared
  them (COMPLIANCE_LOG.md). In generic terms, the ATD's dropout differs in kind: gaps are long,
  rarely a flicker, nearly all of the missing time in long gaps, with fewer players near the
  ball and a slower ball before it is lost than in the copies.
- **Off-game check of one explanation** (the tracker loses the ball while it is dead). On the
  DFL matches (dead share 0.41-0.52, 92-113 stoppages per match, mean 23-27 s), with the ball
  removed exactly while it is dead and nowhere else: missing 0.45, mean gap 25 s, flicker 0.004,
  long share 0.99, crowd ratio 0.66, speed ratio 0.95 (0.00 if lost a second after the ball goes
  dead: the harness's ball stands still while dead). The same pattern as the team's description.
- **Simulator** (`degrade.lost_in_stoppages`, off by default, default copies unchanged): in
  `ball_dead_lost` of the stoppages the ball is lost from `ball_dead_on_s` after it goes dead to
  `ball_dead_off_s` after the restart (spread `ball_dead_lag_sd_s` per stoppage), but for brief
  sightings at `ball_dead_seen_per_min` (mean `ball_dead_seen_s`); the dropout score takes
  `ball_missing` of the other frames. Every draw fixed per stoppage (common random numbers).
  `stressed()` doubles the delays' spread and the sightings.
- **Calibration.** New label-free statistics: the ball's speed just after it is found again
  (still if found before the restart); the players' mean speed (from positions, over 1 s) in
  the 2 s after vs before the start and the end of each gap of 5 s or more (they slow down after
  the ball goes dead and speed up after the restart). Off-game sensitivity, lost stoppages only:
  slowdown 0.73 / 0.73 / 0.76 / 0.81 / 0.87 for the ball lost -0.5 / 0 / +0.5 / +1 / +1.5 s after
  it goes dead; speedup 1.17 / 1.29 / 1.42 / 1.50 / 1.51 / 1.46 for found -1.5 ... +1 s after
  the restart. The five dropout knobs are fitted together by least squares on eight statistics
  (missing, gap, flicker, long share, ball speed before and after a gap, the two player ratios),
  from a tracker that loses every stoppage; a trial that leaves no gap to measure counts as far
  off.
- **Recovery check on DFL** (copies with known knobs, fitted from another seed): share of
  stoppages lost 0.85 -> 0.854, sightings 2.0 -> 1.94 per min, other dropout 0.010 -> 0.010, loss
  delay 0.40 -> 0.32 s, recovery delay -0.60 -> -0.60 s; all statistics within 5% but flicker
  (+11%, few gaps). Without the player ratios the loss delay came back at 1.09 s: the ball's
  speed alone cannot separate it from the sightings.
- **On the ATD:** reproduced within 10%: players (tracked, tracklets, jitter), missing share,
  flicker, long share, ball speed before a gap, players' slowdown, jumps. Not: mean gap, crowd
  ratio, ball speed after a gap, players' speed-up after a gap, ball jitter. Ball and player
  jitter at the bottom of their ranges: the ATD is smoother than DFL's raw tracking, more so now
  that the copies no longer keep the standing dead ball. The two statistics of the recovery
  disagree with each other, most likely because the harness's dead ball stands still where the
  ATD's is placed or carried; the fit is their compromise. When the ball is lost -- where a
  possession ends -- is reproduced; when it is found again is less certain.
- **Decision.** Use this calibration (it replaces the partial one). Cross-validation under it
  running (v2; trained on mixed copies, whose stress side widens the delays and the sightings).

## 2026-10-29 — v2 under the noise with lost stoppages (no change to the model; refit pending)

- **Setup.** As on 10-29 above (leave one match out, three calibrated copies per match, stress
  copy from `degrade.stressed()`), the copies now from the calibration with lost stoppages.
  Variants: v2 as it stands; v2 trained on mixed copies. Decoder settings chosen as before (mean
  over degraded and stress, DFL and M1): v2 kappa 0.3, gamma 1.0 (was 1.5); mixed kappa 0.6,
  gamma 1.5.
- **Result** (spellF1 degraded [5-95%] / stress / clean):

  | | DFL | M1 | M2 |
  |---|---|---|---|
  | baseline_smooth50 | 0.268 / 0.179 / 0.309 | 0.281 / 0.184 / 0.329 | 0.286 / 0.181 / 0.343 |
  | v1 | 0.393 / 0.275 / 0.378 | 0.430 / 0.294 / 0.416 | 0.483 / 0.321 / 0.473 |
  | v2 | 0.711 [0.694, 0.729] / 0.477 / 0.754 | 0.753 [0.739, 0.766] / 0.516 / 0.797 | 0.745 [0.731, 0.760] / 0.559 / 0.779 |
  | v2, mixed copies | 0.715 [0.696, 0.733] / 0.571 / 0.749 | 0.749 [0.736, 0.762] / 0.612 / 0.786 | 0.739 [0.724, 0.754] / 0.648 / 0.769 |

  macroF1 degraded 0.943 (both). Predicted / true possessions on DFL 0.77 (mixed 0.80).
  Paired, mixed minus v2, degraded / stress: DFL +0.004 [-0.001, +0.008] / +0.094 [+0.080,
  +0.106], M1 -0.003 [-0.009, +0.002] / +0.096, M2 -0.006 [-0.013, +0.002] / +0.089; macroF1
  -0.001 / +0.043.
- **Reading.** Again a different test noise, so the rise over the partly calibrated numbers is
  not a gain of the method. A ball lost in stoppages is easier for every method than one lost in
  crowds: the gaps fall where possessions end, not where they change. The stress copies are now
  much harder for v2 (0.477 on DFL, 0.573 under the partial calibration): they widen the delays
  and double the sightings in stoppages, the side of the dropout (when the ball is found again)
  that the calibration does not reproduce. Mixed copies cost nothing on the degraded copies and
  keep 0.09 of it under stress.
- **Decision.** By AGENTS.md 5.5 mixed copies are not accepted (within noise on the degraded
  copies). They hedge exactly the part of the dropout the calibration leaves uncertain, at no
  measured cost; raised with the team. v2 with or without the reception features (plan step 4b)
  is decided next on these same copies; `models/v2.json` is refitted once, after that.

## 2026-10-30 — plan step 4b: reception features (rejected)

- **Hypothesis.** Who gets the ball next decides a switch: a pass to a teammate keeps
  possession, an opponent who receives and holds the ball takes it. Features that look across
  the ball's flight to the next contact should cut wrong and missed switches.
- **Change** (`features.matrix`, group `train_v2.RECEPTION`). A contact: the ball seen within the
  touch radius of a team's player with no opponent nearer. Per frame: time since the last contact
  and until the next, the ball's travel between them; per team, whether it made the last / next
  contact and how firmly it held the ball in the 1 s before letting go / after receiving it
  (share of frames nearest). Unknown -> NaN. A synthetic pass and interception come out as
  intended (`tests/test_v2.py`).
- **Result.** Same copies as the entry above, same matches and seeds; fitting is
  deterministic, so v2 there is the paired reference. Decoder setting chosen as before: kappa
  0.45, gamma 1.0. Paired, with minus without, spellF1 degraded / stress:

  | | DFL | M1 | M2 |
  |---|---|---|---|
  | degraded | +0.009 [+0.004, +0.015] | +0.002 [-0.004, +0.009] | -0.004 [-0.010, +0.002] |
  | stress | -0.001 [-0.010, +0.008] | -0.005 [-0.015, +0.006] | -0.017 [-0.026, -0.008] |

  macroF1 degraded +0.001 / +0.000 / -0.000, stress -0.002 / -0.002 / -0.003. At v2's own
  setting (kappa 0.3, gamma 1.0) the degraded spellF1 is 0.711 / 0.752 / 0.747 with and
  0.711 / 0.753 / 0.745 without: DFL's gain comes with the other decoder setting, not from the
  features.
- **Reading.** Most likely the touch features and the second stage over ±8 s already carry what
  the next contact says: under the calibrated noise the ball is lost in stoppages rather than in
  flight, so the contacts on either side of a flight are usually seen, and the second stage
  reads them. Where the ball is found again late (stress), the new features cost a little.
- **Decision.** Rejected (AGENTS.md 5.5: beyond its interval on one label set only, within
  noise on the others, worse under stress on M2). The group stays in the code as an ablation;
  the accepted variant leaves it out (`drop=RECEPTION`). `models/v2.json` refitted: v2 as
  accepted, trained on three calibrated copies per match, kappa 0.3, gamma 1.0.

## 2026-10-30 — `models/v2.json` refitted on the calibrated copies

- **Change.** v2 as accepted (85 features, reception group left out), trained on all seven
  matches with three copies each from the calibration with lost stoppages; kappa 0.3, gamma 1.0
  (was 1.5). The previous model is kept outside the repo.
- **Checks:** tests pass; two runs of the one command give byte-identical files; the official
  validator accepts the file.
- **Unsupervised diagnostics** (our predictions only; before -> after): 284 -> 309 possessions,
  duration p10/p50/p90 2.0 / 7.1 / 29.5 s, 1.9% under 1 s, 42.3% dead; pre-kickoff stretch dead
  97.5% -> 97.2%. Agreement with Smart Tagging team tags 0.72 -> 0.69 (attacking phases), 0.86
  -> 0.84 (defending phases). Stability with the ball removed on about 10% more frames: 1.8% of
  graded frames change (was 1.1%), possessions x1.02, spellF1 against the original 0.94 (was
  0.97). No sign of a bug. The lower stability is what the training noise predicts: the
  perturbation removes the ball in play, which the calibrated copies rarely do. These numbers
  catch bugs; they do not rank the two models. Public window not looked at.

## 2026-10-30 — mixed training copies adopted (team decision, exception to AGENTS.md 5.5)

- **Decision by the team.** Train v2 on mixed copies (three calibrated, three drawn between the
  calibrated and the stress noise), although they do not pass the rule of AGENTS.md 5.5: on the
  calibrated copies they are within noise (DFL +0.004 [-0.001, +0.008], M1 -0.003, M2 -0.006),
  under stress +0.09 on every label set (entry "v2 under the noise with lost stoppages"). The
  stress copies widen the delays and the sightings in stoppages, the side of the dropout the
  calibration does not reproduce on the ATD, so the mixed copies hedge the known gap of the
  simulator at no measured cost.
- **Change.** `train_v2.VARIANTS`: `Variant(stacked=True, copies="mixed", drop=RECEPTION)`;
  `models/v2.json` refitted at the setting the cross-validation chose for it, kappa 0.6, gamma
  1.5. Pipeline version v2.2.
- **Checks:** tests pass; two runs of the one command byte-identical; the official validator
  accepts the file. After the commit (c1d1d6a), `scripts/check_reproducibility.sh`: two
  clean-room runs (fresh export, fresh virtualenv, empty cache) byte-identical to each other and
  to the file of the working copy.
- **Unsupervised diagnostics** (our predictions only; the plain refit of the entry above ->
  mixed): 309 -> 322 possessions, duration p10/p50/p90 1.9 / 6.2 / 29.1 s, under 1 s 1.9% ->
  3.7%, 41.5% dead; pre-kickoff stretch dead 97.6%. Agreement with Smart Tagging team tags 0.71
  (attacking phases), 0.85 (defending phases). Stability with the ball removed on about 10% more
  frames: 1.3% of graded frames change, possessions x1.03, spellF1 against the original 0.94.
  More and shorter possessions, as on the harness (predicted / true 0.80 against 0.77 on DFL).
  No sign of a bug. Public window not looked at.
