# v1 plan (2026-10-29 – 11-02)

AGENTS.md sections 5 and 7. Every decision is taken on the off-game harness (degraded), never on
the public window. One experiment = one hypothesis = one entry in EXPERIMENTS.md.

## 1. Off-game harness — `harness/`

- [x] **IDSSE / DFL open data** (7 Bundesliga matches, CC BY 4.0, figshare 10.6084/m9.figshare.28196177).
  TRACAB tracking at 25 Hz with per-frame `BallPossession` (1 home, 2 away) and `BallStatus`
  (0 dead, 1 alive) — the closest analogue to how our ground truth was made.
  `harness/idsse.py`: DFL XML -> the BAR-ATH array format (players (n, slots, 3) normalised x, y +
  speed m/s, ball (n, 3), slot_team, labels H/A/D, X between halves). Raw files and parsed
  arrays live in `external/` (gitignored).
- [x] `harness/score.py`: run a method over every match, score each with the official
  `possession_metrics.evaluate()`, report mean spellF1 / macroF1 with a bootstrap interval over
  match halves.
- [x] Sanity check: the four baselines on the harness. Expect the BAR-ATH pattern (raw flickers,
  spellF1 peaks at a middle window). Not used to calibrate anything.

## 2. ATD noise simulator — `harness/degrade.py`

- [x] Ball dropout to ~41% of graded frames, more likely when the ball is high (z), fast, or in a
  crowd; drop in runs, not single frames.
- [x] Tracklets: fragment each player track into short pieces spread over 46 slots, identity swaps
  between nearby players, missing players, positional jitter. (Duplicated players: not yet.)
- [x] Defaults from numbers DATA.md publishes (ball present 59.3% of graded frames, 20.5 tracked
  objects on average, 46 slots). A firewalled calibration script later writes ATD aggregates to
  `cache/aggregates/` (STRICT) and the simulator reads them from there. **Open: needs the data.**
  Fit the simulator so that its own aggregates match the ATD's, entirely behind the firewall.
- [x] Every result reported on clean and degraded.

## 3. Method v1 — `mtah/`

- [x] Dead/alive evidence, first cut (`features.py`, `emissions.py`): collective speed, ball last
  seen beyond a line, time since seen -> P(dead). Alive/dead F1 ~0.9 on the degraded harness.
- [ ] More dead-ball evidence: ball stationary at a restart spot, kickoff formation, goals.
- [x] Possession evidence, round 1: team movement (`flow`) accepted, +0.022; ball-player
  co-movement (`control`) and team spread rejected. Still too few switches on short spells.
- [ ] Possession evidence, round 2 (ideas): (player of team T near the ball for several frames, ball
  velocity ~ player velocity or decelerating into the player; flights stay with the passer's team;
  long gaps from team shape) -> per-frame P(H), P(A).
- [x] `decoder.py`: HMM over {H, A, D}, Viterbi path, tempered evidence; unit tests ("a
  deflection does not switch", "H H D D H H is two spells"). Accepted: +0.080 spellF1 over
  smooth50 on the degraded harness.
- [x] Duration chains (k = 2, 3) and boundary refinement tried: +0.002 and +0.003, within noise,
  rejected (code kept, switched off) — retry after calibration.
- [ ] Smart Tagging as soft evidence at decoding time (fixed, documented weight).
- [ ] Ablations: with/without decoder, with/without Smart Tagging, clean vs degraded.

## 4. BAR-ATH diagnostics (no labels)

- [ ] Possession count and duration distribution vs the harness priors, flicker rate, agreement
  with Smart Tagging team tags, stability when 10% more ball frames are dropped.
- [ ] One smoke test on the public window after v1 lands (alarm only, logged).

## Waiting on the team

- Real data in `./data` to verify v0 on BAR-ATH and to run the calibration script.
- Organiser answers (docs/ORGANISER_QUESTIONS.md).
