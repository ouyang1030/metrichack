# v2 plan (started early on 2026-10-27; AGENTS.md calls for 11-03 – 11-09)

Learned emissions, AGENTS.md section 7. The decision rule is unchanged: degraded off-game
harness, paired bootstrap; the stress copy as a second, never-trained-on noise level.

- [x] Frame classifier: gradient-boosted trees over 66 team-symmetric features with ±2 s
  context, trained with team-swapped copies, exported to JSON, evaluated in numpy. Accepted:
  +0.218 spellF1 over v1 (EXPERIMENTS.md).
- [x] Diagnosis (EXPERIMENTS.md): v2 under-segments, it never over-segments (splits 0%).
  Misses concentrate in true possessions under 2 s (30% of all; 8% found, 59% missed
  outright) and in merges (21-26% of possessions up to 10 s), and a merge is almost always a
  short spell of the *other* team between two spells of one team (under 1 s: 33% reproduced).
  Stoppages are long and found (97-100% of those over 2 s). Oracle boundaries with v2's labels:
  spellF1 0.696 vs 0.593 decoded, so the remaining gain is in the segmentation.

### Chosen plan (docs/LITERATURE.md), in order; each step one experiment, accepted on the harness

0. [x] **Label semantics first.** Metrica: a touch without control, a deflection or a lost
   aerial duel does not change possession. DFL's sub-second spells of the other team are
   probably just that. Add a second label set to the harness that follows Metrica's written rule
   (a spell of the other team shorter than a control time, between two spells of one team,
   goes to that team; control time 1 s from football logic, 2 s as a check) and accept a change
   only if it does not lose on either label set. Ask the organisers (docs/ORGANISER_QUESTIONS.md,
   question 6). Until then, do not chase sub-second spells.
   *Done:* `harness/semantics.py`; every result is now scored on DFL, M1 and M2. Training on M1
   changes nothing (-0.003); kappa is now chosen on DFL and M1; a stricter rule favours a lower
   kappa.
1. [x] **Noise randomisation in training** (Tobin 2017): each training copy draws its simulator
   parameters between the defaults and the stress copy. Expected: stress score up, degraded the
   same, kappa less fragile.
   *Done:* three default copies per match accepted (+0.006). Randomised noise trades -0.000 to
   -0.010 at the default noise for +0.05 under stress: deferred to the calibration.
2. [x] **Switch-aware decoder** (Bourlard & Morgan; Bengio & Frasconi; PathCRF; ASRF).
   a. A switch penalty separate from kappa (2-D grid): today's 0.65 predicted / true
      possessions says the decoder is too reluctant, and raising kappa instead hurts under
      heavy noise.
   b. A second classifier, P(possession or play state changes within ±0.2 s | frame), turned
      into input-dependent transition probabilities (IOHMM). Targets the merges and the
      boundaries at once.
   *Done:* (a) rejected, within +-0.003 (a constant switch cost duplicates kappa). (b) accepted:
   +0.002 to +0.006 degraded, +0.009 to +0.017 stress, gamma 1, kappa 0.2.
3. [x] **Stacked temporal context** (auto-context; MS-TCN; Lang 2022): stage 2 over stage-1
   out-of-fold probabilities and switch scores in windows up to ±8 s, plus a few raw features.
   *Done:* accepted, +0.012 to +0.015 degraded, +0.03 stress (all 66 features kept as inputs).
4. [x] **Touch and duel features** (Vidal-Codina 2022; Link & Hoernig): change of ball
   direction or speed within 1 m of a player, two opponents within 1 m of the ball, time the
   same team has been nearest. These are what tells control from a deflection, i.e. Metrica's
   rule. *Done:* accepted, +0.010 to +0.014 degraded, 0 under stress, +0.05 clean: the gain
   depends on how well the ball is located.
5. [ ] Only if 1-4 level off: a neural sequence model (TCN or set transformer + Bi-LSTM,
   exported to numpy). Seven labelled matches are few for it; would need more data (e.g. PFF
   World Cup 2022, registration required). *Deferred* after the diagnosis of 10-29: the
   decoder finds the short spells the evidence supports, the evidence is what is missing, and
   the trees' learning curve is flat (+0.008 per doubling of matches). First:
   4b. [ ] **Reception and interception features**: where a ball flight ends and with whom
   (the team of the player the ball decelerates into after a flight, the team that played it
   before), the information the short spells lack, at a fraction of step 5's cost.

Not planned: player-level possession paths (PathCRF needs identities; the ATD's 46 slots have
none), explicit duration models (tried, +0.002; the input-dependent transitions are the better
form), video.

- [ ] Ablations for the document: without the look-ahead windows, without team swapping,
  trained on clean instead of degraded.
- [ ] Smart Tagging fused at decoding time with a fixed, documented weight (needs the data).
- [ ] Simulator calibration from ATD aggregates (STRICT), then re-choose kappa and retrain
  (needs the data). Includes the false-detection rate (`degrade.Params.ball_false`, off by
  default), from aggregates such as the share of in-play frames where the ball jumps faster
  than a ball flies (over ~40 m/s between frames) and how long such runs last; then decide
  between default and randomised training noise (step 1).
- [ ] BAR-ATH diagnostics of v2 (possession count, durations, flicker, Smart Tagging
  agreement, stability when more ball frames are dropped).
