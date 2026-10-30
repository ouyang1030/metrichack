# Literature behind the method (2026-10-27)

What each line of work tells us, and what we take from it. Read for methods only; nothing here
is data about the provided match.

| work | what it does | what we take |
|---|---|---|
| Bassek et al. 2025, *Sci Data* 12:195 (IDSSE / Sportec open DFL data) | 7 Bundesliga matches, TRACAB tracking; `BallPossession` "defined by the player in control of the ball", `BallStatus` inactive "when the match is interrupted by the referee" | our harness; DFL's notion of control may still count touches Metrica does not (see diagnosis in EXPERIMENTS.md) |
| Link & Hoernig 2017 / Hoernig et al. 2016, individual ball possession | possession per player from ball-player distance and ball dynamics, several "levels" of control; 80-92% precision/recall | control is graded, not binary: a short touch is weak evidence |
| Vidal-Codina et al. 2022, *Sports Eng.* (arXiv 2202.00804) | rules on tracking: possession zone 0.5-1 m, duel zone 1 m with two opponents (duel wins over possession), a gain needs a change of ball direction or speed at the player, a loss needs the ball to leave the zone and travel | touch / duel / gain features that separate control from deflection |
| Lang et al. 2022, *Sci Rep* | in play vs stopped from players only; ±80 s of shifted features, AdaBoost; frame accuracy 92%, stoppage F1 0.80 (IoU 0.5); shifts within ±4 s matter most; teams swapped in training | longer context than our ±2 s; our team swap is standard practice |
| Borghesi et al. 2023, *ESWA* | TCN with a permutation-invariant layer for possession from tracking; 86% accuracy, beats rules | a neural sequence model is the next step up, not the first |
| Kim et al. 2023, KDD (ballradar) | set transformer + Bi-LSTM infer the ball and its possessor from players only; trained on Metrica sample data among others | team shape carries the ball's whereabouts; our team features do this cheaply |
| Capellera et al. 2024, ACCV (TranSPORTmer) | per-frame state (pass / possession / uncontrolled / out of play) with the ball partly or fully masked; ~81% with 90% of the ball hidden; no temporal post-processing | masking the ball in training = our dropout simulator |
| Kim et al. 2026, arXiv 2602.12080 (PathCRF) | ball-free event detection on the same DFL data as a CRF over possession edges; frame-wise argmax over-segments (event F1 65%); constrained Viterbi 73%; CRF with input-dependent transitions 76% | decoding with constraints matters, and transitions that depend on the input beat a fixed matrix |
| Bengio & Frasconi 1995 (IOHMM) | HMM whose transition probabilities depend on the input sequence | the form for a switch-aware decoder |
| Bourlard & Morgan 1994 (hybrid HMM / neural nets) | classifier posteriors divided by class priors = scaled likelihoods for an HMM; in practice scaled, with a separate insertion penalty | our kappa is that scale; a switch penalty is a second, separate knob |
| Ishikawa et al. 2021, WACV (ASRF); Farha & Gall 2019 (MS-TCN) | temporal action segmentation: a boundary branch, multi-stage refinement and a smoothing loss against over-segmentation | a boundary detector next to the frame classifier; stacked refinement |
| Tu 2008, CVPR (auto-context); Cohen & Carvalho 2005, IJCAI (stacked sequential learning) | feed a classifier's own predictions over a neighbourhood back in as features, retrain | stage 2 over stage-1 probabilities in long windows, with out-of-fold training |
| Tobin et al. 2017, IROS (domain randomization) | train on widely randomised simulations so the real domain is one more variation | randomise the noise simulator's parameters in training: the ATD's noise level is unknown |
| Chen et al. 2020, CVPR (SSTDA) | unsupervised domain adaptation for action segmentation, aligning features of labelled source and unlabelled target | later, once the data is here: adaptation to BAR-ATH without labels |
