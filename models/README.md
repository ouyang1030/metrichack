# Models

Small trained weights, from off-game data only, each with the script that trains it.

| file | trained by | data |
|---|---|---|
| `v2.json` | `python -m harness.train_v2` | IDSSE / DFL open data, 7 matches, six degraded copies of each (three with the noise calibrated to the ATD, three drawn between it and a much noisier one), every 5th in-play frame of each plus its team-swapped copy (DATASETS.md) |
| `v1.json` | `python -m harness.train_v1` | IDSSE / DFL open data, 7 matches, degraded seed 0 (DATASETS.md) |

`v2.json` (the pipeline's) holds the feature names it was fitted on (`features.matrix`), three
gradient-boosted classifiers exported from scikit-learn (per classifier, `baseline` raw scores
and, per iteration and class, one tree as flat arrays; `mtah/trees.py` evaluates them): the
first stage at the top level, the change detector under `detector`, the second stage under
`second` (its inputs: `features.second_stage`). Then class priors, the H/A/D transition
probabilities and mean durations, the change window, and the decoder settings `kappa` and
`gamma`.

`v1.json` (the reference v2 is measured against) holds two logistic models (P(dead | x);
P(home | alive, x) on the terms listed in `team_terms`), class priors, the H/A/D transition probabilities and mean durations, the decoder
settings (`chain`, `refine`) and the temperatures. Nothing in it comes from BAR-ATH.
