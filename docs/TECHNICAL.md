# Technical document — draft

Skeleton for the deliverable SUBMISSION.md describes. Filled in as versions land; final by
2026-11-14.

## Approach

v0: the published floor, `baseline_smooth50` — the team of the tracked player nearest the ball,
majority vote over ±50 frames — computed from a parsed cache by a vectorised reimplementation
that is byte-identical to `starter/baseline.py`. Planned: segmentation into alive/dead and team
spells with a duration-aware decoder (AGENTS.md section 7).

## Assets used and dropped

| asset | status |
|---|---|
| tactical ATD | used (v0) |
| Smart Tagging | parsed and cached; not used in v0 |
| tactical homography | parsed and cached; not used |
| tactical video, broadcast feed | not used |

## Training data

None in v0. Any future dataset is listed in DATASETS.md: source, labeller, size. None of it comes
from the provided match.

## Possession rules implemented

v0: none beyond the nearest player and smoothing. No dead-ball model: a frame is `D` only when no
frame in its ±50-frame window has a located ball with a tracked player.

## Limitations

v0 is the baseline. See EXPERIMENTS.md.

## Reproduction

See README.md: one command, `python scripts/make_submission.py`; two clean-room runs must give
byte-identical files (`scripts/check_reproducibility.sh`).
