# Off-game harness

The decision tool (AGENTS.md section 5): other matches in the BAR-ATH array format, made ATD-like
by a noise simulator, scored with the official `evaluation/possession_metrics.evaluate()`.

| file | what |
|---|---|
| `idsse.py` | IDSSE / DFL open data (7 matches) -> arrays + H/A/D labels; `python -m harness.idsse` |
| `degrade.py` | ATD noise simulator: ball dropout (rough and smooth noise, crowd, speed, height, off the pitch, dead ball), stoppages in which the ball is lost (when lost and found again, brief sightings), false ball detections, tracklets over 46 slots, swaps, jitter. `Params()` are guesses; `calibrated()` reads the knobs fitted to the ATD; `stressed()` makes much noisier ones |
| `calibrate.py` | noise statistics that need no labels (ball missing share, gap lengths, flicker, long losses, crowd and speed lifts, the ball's speed when found again, the players' slowdown and speed-up around long gaps, jumps, jitter; tracked players, tracklet length, jitter), and the fit of the simulator's knobs to them. Run through `scripts/calibrate_noise.py`, behind the firewall: under STRICT the ATD's statistics and the fitted knobs are written to `cache/aggregates/` and never shown |
| `score.py` | per-half official scores, bootstrap intervals; `python -m harness.score` |
| `train_v1.py` | leave-one-match-out cross-validation of v1, then fit `models/v1.json`; `python -m harness.train_v1` |
| `semantics.py` | the labels under Metrica's possession rule (a touch without control does not change possession): label sets M1 / M2 next to DFL's own |
| `train_v2.py` | the same for v2 (first stage, change detector, second stage), against v1 and smooth50, on clean, degraded and a stress copy, scored on every label set; fits `models/v2.json`; `python -m harness.train_v2` (needs the calibration and `requirements-dev.txt`; about an hour, one job at a time: two training jobs on the same cores slow each other down many times over) |

Raw downloads and parsed arrays live in `external/idsse/` (gitignored). To fetch them, download
every file of figshare article 28196177 into that folder, then run `python -m harness.idsse`.
