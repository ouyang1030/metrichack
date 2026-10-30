# Compliance log

Every look at the public 5-minute window, every external dataset, every rule question
(AGENTS.md section 9). `scripts/smoke_test_public.py` appends a line to the last section for each
scorer run; add underneath, by hand, what was (not) changed because of it.

## Data handling

- 2026-10-27 — v0 scaffold. The coding agent (Claude Code, a hosted LLM) read only the official
  repository's documentation and code (commit `f8f2629`). The data bundle was not on the machine.
  The agent did not open `evaluation/gt/*.csv`, the four `baseline_*.csv` files or the official
  `data/README.md`; the CSVs were copied with `cp`. Tests and dry runs used synthetic bundles
  only (`tests/synthetic_bundle.py`, random positions).

- 2026-10-29 — The team put the data bundle in `data/` (tactical and broadcast ATD, homography,
  Smart Tagging; no videos). The agent never listed or opened anything under `data/`, `cache/`
  or `debug/`: a listing of `data/` was refused by the permission guard, and the layout was
  settled by the team against the paths `games.py` documents. Every script that read the data
  ran behind `mtah/safe_run.py`, and the agent saw only: pass/fail checks (parser against the
  four published baselines, byte for byte; two runs of the one command byte-identical; the
  official validator; Smart Tagging kickoff windows containing the published kickoffs; the team
  names mapping onto home and away), runtimes and memory, and summaries of our own predictions
  (`scripts/diagnose.py`: possessions, durations, dead share, the share of the pre-kickoff
  stretch predicted dead, agreement with Smart Tagging team tags, stability when 10% more ball
  frames are removed).
- 2026-10-29 — Simulator calibration under STRICT (`scripts/calibrate_noise.py`): the noise
  statistics of the ATD and the simulator knobs fitted to them were written to
  `cache/aggregates/` and not shown; the agent saw per knob only whether the ATD's statistic
  lay within the knob's search range, and per statistic whether the fit reproduced it within
  10%. The harness now trains and scores on the calibrated noise; the knobs are never printed.
- 2026-10-29 — The second half is graded from before its kickoff (DATA.md); the pipeline now
  marks graded frames before a half's Smart Tagging "Kick Off (Start)" window as dead. Prompted
  by the unsupervised diagnostic (only 31% of that stretch was predicted dead), not by any
  ground truth; no frame number of ours enters the code.
- 2026-10-29 — Disclosure by the team, under STRICT. The calibration could not reproduce the
  ATD's five ball-dropout statistics, and the agent asked the team to compare
  `atd_noise.json` with `fitted_noise.json` in their own terminal and to describe the
  difference in direction and rough size only. The team pasted the five statistics of both into
  the session instead: ten numbers, five of them aggregates of the ATD's graded frames (mean gap,
  flicker share, long-gap share, crowd and speed ratios), nothing frame-indexed. That goes beyond
  STRICT, which the team keeps until the organisers confirm aggregates
  (`docs/ORGANISER_QUESTIONS.md`); section 4 leaves what to share to the team. Use: only to
  choose the structure of the simulator's dropout model (a tracker that loses the ball in
  stoppages). No number from them enters code, tests or docs; the knobs are still fitted by
  `scripts/calibrate_noise.py` from the JSON and never printed.

## External datasets

Each one goes in `DATASETS.md` with source, licence and who labelled it.

- 2026-10-27 — IDSSE / DFL open data (7 Bundesliga matches, CC BY 4.0), downloaded from figshare
  into `external/idsse/` (gitignored). Other matches, labelled by DFL; nothing about BAR-ATH.
  Inspected normally (AGENTS.md section 4 allows it for other games).
- 2026-10-27 — Harness sanity check: the four baselines' spellF1 on the degraded harness was set
  beside the full-match scores the organisers publish (OVERVIEW.md, starter/baseline.py). Those
  published scores are only a comparison point: OVERVIEW.md withholds full-match ground-truth
  statistics precisely so that nothing is tuned against them, so no simulator parameter is set
  from them. The simulator is calibrated from ATD aggregates under STRICT only.

## Rule questions

Drafted in `docs/ORGANISER_QUESTIONS.md` (status: drafted, not yet posted).

## Public-window looks

- 2026-10-27 `smoke_test_public` on `baseline_raw.csv`, `baseline_smooth25.csv`, `baseline_smooth50.csv`, `baseline_smooth125.csv` -- reason: v0 scorer wiring check: reproduce the README's published public-window board for the four official baselines; no file of ours scored. Alarm only; not used to choose between variants or to set anything.
  - Board identical to the one published in README.md; nothing was changed because of it. Only
    the official files were scored.
- 2026-10-29 `smoke_test_public` on `TEAM_NAME.csv`, `baseline_smooth50.csv` -- reason: v2 alarm after the first run on the real bundle: alignment, label swap, half boundaries; smooth50 beside it as the published reference. Alarm only; not used to choose between variants or to set anything.
  - Only the ranked table was seen (per-file reports with ground-truth transition frames went to
    `cache/eval/`, unread). No alarm: no sign of misalignment, a label swap or a broken half
    boundary. Only `models/v2.json` was scored, no other variant, and nothing was changed because
    of it.
