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

## External datasets

None yet. Each one goes in `DATASETS.md` with source, licence and who labelled it.

## Rule questions

Drafted in `docs/ORGANISER_QUESTIONS.md` (status: drafted, not yet posted).

## Public-window looks

- 2026-10-27 `smoke_test_public` on `baseline_raw.csv`, `baseline_smooth25.csv`, `baseline_smooth50.csv`, `baseline_smooth125.csv` -- reason: v0 scorer wiring check: reproduce the README's published public-window board for the four official baselines; no file of ours scored. Alarm only; not used to choose between variants or to set anything.
  - Board identical to the one published in README.md; nothing was changed because of it. Only
    the official files were scored.
