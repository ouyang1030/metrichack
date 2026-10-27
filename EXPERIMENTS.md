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
