# More than a Hack 2027, Phase 1: ball possession

Our entry to [More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027).
For every frame of the BAR-ATH tactical video it writes `H`, `A`, `D` or `X`.
Project rules for contributors and the coding agent: [AGENTS.md](AGENTS.md).

**Status: v2.** Gradient-boosted classifiers give, for every frame, P(H, A, D) and the
probability of a change of possession or play state from 85 tracking features; a second stage
looks at how both move over ±8 s; an input-output HMM decodes them into piecewise-constant
spells. Trained on off-game matches only (`models/v2.json`). On the off-game harness, under
noise calibrated to the ATD: spellF1 0.721 [0.706, 0.736], +0.328 over v1 and +0.453 over the
published floor `baseline_smooth50` (EXPERIMENTS.md, 11-01; DFL labels; v2.3).

## Install

Python 3.11 (`.python-version`). Every dependency is pinned exactly.

```bash
git clone https://github.com/ouyang1030/metrichack && cd metrichack
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt   # or: uv venv && uv pip install -r requirements.txt
.venv/bin/pip install -r requirements-dev.txt                            # tests and harness training
scripts/install_hooks.sh                                                 # once, after git init / clone
```

## Data

The match data is gated and is **not** in this repository. Unpack the organisers' bundle into
`./data/` with the names it ships with (`data/tactical/BAR-ATH_tactical_atd.txt`, …), or point
`MTAH_DATA_DIR` to it. Never commit it: `.gitignore` and the pre-commit hook both block it.

Off-game data (DATASETS.md), only for the harness, all under the gitignored `external/`:
`external/idsse/` (IDSSE / DFL, figshare), `external/metrica/` (the files of
github.com/metrica-sports/sample-data `data/Sample_Game_{1,2,3}/`), `external/skillcorner/`
(the `data/matches/` folder of github.com/SkillCorner/opendata, or a link to it).

## The one command

```bash
.venv/bin/python scripts/make_submission.py        # -> evaluation/results/kicker.csv
```

The command parses the bundle into `cache/` (first run only; `--rebuild-cache` after a new
bundle), runs the pipeline, writes the file and runs the official validator on it.

Two runs must give byte-identical files (`scripts/check_reproducibility.sh`). The parser is
checked separately: `scripts/reproduce_baselines.py` must rebuild the four published baselines
byte for byte.

Other entry points:

| command | what it does |
|---|---|
| `scripts/build_cache.py [--source all] [--force]` | parse the bundle into `cache/` |
| `scripts/reproduce_baselines.py` | rebuild the four published baselines; compare them byte for byte with the committed files and with a fresh run of `starter/baseline.py` |
| `scripts/check_reproducibility.sh [expected.csv]` | two clean-room runs of HEAD (fresh export, fresh venv, empty cache); the files must be byte-identical |
| `scripts/smoke_test_public.py --reason "…" <csv>` | score against the public 5-minute window, as an alarm only; logs to COMPLIANCE_LOG.md |
| `scripts/diagnose.py` | unsupervised diagnostics of our predictions on BAR-ATH: possessions and durations, the pre-kickoff stretch, agreement with Smart Tagging team tags, stability when more ball frames are removed |
| `scripts/calibrate_noise.py` | fit the noise simulator to the ATD under STRICT (statistics and knobs go to `cache/aggregates/`, unseen); needed before `harness.train_v2` |
| `python -m harness.idsse` / `harness.score` | off-game harness: parse IDSSE, score the baselines |
| `python -m harness.train_v2` | (after `calibrate_noise.py`) cross-validate v2 against v1 and smooth50 (clean, degraded, stress; DFL's labels and Metrica's rule) and the ablations, save the out-of-fold outputs to `external/idsse/oof/`, then write `models/v2.json` (several hours; `harness.train_v1`: the same for v1) |
| `scripts/calibrate_tagger.py` | (after `harness.train_v2`) fit the simulated phase tagger to Smart Tagging's vote statistics under STRICT (to `cache/aggregates/`, unseen); needed before `harness.fuse` |
| `python -m harness.fuse` | (after `calibrate_tagger.py`) Smart Tagging's weight in the decoder, with the simulated tagger on the out-of-fold outputs |
| `python -m harness.segments` | (after `harness.train_v2`) where the decoded possessions go wrong: found, merged, split, missed, by length |
| `python -m harness.recovery` | does the calibration recover known simulator knobs on DFL |
| `python -m harness.metrica` / `harness.skillcorner` / `harness.external` | parse Metrica's sample games and SkillCorner's open data, then score the exported models on them |
| `.venv/bin/python -m pytest` | tests (synthetic data only) |

## Data firewall

Rules from AGENTS.md section 4. Any script that reads the match data runs through
`mtah/safe_run.py`. The script's console output goes to `debug/logs/`, which only humans read. The
terminal shows only approved lines: pass/fail checks, paths, the official validator's output and
summaries of our own predictions. An exception prints its type, a `file:line` and its
message with the digits masked. Input aggregates are written to `cache/aggregates/` and are not
printed (`FIREWALL_MODE = "STRICT"` in `mtah/config.py`).

## Layout

```
games.py, evaluation/, starter/   official files, copied unchanged (LICENSE-METRICA)
mtah/                             our package
  config.py                       paths, team name, firewall mode
  safe_run.py                     the firewall
  atd.py                          FIFA/EPTS metadata + raw tracking parser (spec-driven)
  cache.py                        parse once into cache/, load arrays from it
  baseline.py                     starter/baseline.py, vectorised and bit-exact
  features.py                     per-frame tracking features (v1: compute, v2: matrix)
  trees.py                        gradient-boosted trees evaluated in numpy from JSON (v2)
  emissions.py                    v1 logistic evidence; tempered classifier evidence (v2)
  decoder.py, refine.py           HMM / Viterbi over H, A, D; boundary refinement (off)
  tagging.py                      Smart Tagging phases as decoder evidence
  pipeline.py                     cache -> labels (v2)
  submission.py, frames.py        write/validate/summarise a file; graded frames
scripts/                          entry points; hooks/ holds the pre-commit data guard
harness/                          off-game harness: IDSSE, Metrica and SkillCorner loaders, ATD
                                  noise simulator and its calibration, training, scoring
models/                           weights fitted on off-game data only (v2.json; v1.json)
tests/                            unit tests and the synthetic-bundle generator
docs/                             technical document, plans, literature, organiser questions
EXPERIMENTS.md, COMPLIANCE_LOG.md, DATASETS.md
data/ cache/ debug/ external/      gitignored, never committed (external/: off-game downloads)
```

## Licence

Files under `games.py`, `evaluation/` and `starter/` are © Metrica Sports, MIT (see
`LICENSE-METRICA`), copied unchanged from the official repository at commit
`f8f2629ff1e0985f81d7e59d8413678514ee1530`. The match data is covered by neither licence.
