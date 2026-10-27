# More than a Hack 2027, Phase 1: ball possession

Our entry to [More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027).
For every frame of the BAR-ATH tactical video it writes `H`, `A`, `D` or `X`.
Project rules for contributors and the coding agent: [AGENTS.md](AGENTS.md).

**Status: v0.** The pipeline reproduces the published floor `baseline_smooth50` byte for byte.
Our own method replaces `mtah/pipeline.py` from v1 on.

## Install

Python 3.11 (`.python-version`). Every dependency is pinned exactly.

```bash
git clone https://github.com/ouyang1030/metrichack && cd metrichack
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt   # or: uv venv && uv pip install -r requirements.txt
.venv/bin/pip install -r requirements-dev.txt                            # tests only
scripts/install_hooks.sh                                                 # once, after git init / clone
```

## Data

The match data is gated and is **not** in this repository. Unpack the organisers' bundle into
`./data/` with the names it ships with (`data/tactical/BAR-ATH_tactical_atd.txt`, …), or point
`MTAH_DATA_DIR` to it. Never commit it: `.gitignore` and the pre-commit hook both block it.

## The one command

```bash
.venv/bin/python scripts/make_submission.py        # -> evaluation/results/TEAM_NAME.csv
```

The command parses the bundle into `cache/` (first run only; `--rebuild-cache` after a new
bundle), runs the pipeline, writes the file and runs the official validator on it.

**Expected output (v0):** byte-identical to the committed `evaluation/results/baseline_smooth50.csv`:

```bash
cmp evaluation/results/TEAM_NAME.csv evaluation/results/baseline_smooth50.csv && echo identical
```

Other entry points:

| command | what it does |
|---|---|
| `scripts/build_cache.py [--source all] [--force]` | parse the bundle into `cache/` |
| `scripts/reproduce_baselines.py` | rebuild the four published baselines; compare them byte for byte with the committed files and with a fresh run of `starter/baseline.py` |
| `scripts/check_reproducibility.sh [expected.csv]` | two clean-room runs of HEAD (fresh export, fresh venv, empty cache); the files must be byte-identical |
| `scripts/smoke_test_public.py --reason "…" <csv>` | score against the public 5-minute window, as an alarm only; logs to COMPLIANCE_LOG.md |
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
  pipeline.py                     cache -> labels (v0: baseline_smooth50)
  submission.py, frames.py        write/validate/summarise a file; graded frames
scripts/                          entry points; hooks/ holds the pre-commit data guard
harness/, models/                 off-game harness and trained weights (from v1)
tests/                            unit tests and the synthetic-bundle generator
docs/                             technical document, questions for the organisers
EXPERIMENTS.md, COMPLIANCE_LOG.md, DATASETS.md
data/ cache/ debug/               gitignored, never committed
```

## Licence

Files under `games.py`, `evaluation/` and `starter/` are © Metrica Sports, MIT (see
`LICENSE-METRICA`), copied unchanged from the official repository at commit
`f8f2629ff1e0985f81d7e59d8413678514ee1530`. The match data is covered by neither licence.
