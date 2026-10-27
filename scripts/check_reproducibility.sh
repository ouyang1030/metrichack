#!/usr/bin/env bash
# Clean-room reproduction (AGENTS.md section 2.7): twice, in a fresh export of HEAD with a fresh
# virtualenv and an empty cache, run the one command; the two files must be byte-identical, and
# identical to an expected file if one is given.
#
#   MTAH_DATA_DIR=/path/to/bundle scripts/check_reproducibility.sh [expected.csv]
#
# The temporary copies hold a parsed cache of the match, so they are deleted on exit. cmp runs
# silently: where two prediction files differ is per-frame information.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${MTAH_DATA_DIR:-$ROOT/data}"
TEAM="$(sed -n 's/^TEAM_NAME = "\(.*\)"$/\1/p' "$ROOT/mtah/config.py")"
EXPECTED="${1:-}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

for run in 1 2; do
  dest="$WORK/run$run"
  mkdir -p "$dest"
  git -C "$ROOT" archive HEAD | tar -x -C "$dest"
  uv venv -q --python "$(cat "$ROOT/.python-version")" "$dest/.venv"
  uv pip install -q --python "$dest/.venv/bin/python" -r "$dest/requirements.txt"
  (cd "$dest" && MTAH_DATA_DIR="$DATA" MTAH_CACHE_DIR="$dest/cache" MTAH_DEBUG_DIR="$dest/debug" \
    .venv/bin/python scripts/make_submission.py)
done

FILE="evaluation/results/$TEAM.csv"
status=0
if cmp -s "$WORK/run1/$FILE" "$WORK/run2/$FILE"; then
  echo "PASS  two clean-room runs are byte-identical"
else
  echo "FAIL  the two clean-room runs differ"; status=1
fi
if [ -n "$EXPECTED" ]; then
  if cmp -s "$WORK/run1/$FILE" "$EXPECTED"; then echo "PASS  identical to $EXPECTED"
  else echo "FAIL  differs from $EXPECTED"; status=1; fi
fi
exit $status
