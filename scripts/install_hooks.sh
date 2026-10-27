#!/bin/sh
# Point git at scripts/hooks so the pre-commit data guard runs on every commit.
set -e
cd "$(dirname "$0")/.."
git config core.hooksPath scripts/hooks
chmod +x scripts/hooks/pre-commit
echo "hooks installed: core.hooksPath = scripts/hooks"
