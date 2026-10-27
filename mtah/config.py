"""Project-wide paths and switches. One place, so nothing else spells a directory out.

The match data itself is located by the official games.py (MTAH_DATA_DIR or ./data). Our own
derived files go under CACHE_DIR, human-only debug output under DEBUG_DIR; both are gitignored and
are never read by the coding agent (AGENTS.md section 4).
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # games.py, evaluation/ and starter/ live at the repository root

# The registered team name. It names the submission file only; it never changes its content.
# Placeholder until the team confirms the name stated in the registration form.
TEAM_NAME = "TEAM_NAME"

# AGENTS.md section 4. "STRICT": input aggregates are written to CACHE_DIR, never printed.
# "AGGREGATES": up to ~20 low-dimensional input aggregates may be printed. Only the team changes
# this, and only after the organisers confirm it.
FIREWALL_MODE = "STRICT"

CACHE_DIR = os.environ.get("MTAH_CACHE_DIR") or os.path.join(ROOT, "cache")
DEBUG_DIR = os.environ.get("MTAH_DEBUG_DIR") or os.path.join(ROOT, "debug")
LOG_DIR = os.path.join(DEBUG_DIR, "logs")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
