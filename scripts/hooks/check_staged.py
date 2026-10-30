"""Pre-commit guard: refuse to commit match data or anything that reconstructs it (AGENTS.md 2.8).

Blocks, among the staged files: anything under data/, cache/, debug/ or external/ at any depth;
video, array and log files; submissions in evaluation/results/ other than the four official
baselines; the scorer's reports; any file over 5 MB; and any CSV with a row per video frame that
is not one of the vendored official files. Standard library only.

Installed by scripts/install_hooks.sh.
"""

import fnmatch
import subprocess
import sys

MAX_BYTES = 5 * 1024 * 1024
FRAME_ROWS = 199319
BLOCKED_DIRS = ("data", "cache", "debug", "external")
BLOCKED_PATTERNS = ("*.mp4", "*.npz", "*.npy", "*.parquet", "*.pkl", "*.log", "*.gzip",
                    "*.pl3container", "evaluation/reports/*", "evaluation/scoreboard.csv")
OFFICIAL_CSVS = ("evaluation/results/baseline_*.csv", "evaluation/gt/*_public.csv")


def blocked_reason(path, size, n_lines):
    """Why this staged file must not be committed, or None."""
    if any(part in BLOCKED_DIRS for part in path.split("/")[:-1]):
        return "path under a data/cache/debug folder"
    if any(fnmatch.fnmatch(path, pattern) for pattern in BLOCKED_PATTERNS):
        return "blocked file type or location"
    if (fnmatch.fnmatch(path, "evaluation/results/*.csv")
            and not fnmatch.fnmatch(path, "evaluation/results/baseline_*.csv")):
        return "a submission file (labels for the whole match)"
    if size > MAX_BYTES:
        return "larger than 5 MB"
    if (path.endswith(".csv") and n_lines >= FRAME_ROWS
            and not any(fnmatch.fnmatch(path, pattern) for pattern in OFFICIAL_CSVS)):
        return "a frame-level CSV that is not a vendored official file"
    return None


def _git(*args):
    return subprocess.run(("git",) + args, capture_output=True, check=True).stdout


def main():
    staged = [p for p in _git("diff", "--cached", "--name-only", "--diff-filter=ACMR",
                              "-z").decode().split("\0") if p]
    problems = []
    for path in staged:
        size = int(_git("cat-file", "-s", ":" + path))
        n_lines = _git("show", ":" + path).count(b"\n") if path.endswith(".csv") else 0
        reason = blocked_reason(path, size, n_lines)
        if reason:
            problems.append("  {}: {}".format(path, reason))

    if problems:
        print("pre-commit: refusing to commit\n" + "\n".join(problems))
        print("Unstage these (git restore --staged <path>). See AGENTS.md section 2.8.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
