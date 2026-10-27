"""Score files against the PUBLIC 5-minute window -- as an alarm only -- and log the look.

    python scripts/smoke_test_public.py --reason "v1 smoke test" evaluation/results/TEAM_NAME.csv

AGENTS.md 5.4: the public window checks that the pipeline is correct (alignment, file format,
label semantics). It never chooses between variants, never sets a threshold, and is never looked
at error by error. Every run appends a line to COMPLIANCE_LOG.md; add what was (not) changed
because of it by hand.

Runs the official run_evaluation.py on copies of the given files, ranked table only
(--no-detail); its reports, which carry ground-truth transition frames, go to cache/eval/.
"""

import argparse
import datetime
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah import config, safe_run  # noqa: E402

COMPLIANCE_LOG = os.path.join(config.ROOT, "COMPLIANCE_LOG.md")


def main(report, args):
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = os.path.join(config.CACHE_DIR, "eval", stamp)
    results = os.path.join(out, "results")
    os.makedirs(results)
    for path in args.files:
        shutil.copy(path, results)

    code = report.evaluate("--gt", "public", "--results", results, "--out", out, "--no-detail")
    report.check("official scorer scored every file", code == 0)

    with open(COMPLIANCE_LOG, "a") as handle:
        handle.write("- {} `smoke_test_public` on {} -- reason: {}. Alarm only; not used to "
                     "choose between variants or to set anything.\n".format(
                         datetime.date.today().isoformat(),
                         ", ".join("`{}`".format(os.path.basename(p)) for p in args.files),
                         args.reason))
    report.wrote(COMPLIANCE_LOG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", help="frame,label CSVs to score")
    parser.add_argument("--reason", required=True,
                        help="why we are looking (logged in COMPLIANCE_LOG.md)")
    sys.exit(safe_run.run(main, parser.parse_args(), name="smoke_test_public"))
