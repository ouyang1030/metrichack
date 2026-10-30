"""The one command: data bundle -> evaluation/results/<TEAM_NAME>.csv, validated.

    python scripts/make_submission.py
    python scripts/make_submission.py --team <name>

Builds the parsed cache if it is missing, checks that Smart Tagging's kickoff windows contain the
published kickoffs (the pipeline places the second-half kickoff by them), runs mtah.pipeline,
writes the file and runs the official validator on it. Runs behind the firewall (mtah/safe_run.py): the terminal only ever
shows checks, the validator's verdict and summaries of our own predictions.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah import cache, config, pipeline, safe_run  # noqa: E402
from mtah.submission import write_submission  # noqa: E402
from starter import load_smart_tagging as official_tagging  # noqa: E402


def main(report, args):
    cache.build(report, force=args.rebuild_cache)
    events = cache.load_smart_tagging()
    report.check("Smart Tagging kickoff windows contain the published kickoffs",
                 official_tagging.check_time_base(events)["agrees"])
    labels = pipeline.predict(cache.load_atd(), events)

    path = write_submission(args.out, labels)
    report.wrote(path)
    report.require("official validator accepts the file", report.validate(path) == 0)
    report.predictions(os.path.basename(path), labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", default=config.TEAM_NAME,
                        help="names the file <team>.csv (default: %(default)s)")
    parser.add_argument("--out", help="write here instead of evaluation/results/<team>.csv")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="re-parse the data bundle even if a cache exists")
    args = parser.parse_args()
    args.out = args.out or os.path.join(config.RESULTS_DIR, args.team + ".csv")
    print("[make_submission] pipeline {}".format(pipeline.VERSION))
    sys.exit(safe_run.run(main, args, name="make_submission"))
