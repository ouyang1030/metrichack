"""Reproduce the four published baselines and check they are byte-identical.

    python scripts/reproduce_baselines.py                 # also re-runs the official script
    python scripts/reproduce_baselines.py --skip-official

For each variant, our vectorised reproduction (mtah/baseline.py, from the cache) is compared byte
for byte with the committed evaluation/results/<variant>.csv and -- unless --skip-official --
with a fresh run of the official starter/baseline.py on the same data. Files go to
cache/baselines/, never over the committed ones.
"""

import argparse
import filecmp
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah import baseline, cache, config, safe_run  # noqa: E402
from mtah.submission import validation_error, write_submission  # noqa: E402
import games  # noqa: E402

OFFICIAL_SCRIPT = os.path.join(config.ROOT, "starter", "baseline.py")


def main(report, args):
    cache.build(report)
    atd = cache.load_atd()
    codes = baseline.nearest_player_codes(atd.players, atd.ball, atd.slot_team,
                                          games.GAMES[games.GAME]["pitch"])
    ours = {}
    for name, labels in baseline.variants(codes).items():
        ours[name] = write_submission(os.path.join(args.out, "ours", name + ".csv"), labels)
        report.check("{}: valid".format(name), validation_error(ours[name]) is None)
        report.check("{}: identical to the committed file".format(name), filecmp.cmp(
            ours[name], os.path.join(config.RESULTS_DIR, name + ".csv"), shallow=False))

    if args.skip_official:
        return
    report.status("running the official starter/baseline.py (pure Python, a few minutes)")
    official_dir = os.path.join(args.out, "official")
    done = subprocess.run([sys.executable, OFFICIAL_SCRIPT, "--out", official_dir],
                          cwd=config.ROOT)  # its output, which describes the input, -> log
    report.require("official starter/baseline.py ran", done.returncode == 0)
    for name, path in ours.items():
        report.check("{}: identical to the official script's output".format(name), filecmp.cmp(
            path, os.path.join(official_dir, name + ".csv"), shallow=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(config.CACHE_DIR, "baselines"))
    parser.add_argument("--skip-official", action="store_true",
                        help="skip re-running starter/baseline.py")
    args = parser.parse_args()
    sys.exit(safe_run.run(main, args, name="reproduce_baselines"))
