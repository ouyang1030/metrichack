"""Parse the data bundle into cache/ (once; later runs reuse it while the inputs are unchanged).

    python scripts/build_cache.py                    # tactical
    python scripts/build_cache.py --source all --force
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah import cache, safe_run  # noqa: E402

SOURCES = ("tactical", "broadcast")


def main(report, args):
    for source in (SOURCES if args.source == "all" else (args.source,)):
        cache.build(report, source=source, force=args.force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=SOURCES + ("all",), default="tactical")
    parser.add_argument("--force", action="store_true", help="rebuild even if current")
    args = parser.parse_args()
    sys.exit(safe_run.run(main, args, name="build_cache"))
