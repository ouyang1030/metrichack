"""Firewall wrapper for every entry point that touches BAR-ATH data (AGENTS.md section 4).

The match data must never reach a hosted LLM, and the coding agent is one. So a script that reads
the data runs as

    sys.exit(safe_run.run(main, args, name="make_submission"))

and inside run():

  * file descriptors 1 and 2 are pointed at a log file under debug/logs/ (gitignored, for humans
    only). Every print, warning, C-level write and child process lands there, never on the
    terminal.
  * the terminal is reachable only through the Report handed to main(), whose methods emit the
    kinds of output the protocol allows: pass/fail checks, paths, runtimes, memory, summaries of
    OUR predictions, the official validator's and scorer's output about our files, and (mode
    AGGREGATES only) a handful of input aggregates.
  * an exception is reported as its type, a file:line, and its message with every digit masked.
    The full traceback goes to the log.
"""

import datetime
import json
import os
import re
import resource
import subprocess
import sys
import time
import traceback

from mtah import config
from mtah.submission import prediction_summary

MAX_MESSAGE = 200
_DIGITS = re.compile(r"\d")


def mask(text):
    """Digits -> '#', one line, truncated. For anything whose content the code does not control."""
    text = _DIGITS.sub("#", " | ".join(str(text).split("\n")))
    return text if len(text) <= MAX_MESSAGE else text[:MAX_MESSAGE - 3] + "..."


def _relpath(path):
    path = os.path.abspath(path)
    return os.path.relpath(path, config.ROOT) if path.startswith(config.ROOT + os.sep) else path


class CheckFailed(Exception):
    """Raised by Report.require. Carries only the check name, which is a literal in our code."""


class Report:
    """The only channel from a firewalled script to the terminal."""

    def __init__(self, stream, mode, aggregates_dir):
        self._stream = stream
        self.mode = mode
        self.aggregates_dir = aggregates_dir
        self.failed = []

    def _emit(self, line):
        self._stream.write(line + "\n")
        self._stream.flush()

    def status(self, message):
        """Progress text. Digits are masked so that no value can ride along."""
        self._emit("  ..    " + mask(message))

    def check(self, name, passed):
        """A pass/fail line. Failures are remembered and make the run exit non-zero."""
        self._emit("  {}  {}".format("PASS" if passed else "FAIL", name))
        if not passed:
            self.failed.append(name)
        return passed

    def require(self, name, passed):
        """A check that stops the script when it fails."""
        if not self.check(name, passed):
            raise CheckFailed(name)

    def wrote(self, path):
        self._emit("  out   {}".format(_relpath(path)))

    def predictions(self, name, labels):
        """Summary diagnostics of OUR per-frame predictions (never the input data)."""
        summary = prediction_summary(labels)
        self._emit("  pred  {}: {} possessions, duration s p10/p50/p90 {:.1f}/{:.1f}/{:.1f}, "
                   "short(<1 s) {:.1%}, D share {:.1%}, X on graded {}".format(
                       name, summary["n_possessions"], summary["duration_p10_s"],
                       summary["duration_p50_s"], summary["duration_p90_s"],
                       summary["short_share"], summary["dead_share"], summary["x_on_graded"]))
        return summary

    def validate(self, path):
        """The official validator's verdict on one of our files. Returns its exit code."""
        return self._relay("validate_submission.py", path)

    def evaluate(self, *args):
        """The official scorer's output on our files. Returns its exit code."""
        return self._relay("run_evaluation.py", *args)

    def _relay(self, script, *args):
        done = subprocess.run([sys.executable, os.path.join("evaluation", script)] + list(args),
                              capture_output=True, text=True, cwd=config.ROOT)
        sys.stderr.write(done.stderr)  # -> log
        for line in done.stdout.splitlines():
            self._emit("  |     " + line)
        return done.returncode

    def aggregates(self, name, values):
        """At most ~20 low-dimensional input aggregates. STRICT: written to a JSON, never shown."""
        os.makedirs(self.aggregates_dir, exist_ok=True)
        with open(os.path.join(self.aggregates_dir, name + ".json"), "w") as handle:
            json.dump(values, handle, indent=2, sort_keys=True)
        if self.mode == "STRICT":
            self._emit("  agg   {}: {} values written OK (STRICT, not shown)".format(
                name, len(values)))
        else:
            self._emit("  agg   {}: {}".format(name, ", ".join(
                "{}={:.4g}".format(k, v) for k, v in sorted(values.items()))))


class _Firewall:
    """Point fds 1 and 2 at a log file; hand back a stream to the real terminal."""

    def __init__(self, log_path):
        self.log_path = log_path

    def __enter__(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        sys.stdout.flush()
        sys.stderr.flush()
        self._saved = (os.dup(1), os.dup(2))
        self._log = open(self.log_path, "w")
        os.dup2(self._log.fileno(), 1)
        os.dup2(self._log.fileno(), 2)
        self.out = os.fdopen(os.dup(self._saved[0]), "w")
        return self

    def __exit__(self, *exc_info):
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self._saved[0], 1)
        os.dup2(self._saved[1], 2)
        for fd in self._saved:
            os.close(fd)
        self._log.close()
        self.out.close()
        return False


def describe_exception(exc):
    """Type, our innermost file:line, and the masked message. Nothing else."""
    venv = os.sep + ".venv" + os.sep
    where = [frame for frame in traceback.extract_tb(exc.__traceback__)
             if frame.filename.startswith(config.ROOT) and venv not in frame.filename][-1]
    return "  ERROR {} at {}:{}: {}".format(type(exc).__name__, _relpath(where.filename),
                                           where.lineno, mask(exc))


def _peak_rss_mb():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2.0 ** 20 if sys.platform == "darwin" else peak / 2.0 ** 10


def run(main, *args, name):
    """Run main(report, *args) behind the firewall and return a process exit code."""
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(config.LOG_DIR, "{}-{}.log".format(name, stamp))
    start = time.perf_counter()

    with _Firewall(log_path) as wall:
        report = Report(wall.out, config.FIREWALL_MODE,
                        os.path.join(config.CACHE_DIR, "aggregates"))
        report._emit("[{}] firewall {}; raw output -> {}".format(
            name, config.FIREWALL_MODE, _relpath(log_path)))
        try:
            main(report, *args)
            code = 1 if report.failed else 0
        except CheckFailed as exc:
            report._emit("  STOP  required check failed: {}".format(exc))
            code = 1
        except BaseException as exc:  # noqa: B902 -- nothing may reach the terminal unmasked
            traceback.print_exc()
            report._emit(describe_exception(exc))
            code = 1
        report._emit("[{}] {} in {:.1f} s, peak RSS {:.0f} MB".format(
            name, "ok" if code == 0 else "FAILED", time.perf_counter() - start,
            _peak_rss_mb()))
    return code
