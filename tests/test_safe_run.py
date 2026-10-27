"""The firewall must keep everything but approved report lines off the terminal."""

import io
import os
import subprocess
import sys

import pytest

from mtah.safe_run import Report, mask

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROBE = r"""
import os, subprocess, sys
sys.path.insert(0, {root!r})
from mtah import safe_run

def main(report):
    print("secret 3141592")
    sys.stderr.write("secret 2718281\n")
    subprocess.run(["echo", "secret 1618033"])
    os.write(1, b"secret 1414213\n")
    report.check("a passing check", True)
    report.status("progress with value 5772156")
    raise ValueError("bad value 8675309 at row 4669201")

sys.exit(safe_run.run(main, name="probe"))
"""
SECRETS = ("3141592", "2718281", "1618033", "1414213", "5772156", "8675309", "4669201")


@pytest.fixture
def probe(tmp_path):
    env = dict(os.environ, MTAH_DEBUG_DIR=str(tmp_path), MTAH_CACHE_DIR=str(tmp_path / "c"))
    done = subprocess.run([sys.executable, "-c", PROBE.format(root=ROOT)],
                          capture_output=True, text=True, env=env)
    logs = list((tmp_path / "logs").iterdir())
    return done, logs


def test_terminal_sees_only_report_lines(probe):
    done, _ = probe
    terminal = done.stdout + done.stderr
    assert done.returncode == 1
    for secret in SECRETS:
        assert secret not in terminal
    assert "PASS  a passing check" in terminal
    assert "ERROR ValueError at" in terminal and "bad value ####### at row #######" in terminal


def test_everything_else_lands_in_the_log(probe):
    _, logs = probe
    assert len(logs) == 1
    text = logs[0].read_text()
    for secret in ("3141592", "2718281", "1618033", "1414213", "8675309"):
        assert secret in text
    assert "Traceback" in text


def test_aggregates_are_shown_only_outside_strict(tmp_path):
    for mode, shown in (("STRICT", False), ("AGGREGATES", True)):
        stream = io.StringIO()
        Report(stream, mode, str(tmp_path)).aggregates("probe", {"rate": 0.123})
        assert ("0.123" in stream.getvalue()) is shown
        assert (tmp_path / "probe.json").exists()


def test_mask():
    assert mask("frame 11185 at x=0.52") == "frame ##### at x=#.##"
    assert mask("not found:\n  /data/a1.txt") == "not found: |   /data/a#.txt"
    assert len(mask("9" * 1000)) == 200
