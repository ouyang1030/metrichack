import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import synthetic_bundle  # noqa: E402


@pytest.fixture
def small_bundle(tmp_path):
    return synthetic_bundle.write_bundle(tmp_path / "data", n_frames=3000, seed=1)


@pytest.fixture
def edge_bundle(tmp_path):
    """Every case where starter/baseline.py's loop has a non-obvious answer."""
    return synthetic_bundle.write_bundle(tmp_path / "data", n_frames=2000, seed=2,
                                         edge_cases=True)
