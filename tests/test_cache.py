import io

import numpy as np

from mtah import cache
from mtah.safe_run import Report


def test_build_load_and_reuse(small_bundle, tmp_path):
    cache_dir = str(tmp_path / "cache")
    report = Report(io.StringIO(), "STRICT", str(tmp_path / "aggregates"))
    build = dict(data_dir=small_bundle["data_dir"], cache_dir=cache_dir, n_frames=3000)
    assert cache.build(report, **build)
    assert report.failed == []

    atd = cache.load_atd(cache_dir=cache_dir)
    np.testing.assert_array_equal(atd.players, small_bundle["players"])
    np.testing.assert_array_equal(atd.ball, small_bundle["ball"])
    np.testing.assert_array_equal(atd.slot_team, small_bundle["slot_team"])
    matrices, _, valid = cache.load_homography(cache_dir=cache_dir)
    assert matrices.shape == (3000, 3, 3) and valid.all()
    assert len(cache.load_smart_tagging(cache_dir=cache_dir)) == 4

    assert not cache.build(report, **build)
    assert cache.build(report, force=True, **build)
