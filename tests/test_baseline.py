"""mtah/baseline.py must reproduce starter/baseline.py bit for bit, quirks included."""

import numpy as np

from mtah import atd, baseline
from mtah.submission import write_submission
from starter import baseline as official

PITCH = (105.3, 68.0)


def _both(bundle):
    paths, n_frames = bundle["paths"], bundle["n_frames"]
    team = official.team_of_each_track(paths["atd_meta"])
    expected = official.nearest_player_labels(paths["atd_raw"], team, n_frames, PITCH)
    players, ball = atd.parse_raw(paths["atd_raw"], len(team), 3, n_frames)
    return expected, baseline.nearest_player_codes(players, ball, team, PITCH, chunk=700)


def test_codes_match_the_official_loop_on_every_edge_case(edge_bundle):
    expected, ours = _both(edge_bundle)
    np.testing.assert_array_equal(ours, expected)
    team = edge_bundle["slot_team"]
    assert (ours[0:10] == team[3]).all()      # tie -> first slot
    assert (ours[10:20] == team[2]).all()     # NaN distance first -> it sticks
    assert (ours[20:30] == team[11]).all()    # NaN distance later -> ignored
    assert (ours[30:40] == team[4]).all()     # ball without y -> first usable slot
    assert (ours[40:50] == official.NOBODY).all()
    assert (ours[50:60] == team[6]).all()     # slot without x is skipped


def test_codes_match_on_random_tracking(small_bundle):
    expected, ours = _both(small_bundle)
    np.testing.assert_array_equal(ours, expected)


def test_files_are_byte_identical_to_the_official_writer(edge_bundle, tmp_path):
    expected, ours = _both(edge_bundle)
    for name, window in official.VARIANTS:
        theirs = official.write_submission(str(tmp_path / ("official_" + name)),
                                           official.smooth(expected, window))
        mine = write_submission(str(tmp_path / ("ours_" + name)), baseline.smooth(ours, window))
        with open(theirs, "rb") as a, open(mine, "rb") as b:
            assert a.read() == b.read(), name
