import numpy as np

import synthetic_bundle
from mtah import atd


def test_meta_gives_slot_teams_and_ball_channels(small_bundle):
    slot_team, n_ball = atd.read_meta(small_bundle["paths"]["atd_meta"])
    np.testing.assert_array_equal(slot_team, small_bundle["slot_team"])
    assert n_ball == 3


def test_raw_parse_round_trips_exactly(small_bundle):
    players, ball = atd.parse_raw(small_bundle["paths"]["atd_raw"], 46, 3, 3000)
    np.testing.assert_array_equal(players, small_bundle["players"])  # NaN == NaN here
    np.testing.assert_array_equal(ball, small_bundle["ball"])


def test_slot_order_follows_the_spec_not_player_order(tmp_path):
    order = list(reversed(range(46)))
    bundle = synthetic_bundle.write_bundle(tmp_path, n_frames=50, spec_order=order)
    slot_team, _ = atd.read_meta(bundle["paths"]["atd_meta"])
    np.testing.assert_array_equal(slot_team, bundle["slot_team"][order])
    players, _ = atd.parse_raw(bundle["paths"]["atd_raw"], 46, 3, 50)
    np.testing.assert_array_equal(players, bundle["players"][:, order])
