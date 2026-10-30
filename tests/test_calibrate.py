"""Calibrating the noise simulator (harness/calibrate.py), on synthetic matches."""

from dataclasses import replace

import numpy as np

from harness import calibrate, degrade
from tests.test_harness import _match

TRUE = replace(degrade.Params(), ball_dead_lost=1.0, ball_dead_seen_per_min=2.0,
               ball_missing=0.05, ball_corr_s=2.0, person_missing=0.12, gap_mean_s=3.0,
               jitter_m=0.4, ball_jitter_m=0.5, ball_false=0.05)


def test_residual_is_zero_on_a_straight_line_and_nan_where_the_window_misses_a_value():
    xy = np.column_stack((np.arange(20.0), 3 * np.arange(20.0)))
    xy[10] = np.nan
    residual = calibrate._residual(xy, 5)
    assert np.isnan(residual[[0, 1, 8, 9, 10, 11, 12, 18, 19]]).all()
    np.testing.assert_allclose(residual[[2, 7, 13, 17]], 0.0, atol=1e-12)


def test_run_lengths_stay_within_the_runs():
    flags = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1], dtype=bool)
    np.testing.assert_array_equal(calibrate._run_lengths(flags, [(0, 5), (5, 9)]), [2, 2, 2, 1])


def test_solve_meets_a_reachable_target_and_flags_one_beyond_reach():
    value, reached = calibrate.solve(lambda v: -v ** 2, -2.0, 0.0, 4.0, steps=30)
    assert reached and abs(value - np.sqrt(2)) < 1e-6
    assert calibrate.solve(lambda v: v, 7.0, 0.0, 4.0) == (4.0, False)


def _smooth_match(seed):
    """_match with a ball that has no height and no speed: their per-frame white noise would make
    the dropout flicker."""
    match = _match(seed)
    ball = match.ball.copy()
    ball[~np.isnan(ball[:, 2]), 2] = 0.0
    return replace(match, ball=ball, ball_z=np.zeros(len(ball)))


def test_fit_reproduces_the_statistics_of_a_copy_and_its_player_knobs():
    matches = [_smooth_match(seed) for seed in range(4)]
    target = calibrate.mean_statistics(matches, TRUE, seed=7)
    # Labels in 200-frame runs give too few stoppages to fit the share of them lost, and a
    # random-walk ball moves alike before and after a stoppage, which leaves the delays of its
    # loss and recovery unmeasured; they keep the values the copy was made with (whether they
    # come back on the DFL matches: EXPERIMENTS.md).
    knobs = {knob: span for knob, span in calibrate.DROPOUT_KNOBS.items()
             if knob not in ("ball_dead_lost", "ball_dead_on_s", "ball_dead_off_s")}
    statistics = ("ball_gap_s", "ball_flicker")
    params, _ = calibrate.fit(matches, target, dropout_knobs=knobs, dropout_statistics=statistics)
    fitted = calibrate.mean_statistics(matches, params)
    for name in (list(calibrate.PLAYER_KNOBS) + ["ball_missing"] + list(statistics)
                 + list(calibrate.DETECTION_KNOBS)):
        assert abs(fitted[name] / target[name] - 1) < 0.1, name
    # The ball knobs are not checked one by one: the random-walk ball takes white steps of about
    # 0.6 m, which pass for jitter and for false detections' jumps, so those trade off against
    # each other and the missing share. Whether the ball knobs come back on real tracking is
    # checked on the DFL matches (EXPERIMENTS.md). The player knobs each have their statistic.
    for knob, tolerance in [("person_missing", 0.03), ("jitter_m", 0.05)]:
        assert abs(getattr(params, knob) - getattr(TRUE, knob)) < tolerance, knob
    assert abs(params.gap_mean_s / TRUE.gap_mean_s - 1) < 0.3


def test_speed_lifts_tell_a_ball_lost_still_from_one_found_moving():
    n = 200
    ball = np.zeros((n, 3))
    ball[:, 0] = np.r_[np.linspace(0.1, 0.3, 80), np.full(40, 0.3), np.linspace(0.3, 0.7, 80)]
    ball[100:120, :2] = np.nan  # lost while still, found moving
    players = np.random.default_rng(0).random((n, 22, 3))
    stats = calibrate.ball_statistics(players, ball, (105.0, 68.0), [(0, n)])
    assert stats["ball_speed_lift"] == 0.0 and stats["ball_return_lift"] > 1.0


def test_players_slowing_after_a_long_gap_and_speeding_up_after_it_show_in_the_edge_ratios():
    n, pitch = 1000, (105.0, 68.0)
    speed = np.r_[np.full(400, 4.0), np.full(200, 1.0), np.full(400, 3.0)]  # m/s
    x = np.cumsum(speed) / 25 / pitch[0]
    players = np.zeros((n, 22, 3))
    players[:, :, 0] = 0.1 + x[:, None]
    ball = np.zeros((n, 3))
    ball[:, 0] = 0.1 + x
    ball[400:600, :2] = np.nan  # 8 s lost while the players slow down
    stats = calibrate.ball_statistics(players, ball, pitch, [(0, n)])
    assert stats["loss_slowdown"] < 0.8 and stats["return_speedup"] > 1.5


def test_stressed_is_noisier_on_every_knob_it_changes():
    params = degrade.Params(ball_false=0.05)
    stressed = degrade.stressed(params)
    assert stressed.ball_missing > params.ball_missing and stressed.ball_false > params.ball_false
    assert stressed.ball_dead_lag_sd_s > params.ball_dead_lag_sd_s
    assert stressed.person_missing > params.person_missing and stressed.jitter_m > params.jitter_m
    assert stressed.ball_weight_outside == 0.0
