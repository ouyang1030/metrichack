"""The off-game harness: the noise simulator and the scoring wrapper, on synthetic matches."""

from dataclasses import replace

import numpy as np

from harness import degrade, score
from harness.idsse import Match
from evaluation import possession_metrics as pm

HALF, GAP = 3000, 100


def _match(seed=0):
    """Two halves of random-walk tracking for 22 people, labels in 200-frame runs."""
    rng = np.random.default_rng(seed)
    n = 2 * HALF + GAP
    walk = 0.5 + np.cumsum(rng.normal(0, 0.002, (n, 22, 2)), axis=0)
    players = np.concatenate((walk, rng.random((n, 22, 1)) * 8), axis=2)
    ball = np.column_stack((0.5 + np.cumsum(rng.normal(0, 0.004, (n, 2)), axis=0),
                            rng.random(n) * 20))
    labels = np.repeat(rng.choice([pm.H, pm.A, pm.D], n // 200 + 1), 200)[:n].astype(np.int8)
    for array in (players, ball):
        array[HALF:HALF + GAP] = np.nan
    labels[HALF:HALF + GAP] = pm.X
    return Match("synthetic", players, np.repeat([0, 1], 11).astype(np.int8), ball,
                 rng.random(n) * 3, labels, np.array([[0, HALF], [HALF + GAP, n]]),
                 (105.0, 68.0))


def test_degrade_hits_the_documented_rates():
    match = _match()
    noisy = degrade.degrade(match, seed=0)
    in_play = match.labels != pm.X
    assert abs(np.isnan(noisy.ball[in_play, 0]).mean() - 0.41) < 1e-3
    assert 19.5 < (~np.isnan(noisy.players[in_play, :, 0])).sum(axis=1).mean() < 21.5
    assert noisy.players.shape[1] == 46
    assert list(noisy.slot_team) == [0] * 23 + [1] * 23
    np.testing.assert_array_equal(noisy.labels, match.labels)


def test_tracklets_carry_real_people_of_the_slot_team():
    match = _match()
    params = replace(degrade.Params(), jitter_m=0.0, team_error=0.0, swap_prob=0.0)
    noisy = degrade.degrade(match, seed=1, params=params)
    same = (noisy.players[:, :, None, :] == match.players[:, None, :, :]).all(axis=3)
    tracked = ~np.isnan(noisy.players[:, :, 0])
    assert same.any(axis=2)[tracked].all()
    person = same.argmax(axis=2)
    assert (match.slot_team[person] == noisy.slot_team[None, :])[tracked].all()


def test_false_detections_take_their_share_on_players_or_spots():
    match = _match()
    params = replace(degrade.Params(), ball_false=0.3)
    ball = match.ball.copy()
    false = degrade.false_detections(match, ball, params, np.random.default_rng(0))
    in_play = match.labels != pm.X
    assert 0.22 < false[in_play].mean() < 0.31  # runs overlap now and then
    length, width = match.pitch
    near = np.hypot((match.players[false, :, 0] - ball[false, None, 0]) * length,
                    (match.players[false, :, 1] - ball[false, None, 1]) * width).min(axis=1)
    on_player = near < 1.5
    assert 0.3 < on_player.mean() < 0.7
    assert (ball[false][~on_player, 2] == 0).all()


def test_the_missing_share_holds_with_false_detections():
    match = _match()
    noisy = degrade.degrade(match, seed=0, params=replace(degrade.Params(), ball_false=0.1))
    in_play = match.labels != pm.X
    assert abs(np.isnan(noisy.ball[in_play, 0]).mean() - 0.41) < 1e-3


STOPPAGES = replace(degrade.Params(), ball_missing=0.0, ball_dead_lost=1.0, ball_dead_lag_sd_s=0.0)


def test_a_tracker_that_loses_every_stoppage_misses_the_ball_exactly_while_it_is_dead():
    match = _match()
    noisy = degrade.degrade(match, seed=0, params=STOPPAGES)
    in_play = match.labels != pm.X
    np.testing.assert_array_equal(np.isnan(noisy.ball[in_play, 0]),
                                  match.labels[in_play] == pm.D)


def test_stoppages_are_lost_late_and_found_late_by_the_delays():
    match = _match()
    late = replace(STOPPAGES, ball_dead_on_s=1.0, ball_dead_off_s=1.0)
    lost = degrade.lost_in_stoppages(match, late, np.random.default_rng(0))
    dead = match.labels == pm.D
    expected = np.r_[np.zeros(25, dtype=bool), dead[:-25]] & (match.labels != pm.X)
    np.testing.assert_array_equal(lost, expected)


def test_sightings_split_lost_stoppages_and_none_are_lost_by_default():
    match = _match()
    def lost(params):
        return degrade.lost_in_stoppages(match, params, np.random.default_rng(0))

    whole, seen = lost(STOPPAGES), lost(replace(STOPPAGES, ball_dead_seen_per_min=6.0))
    assert (seen <= whole).all() and seen.sum() < whole.sum()
    assert np.sum(np.diff(seen.astype(int)) == 1) > np.sum(np.diff(whole.astype(int)) == 1)
    assert not lost(degrade.Params()).any()


def test_without_false_detections_the_copies_are_unchanged():
    match = _match()
    a = degrade.degrade(match, seed=2)
    b = degrade.degrade(match, seed=2, params=replace(degrade.Params(), ball_false_s=2.0))
    np.testing.assert_array_equal(a.ball, b.ball)
    np.testing.assert_array_equal(a.players, b.players)


def test_degrade_is_deterministic():
    a, b = degrade.degrade(_match(), seed=3), degrade.degrade(_match(), seed=3)
    np.testing.assert_array_equal(a.players, b.players)
    np.testing.assert_array_equal(a.ball, b.ball)


def test_perfect_prediction_scores_one():
    match = _match()
    rows = score.score_halves(lambda m: np.array(pm.LABELS)[m.labels], match)
    np.testing.assert_array_equal(rows, np.ones((2, 2)))
    mean, low, high = score.interval(rows)
    assert (mean == 1).all() and (low == 1).all() and (high == 1).all()
