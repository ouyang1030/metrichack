"""v1 building blocks on synthetic sequences: features, emissions, decoder."""

import numpy as np

from mtah import decoder, emissions, features
from evaluation import possession_metrics as pm

H, A, D = emissions.H, emissions.A, emissions.D
# Typical off-game transitions: a possession change every ~10 s, a stoppage every ~30 s.
TRANS = np.log(np.array([[0.9955, 0.0040, 0.0005],
                         [0.0040, 0.9955, 0.0005],
                         [0.0010, 0.0010, 0.9980]]))


def _emit(states, strength=0.5):
    """(n, 3) log evidence favouring the given state on each frame."""
    log_emit = np.zeros((len(states), 3))
    log_emit[np.arange(len(states)), states] = strength
    return log_emit


def test_a_deflection_does_not_switch():
    states = np.full(300, H)
    log_emit = _emit(states)
    log_emit[150, A] = 3.0  # one frame of strong evidence for the other team
    np.testing.assert_array_equal(decoder.viterbi(log_emit, TRANS), states)


def test_controlled_possession_does_switch():
    states = np.array([H] * 300 + [A] * 300)
    path = decoder.viterbi(_emit(states), TRANS)
    np.testing.assert_array_equal(path, states)


def test_h_h_d_d_h_h_is_two_spells():
    states = np.array([H] * 300 + [D] * 100 + [H] * 300)
    path = decoder.viterbi(_emit(states, strength=1.0), TRANS)
    np.testing.assert_array_equal(path, states)
    assert len(pm.possession_spells(path, [(0, len(path))])) == 2


def test_transition_probabilities_are_rows_of_a_stochastic_matrix():
    labels = np.array([H] * 50 + [D] * 20 + [A] * 30, dtype=np.int8)
    probs = np.exp(decoder.transition_log_probs(labels, [(0, 100)]))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert probs[H, D] > probs[H, A]


def test_logistic_fit_recovers_known_weights():
    rng = np.random.default_rng(0)
    X = np.column_stack((np.ones(20000), rng.normal(size=20000)))
    y = (rng.random(20000) < 1 / (1 + np.exp(-(X @ [0.5, -2.0])))).astype(float)
    np.testing.assert_allclose(emissions.fit_logistic(X, y), [0.5, -2.0], atol=0.1)


def _scene():
    """Six frames, two slots (home at x=10 m, away at x=20 m), the ball at x=11 m."""
    players = np.full((6, 2, 3), np.nan)
    players[:, 0] = [10 / 105, 0.5, 2.0]
    players[:, 1] = [20 / 105, 0.5, 4.0]
    ball = np.tile([11 / 105, 0.5, 0.0], (6, 1))
    ball[2:5] = np.nan              # lost for three frames
    ball[1] = [-1 / 105, 0.5, 0.0]  # last seen 1 m beyond the goal line
    return players, ball, np.array([0, 1], dtype=np.int8)


def test_features_of_a_small_scene():
    f = features.compute(*_scene(), (105.0, 68.0))
    np.testing.assert_allclose(f.speed, 3.0)
    np.testing.assert_array_equal(f.seen, [1, 1, 0, 0, 0, 1])
    np.testing.assert_array_equal(f.last_out, [0, 1, 1, 1, 1, 0])
    np.testing.assert_allclose(f.gap * features.FPS, [0, 0, 1, 2, 3, 0])
    np.testing.assert_allclose(f.near_home[[0, 5]], 1.0)
    np.testing.assert_allclose(f.near_away[[0, 5]], 9.0)
    assert np.isinf(f.near_home[3])


def test_no_ball_means_no_team_evidence():
    f = features.compute(*_scene(), (105.0, 68.0))
    model = {"dead_w": [0.0] * 6, "team_terms": ["near"], "team_w": [5.0],
             "prior": [0.3, 0.3, 0.4]}
    log_emit = emissions.log_likelihoods(f, model, 1.0, 1.0)
    np.testing.assert_allclose(log_emit[3, H], log_emit[3, A])
    assert log_emit[0, H] > log_emit[0, A]


def test_a_chain_lasts_at_least_k_frames_and_keeps_the_mean():
    log_trans, owner = decoder.chain(TRANS, [200.0, 200.0, 400.0], 3)
    probs = np.exp(log_trans)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    np.testing.assert_array_equal(owner, [0, 0, 0, 1, 1, 1, 2, 2, 2])
    # only the last sub-state of a state can leave it, and only into a first sub-state
    leaving = probs * (owner[:, None] != owner[None, :])
    assert set(np.flatnonzero(leaving.sum(axis=1))) == {2, 5, 8}
    assert set(np.flatnonzero(leaving.sum(axis=0))) == {0, 3, 6}
    # expected time in H: three sub-states of mean 1 / (1 - stay) each
    assert abs(3 / (1 - probs[0, 0]) - 200.0) < 1e-9


def test_a_one_state_chain_is_a_plain_hmm_with_the_given_mean():
    probs = np.exp(decoder.chain(TRANS, [200.0, 200.0, 400.0], 1)[0])
    np.testing.assert_allclose(np.diag(probs), [1 - 1 / 200, 1 - 1 / 200, 1 - 1 / 400])
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)


def test_mean_durations_per_code():
    labels = np.array([H] * 10 + [D] * 4 + [H] * 20 + [A] * 6, dtype=np.int8)
    assert decoder.mean_durations(labels, [(0, 40)]) == [15.0, 6.0, 4.0]


def _refine_features(n, seen, closer_home, last_out):
    f = features.Features(*(np.zeros(n) for _ in range(11)))
    f.seen, f.last_out = np.asarray(seen, bool), np.asarray(last_out, bool)
    f.near_home = np.where(closer_home, 1.0, 5.0)
    f.near_away = np.where(closer_home, 5.0, 1.0)
    return f


def test_a_late_switch_moves_back_to_the_new_teams_first_touch():
    from mtah import refine
    path = np.array([H] * 30 + [A] * 30, dtype=np.int8)
    closer_home = np.array([True] * 20 + [False] * 40)   # the away team touched it at frame 20
    seen = np.ones(60, bool)
    seen[18:22] = False                                  # lost around the touch
    f = _refine_features(60, seen, closer_home, np.zeros(60))
    np.testing.assert_array_equal(refine.boundaries(path, f),
                                  [H] * 22 + [A] * 38)  # first located away touch


def test_play_stops_when_the_ball_went_out():
    from mtah import refine
    path = np.array([H] * 40 + [D] * 20, dtype=np.int8)
    last_out = np.array([False] * 33 + [True] * 27)
    f = _refine_features(60, np.ones(60), np.ones(60, bool), last_out)
    np.testing.assert_array_equal(refine.boundaries(path, f), [H] * 33 + [D] * 27)
