"""v2 building blocks on synthetic data: the feature tables' team symmetry, the tree export, the
input-output HMM."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from mtah import atd, decoder, features, frames, pipeline, trees

PITCH = (105.0, 68.0)


def _scene(n=400, seed=0):
    """Eight slots, four per team, wandering; the ball follows slot 0 and is lost now and then."""
    rng = np.random.default_rng(seed)
    players = np.empty((n, 8, 3))
    players[:, :, atd.X] = 0.5 + np.cumsum(rng.normal(0, 0.002, (n, 8)), axis=0)
    players[:, :4, atd.X] -= 0.2  # the home team sits on the smaller-x side
    players[:, :, atd.Y] = 0.5 + np.cumsum(rng.normal(0, 0.002, (n, 8)), axis=0)
    players[:, :, atd.SPEED] = rng.uniform(0, 6, (n, 8))
    players[rng.random((n, 8)) < 0.05] = np.nan
    ball = players[:, 0].copy()
    ball[:, atd.X] = 0.3 + np.cumsum(rng.normal(0, 0.004, n))
    ball[rng.random(n) < 0.3] = np.nan
    return players, ball, np.array([atd.HOME] * 4 + [atd.AWAY] * 4)


def test_swapping_the_teams_permutes_the_columns():
    players, ball, slot_team = _scene()
    X, names = features.matrix(players, ball, slot_team, PITCH)
    swapped, _ = features.matrix(players, ball, np.where(slot_team == atd.HOME, atd.AWAY,
                                                         atd.HOME), PITCH)
    index = features.swap_index(names)
    np.testing.assert_array_equal(index[index], np.arange(len(names)))
    np.testing.assert_allclose(swapped, X[:, index], equal_nan=True)


def _pass(receiver):
    """The reception features mid-flight: two home slots and two away ones standing still; home
    slot 0 holds the ball for 2 s, plays it to the receiver over 2 s, who then holds it for 4 s."""
    n = 200
    spots = np.array([(20.0, 34.0), (40.0, 34.0), (40.0, 50.0), (80.0, 10.0)]) / PITCH
    players = np.zeros((n, 4, 3))
    players[:, :, :2] = spots
    path = np.r_[np.zeros(50), np.linspace(0, 1, 50), np.ones(100)]
    ball = np.zeros((n, 3))
    ball[:, :2] = spots[0] + path[:, None] * (spots[receiver] - spots[0])
    X, names = features.matrix(players, ball, np.array([atd.HOME] * 2 + [atd.AWAY] * 2), PITCH)
    return dict(zip(names, X[75]))


def test_a_pass_and_an_interception_show_who_gets_the_ball_next():
    completed, intercepted = _pass(1), _pass(2)
    for flight in (completed, intercepted):
        assert flight["contact_last_home"] == 1 and flight["contact_last_away"] == 0
        assert flight["hold_last_home"] > 0.9
    assert completed["contact_next_home"] == 1 and completed["contact_next_away"] == 0
    assert intercepted["contact_next_away"] == 1 and intercepted["contact_next_home"] == 0
    assert completed["hold_next_home"] > 0.9 and intercepted["hold_next_away"] > 0.9
    assert 15 < completed["flight_m"] < 20 < intercepted["flight_m"] < 26


def test_exported_trees_give_scikit_learns_probabilities():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 6)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 1)
    X[rng.random(X.shape) < 0.2] = np.nan
    classifier = HistGradientBoostingClassifier(max_iter=20, early_stopping=False,
                                                random_state=0).fit(X, y)
    np.testing.assert_allclose(trees.probabilities(X, trees.export(classifier)),
                               classifier.predict_proba(X), rtol=0, atol=1e-12)


def test_swapping_the_teams_permutes_the_second_stage_columns():
    players, ball, slot_team = _scene()
    X, names = features.matrix(players, ball, slot_team, PITCH)
    rng = np.random.default_rng(1)
    probs, change = rng.dirichlet(np.ones(3), len(X)), rng.dirichlet(np.ones(4), len(X))
    table, added = features.second_stage(X, probs, change)
    swapped, _ = features.second_stage(X[:, features.swap_index(names)], probs[:, [1, 0, 2]],
                                       change)
    np.testing.assert_allclose(swapped, table[:, features.swap_index(names + added)],
                               equal_nan=True)


def test_run_lengths_count_back_and_forward():
    past, future = features.run_length(np.array([0, 1, 1, 1, 0, 1, 1, 0], dtype=bool))
    assert past.tolist() == [0, 1, 2, 3, 0, 1, 2, 0]
    assert future.tolist() == [0, 3, 2, 1, 0, 2, 1, 0]


LOG_TRANS = np.log(np.array([[0.98, 0.01, 0.01], [0.01, 0.98, 0.01], [0.02, 0.02, 0.96]]))
MEAN_FRAMES = [100.0, 100.0, 50.0]


def test_viterbi_takes_per_frame_transitions():
    log_emit = np.log(np.random.default_rng(0).dirichlet(np.ones(3), 200))
    log_trans = decoder.chain(LOG_TRANS, MEAN_FRAMES, 1)[0]
    np.testing.assert_array_equal(
        decoder.viterbi(log_emit, np.broadcast_to(log_trans, (200, 3, 3))),
        decoder.viterbi(log_emit, log_trans))


def test_without_the_detector_the_input_output_hmm_is_the_plain_one():
    change = np.random.default_rng(0).dirichlet(np.ones(4), 50)
    varying = decoder.input_transitions(LOG_TRANS, MEAN_FRAMES, change, 5, 0.0)
    np.testing.assert_allclose(varying, np.broadcast_to(
        decoder.chain(LOG_TRANS, MEAN_FRAMES, 1)[0], (50, 3, 3)))


def test_a_likely_switch_makes_switching_cheaper_there_only():
    change = np.array([[0.97, 0.01, 0.01, 0.01], [0.1, 0.8, 0.05, 0.05]])
    plain = decoder.chain(LOG_TRANS, MEAN_FRAMES, 1)[0]
    varying = decoder.input_transitions(LOG_TRANS, MEAN_FRAMES, change, 5, 1.0)
    assert varying[0, 0, 1] < plain[0, 1] < varying[1, 0, 1]
    np.testing.assert_allclose(np.exp(varying).sum(axis=2), 1.0)


def _kickoff(start, end):
    return {"code": "SET PIECES", "start_frame": start, "end_frame": end,
            "tags": {"Type": "Kick Off (Start)"}}


def test_frames_before_the_second_half_kickoff_are_dead_and_nothing_else():
    (first, _), (second, stop) = frames.graded_runs()
    events = [_kickoff(first - 50, first + 75),            # the first half's: graded from it
              _kickoff(first + 9000, first + 9125),        # after a goal, should it be tagged
              {"code": "BUILD UP", "start_frame": second, "end_frame": second + 500, "tags": {}},
              _kickoff(second + 3000, second + 3125)]
    mask = pipeline.before_kickoff(events)
    np.testing.assert_array_equal(np.flatnonzero(mask), np.arange(second, second + 3000))
