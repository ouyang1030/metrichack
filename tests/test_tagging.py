"""Smart Tagging as decoder evidence (mtah/tagging.py) and its simulated tagger
(harness/tagger.py), on synthetic data."""

import numpy as np

from harness import tagger
from mtah import tagging
from evaluation import possession_metrics as pm


def _event(code, team, start, end):
    return {"code": code, "start_frame": start, "end_frame": end, "tags": {"Team": team}}


def test_attacking_phases_vote_for_their_team_and_defending_ones_for_the_other():
    sides = {"Home FC": tagging.H, "Away FC": tagging.A}
    events = [_event("BUILD UP", "Home FC", 0, 100), _event("DEFENDING IN LOW BLOCK", "Away FC",
                                                             200, 300),
              _event("SET PIECES", "Home FC", 400, 500), _event("PROGRESSION", "N/A", 600, 700)]
    spans = tagging.phases(events, sides)
    assert spans == [(0, 100, tagging.H), (200, 300, tagging.H)]
    vote = tagging.votes(spans + [(250, 350, tagging.A)], 800)
    edge = int(tagging.EDGE_S * 25)
    assert (vote[edge:100 - edge] == tagging.H).all()
    assert (vote[:edge] == tagging.NONE).all()
    assert (vote[100 - edge:200 + edge] == tagging.NONE).all()
    assert (vote[200 + edge:250 + edge] == tagging.H).all()
    assert (vote[250 + edge:300 - edge] == tagging.NONE).all()  # the phases disagree
    assert (vote[300 - edge:350 - edge] == tagging.A).all()


def test_team_names_map_by_shared_words_or_not_at_all():
    assert tagging.team_sides(["FC Barcelona", "Athletic Club"], "FC Barcelona",
                              "Athletic Club") == {"FC Barcelona": tagging.H,
                                                   "Athletic Club": tagging.A}
    assert tagging.team_sides(["Club"], "Club One", "Club Two") is None


def test_em_recovers_how_reliable_the_votes_are():
    rng = np.random.default_rng(0)
    n = 60000
    state = rng.choice(3, n, p=[0.35, 0.3, 0.35])
    truth = np.array([[0.7, 0.1, 0.2], [0.1, 0.7, 0.2], [0.05, 0.05, 0.9]])
    vote = np.array([rng.choice(3, p=truth[s]) for s in state])
    # A calibrated classifier that is right with probability 0.8, spreading the rest evenly.
    probs = np.full((n, 3), 0.1)
    guess = np.where(rng.random(n) < 0.8, state, rng.choice(3, n))
    probs[np.arange(n), guess] = 0.8
    probs /= probs.sum(axis=1, keepdims=True)
    # The posterior of such a classifier is not exactly its output, so the estimate is checked
    # loosely: it must find the votes far better than chance and their errors rare.
    table = tagging.confusion(probs, vote, np.ones(n, dtype=bool))
    np.testing.assert_allclose(table.sum(axis=1), 1.0)
    assert table[0, 0] == table[1, 1] and table[0, 1] == table[1, 0]  # symmetric in the teams
    assert table[0, 0] > 0.55 and table[0, 1] < 0.2 and table[2, 2] > 0.75
    evidence = tagging.log_evidence(table, np.array([tagging.H, tagging.A, tagging.NONE]))
    assert evidence[0].argmax() == 0 and evidence[1].argmax() == 1 and evidence[2].argmax() == 2


def test_the_simulated_tagger_follows_the_possessions_and_flips_their_team():
    labels = np.repeat(np.array([pm.H, pm.D, pm.H, pm.A, pm.H], dtype=np.int8),
                       [500, 100, 500, 20, 500])
    halves = np.array([[0, len(labels)]])
    d = tagger.draws(labels, halves, np.random.default_rng(0))
    # The 0.8 s away spell gets no phase; the home phases around it merge across it, not
    # across the stoppage.
    phases = tagger.phases(d, tagger.Knobs(min_s=2.0, merge_s=2.0, miss=0.0, flip=0.0))
    assert [team for _, _, team in phases] == [tagging.H, tagging.H]
    assert tagger.phases(d, tagger.Knobs(min_s=30.0, merge_s=2.0, miss=0.0, flip=0.0)) == []
    assert tagger.phases(d, tagger.Knobs(min_s=2.0, merge_s=2.0, miss=1.0, flip=0.0)) == []
    flipped = tagger.phases(d, tagger.Knobs(min_s=2.0, merge_s=2.0, miss=0.0, flip=1.0))
    assert [team for _, _, team in flipped] == [tagging.A, tagging.A]


def test_vote_statistics():
    vote = np.repeat(np.array([tagging.NONE, tagging.H, tagging.NONE, tagging.H, tagging.A,
                               tagging.NONE], dtype=np.int8), [50, 100, 50, 250, 250, 300])
    path = np.repeat(np.array([pm.A, pm.H, pm.D, pm.A], dtype=np.int8), [100, 200, 300, 400])
    stats = tagger.statistics(vote, path, tagger.runs_of(np.ones(len(vote), dtype=bool)))
    assert stats["voted"] == 0.6
    assert np.isclose(stats["run_s"], 200 / 25)
    assert np.isclose(stats["short_runs"], 1 / 3)
    assert np.isclose(stats["switches_per_min"], 1 / (1000 / 25 / 60))
    # Alive and voted: 50 A and 150 H on H votes, 100 A on A votes (the rest is dead).
    assert np.isclose(stats["agreement"], 250 / 300)
    assert tagger.runs_of(np.array([0, 1, 1, 0, 1], dtype=bool)) == [(1, 3), (4, 5)]
