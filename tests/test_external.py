"""Labels from Metrica's events (harness/metrica.py), on synthetic event lists."""

import numpy as np

from harness import metrica
from mtah import atd
from evaluation import possession_metrics as pm

H, A = atd.HOME, atd.AWAY


def _labels(events, n=400):
    labels, halves = metrica.event_labels(events, n)
    return labels, halves


def test_possession_follows_control_and_stoppages_run_to_the_restart():
    events = [(H, "SET PIECE", {"KICK OFF"}, 1, 11, 11),
              (H, "PASS", set(), 1, 11, 30),
              (H, "BALL LOST", {"INTERCEPTION"}, 1, 50, 50),  # no switch by itself
              (A, "CHALLENGE", {"GROUND", "WON"}, 1, 60, 60),
              (A, "RECOVERY", {"INTERCEPTION"}, 1, 70, 70),   # control: the switch
              (A, "PASS", set(), 1, 80, 100),
              (A, "BALL OUT", set(), 1, 100, 120),            # dead from the ball crossing
              (H, "RECOVERY", set(), 1, 150, 150),            # the dead ball handed back
              (H, "PASS", {"GOAL KICK"}, 1, 201, 230),        # a goal kick restarts
              (H, "PASS", set(), 1, 230, 300)]
    labels, halves = _labels(events)
    assert halves == [(10, 300)]
    assert (labels[:10] == pm.X).all() and (labels[300:] == pm.X).all()
    assert (labels[10:69] == pm.H).all()
    assert (labels[69:119] == pm.A).all()
    assert (labels[119:200] == pm.D).all()
    assert (labels[200:300] == pm.H).all()


def test_a_foul_with_advantage_is_no_stoppage():
    events = [(H, "SET PIECE", {"KICK OFF"}, 1, 1, 1),
              (H, "PASS", set(), 1, 1, 20),
              (H, "FAULT RECEIVED", set(), 1, 40, 40),
              (A, "CHALLENGE", {"TACKLE", "FAULT", "LOST"}, 1, 40, 40),
              (H, "PASS", set(), 1, 60, 90),                  # play went on
              (H, "SET PIECE", {"FREE KICK"}, 1, 200, 200),
              (H, "PASS", set(), 1, 200, 250)]
    labels, _ = _labels(events)
    assert (labels[:250] == pm.H).all()
