"""The harness's labels under Metrica's possession rule (AGENTS.md section 3).

DFL's BallPossession names the team of the player in control of the ball. Metrica's ground truth
keeps possession with the last team until an opponent gains *controlled* possession: a touch
without control, a deflection that does not intercept, an aerial duel won without control do not
change it. Some of DFL's very short spells of the other team inside one team's possession may be
exactly such touches. metrica() turns DFL's labels into labels under that rule: a spell of one
team shorter than control_s, between two spells of the other team, goes to the other team. Left
to right, so a ball touched back and forth stays with the team that had it. Spells next to a
stoppage are kept: a restart is control, and a touch before the ball goes out ends in the
stoppage either way. Whether the ground truth really drops such spells is organiser question 6
(docs/ORGANISER_QUESTIONS.md); until it is answered the harness scores both label sets.
"""

import numpy as np

from evaluation import possession_metrics as pm


def metrica(labels, halves, control_s):
    """Label codes with each short spell between two spells of the other team given to that team."""
    out = labels.copy()
    for a, b in halves:
        values = out[a:b]
        edges = np.flatnonzero(np.diff(values)) + 1
        starts, stops = np.r_[0, edges], np.r_[edges, len(values)]
        for i in range(1, len(starts) - 1):
            before, after = values[starts[i - 1]], values[starts[i + 1]]
            if (before == after != pm.D and values[starts[i]] != pm.D
                    and stops[i] - starts[i] < control_s * pm.FPS):
                values[starts[i]:stops[i]] = before
    return out
