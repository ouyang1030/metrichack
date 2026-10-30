"""Boundary refinement: move each decoded switch back to the onset the physics points to.

The decoder switches where the accumulated evidence turns, which lags the event: the teams'
movement reacts about a second after a turnover, the players slow down after the whistle. Two
onsets are cheap to find just before a switch (AGENTS.md section 6, "backdate the boundary"):

  * a change of team: the first located frame of the last stretch in which the new team was
    nearest the ball whenever the ball was located -- its first touch;
  * play stopping: the first frame of the last stretch in which the ball was last seen beyond a
    line -- the ball crossing it.

A boundary moves back at most WINDOW frames and never past the previous switch.
"""

import numpy as np

from mtah.emissions import A, D, H

WINDOW = 50  # frames (2 s)


def _onset(t, low, holds):
    """Earliest s in [low, t] with holds[s:t] all true."""
    s = t
    while s > low and holds[s - 1]:
        s -= 1
    return s


def boundaries(path, f):
    path = path.copy()
    closer = np.where(f.near_home < f.near_away, H, A)
    previous = 0
    for t in np.flatnonzero(path[1:] != path[:-1]) + 1:
        old, new = path[t - 1], path[t]
        low = max(t - WINDOW, previous + 1)
        if old != D and new != D:
            s = _onset(t, low, ~f.seen | (closer == new))
            s += int(np.argmax(f.seen[s:t])) if f.seen[s:t].any() else t - s
        elif new == D:
            s = _onset(t, low, f.last_out)
        else:
            s = t
        path[s:t] = new
        previous = t
    return path
