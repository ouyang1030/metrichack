"""starter/baseline.py, vectorised over the parsed cache.

The official script streams the raw ATD line by line (minutes); this computes the same per-frame
nearest-player codes from cached arrays (seconds) and then reuses the official smoothing, so the
four published baseline files can be reproduced byte for byte.

Bit-exactness notes, matching starter/baseline.py's loop:
  * a slot is skipped when its x is missing; the ball is missing when its x is missing;
  * distances are computed in metres with the same float64 operations in the same order;
  * ties go to the first slot (strict '<' in the loop, first index in argmin);
  * the loop's NaN quirk is kept: if the first usable slot's distance is NaN (its y missing, or
    the ball's y missing), nothing can beat it and it wins; a NaN later never wins.
"""

import numpy as np

from mtah import atd
from starter.baseline import AWAY, HOME, NOBODY, VARIANTS, smooth  # noqa: F401 (re-exported)


def nearest_player_codes(players, ball, slot_team, pitch, chunk=16384):
    """Per-frame team code (HOME / AWAY) of the slot nearest the ball, NOBODY without a ball."""
    length, width = pitch
    codes = np.full(players.shape[0], NOBODY, dtype=np.int8)

    for start in range(0, players.shape[0], chunk):
        px = players[start:start + chunk, :, atd.X]
        py = players[start:start + chunk, :, atd.Y]
        ball_x = ball[start:start + chunk, atd.X] * length
        ball_y = ball[start:start + chunk, atd.Y] * width

        dx = px * length - ball_x[:, None]
        dy = py * width - ball_y[:, None]
        distance = dx * dx + dy * dy

        usable = ~np.isnan(px)
        rows = np.flatnonzero(~np.isnan(ball_x) & usable.any(axis=1))
        first = np.argmax(usable[rows], axis=1)
        ranked = np.where(usable[rows] & ~np.isnan(distance[rows]), distance[rows], np.inf)
        best = np.where(np.isnan(distance[rows, first]), first, np.argmin(ranked, axis=1))
        codes[start + rows] = slot_team[best]
    return codes


def variants(codes):
    """{name: labels} for the four published baselines."""
    return {name: smooth(codes, window) for name, window in VARIANTS}
