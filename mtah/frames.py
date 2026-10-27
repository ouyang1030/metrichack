"""The submission's frame contract: length, rate and which frames are graded.

Only numbers the organisers publish appear here, and they come from the official games.py.
"""

import numpy as np

from mtah import config  # noqa: F401  (puts the repository root on sys.path)
import games

N_FRAMES = games.source_frames("tactical", games.GAME)
FPS = games.GAMES[games.GAME]["fps"]


def graded_runs():
    """Half-open [start, stop) video-frame spans of the two graded halves."""
    return [(first, last + 1) for _, first, last in games.half_video_blocks(games.GAME)]


def graded_mask(n_frames=N_FRAMES):
    mask = np.zeros(n_frames, dtype=bool)
    for start, stop in graded_runs():
        mask[start:stop] = True
    return mask
