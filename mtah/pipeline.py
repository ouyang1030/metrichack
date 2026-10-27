"""The submission pipeline: parsed cache in, one label per video frame out.

v0 is the published floor, baseline_smooth50, so the one-command path can be verified byte for
byte against a committed file before anything of ours replaces it.
"""

from mtah import baseline
import games

VERSION = "v0-baseline_smooth50"
V0_WINDOW = 50


def predict(atd):
    """Labels for all 199,319 frames from a cache.Atd."""
    codes = baseline.nearest_player_codes(atd.players, atd.ball, atd.slot_team,
                                          games.GAMES[games.GAME]["pitch"])
    return baseline.smooth(codes, V0_WINDOW)
