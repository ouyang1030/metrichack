"""Parse the provided assets once into CACHE_DIR; everything downstream loads arrays from there.

    cache/<source>/
        atd_players.npy      (n_frames, n_slots, 3) float64: x, y (normalised 0-1), speed (m/s)
        atd_ball.npy         (n_frames, n_ball_channels) float64
        slot_team.npy        (n_slots,) int8 HOME / AWAY, starter/baseline.py's <Player> order
        homography.npz       matrices (n, 3, 3), reprojection_error, valid
        smart_tagging.json   tactical only: the official loader's events, in video frames
        manifest.json        parser version; written last, so it marks a complete build

Rows are video frames (ATD frame_id - 1). A cache is reused while its parser version is current;
after a new data bundle, rebuild with --force. The cache is match data: gitignored, and never
opened by the coding agent. A build runs behind safe_run, so everything it says goes through
`report`.
"""

import json
import os
from dataclasses import dataclass

import numpy as np

from mtah import atd, config
import games
from starter import baseline as official_baseline
from starter import load_homography as official_homography
from starter import load_smart_tagging as official_tagging

PARSER_VERSION = 2


def source_dir(source="tactical", cache_dir=None):
    return os.path.join(cache_dir or config.CACHE_DIR, source)


def build(report, source="tactical", data_dir=None, cache_dir=None, n_frames=None, force=False):
    """Parse one source into the cache unless a build by this parser version is there.

    Returns True if it built.
    """
    folder = source_dir(source, cache_dir)
    manifest = os.path.join(folder, "manifest.json")
    if not force and os.path.exists(manifest):
        with open(manifest) as handle:
            if json.load(handle)["parser_version"] == PARSER_VERSION:
                report.status("cache: up to date ({})".format(source))
                return False

    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], games.GAME, data_dir, source)
    n_frames = n_frames or games.source_frames(source, games.GAME)
    os.makedirs(folder, exist_ok=True)

    # starter/baseline.py maps slots to teams by <Player> order; the spec says which player each
    # group of a raw line really is. v0 follows the starter; this says whether the two agree.
    slot_team = official_baseline.team_of_each_track(meta_path)
    spec_team, n_ball = atd.read_meta(meta_path)
    report.check("atd: raw-line slot order gives the same teams as <Player> order",
                 np.array_equal(slot_team, spec_team))

    report.status("atd: parsing the raw file ({})".format(source))
    players, ball = atd.parse_raw(raw_path, len(slot_team), n_ball, n_frames)
    np.save(os.path.join(folder, "atd_players.npy"), players)
    np.save(os.path.join(folder, "atd_ball.npy"), ball)
    np.save(os.path.join(folder, "slot_team.npy"), slot_team)

    matrices, reprojection_error, valid = official_homography.load(games.GAME, data_dir, source)
    np.savez(os.path.join(folder, "homography.npz"), matrices=matrices,
             reprojection_error=reprojection_error, valid=valid)
    if source == "tactical":
        with open(os.path.join(folder, "smart_tagging.json"), "w") as handle:
            json.dump(official_tagging.load(games.GAME, data_dir), handle)

    with open(manifest, "w") as handle:
        json.dump({"parser_version": PARSER_VERSION}, handle)
    report.wrote(folder)
    return True


@dataclass
class Atd:
    players: np.ndarray    # (n_frames, n_slots, 3): x, y normalised 0-1, speed m/s
    ball: np.ndarray       # (n_frames, n_ball_channels)
    slot_team: np.ndarray  # (n_slots,) HOME / AWAY


def load_atd(source="tactical", cache_dir=None):
    folder = source_dir(source, cache_dir)
    return Atd(*(np.load(os.path.join(folder, name + ".npy"), mmap_mode="r")
                 for name in ("atd_players", "atd_ball", "slot_team")))


def load_homography(source="tactical", cache_dir=None):
    with np.load(os.path.join(source_dir(source, cache_dir), "homography.npz")) as data:
        return data["matrices"], data["reprojection_error"], data["valid"]


def load_smart_tagging(cache_dir=None):
    with open(os.path.join(source_dir("tactical", cache_dir), "smart_tagging.json")) as handle:
        return json.load(handle)
