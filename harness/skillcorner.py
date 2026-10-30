"""SkillCorner open data (10 A-League matches) in the BAR-ATH array format: real broadcast noise.

    python -m harness.skillcorner      # parse every match in external/skillcorner/ once

github.com/SkillCorner/opendata (DATASETS.md), read from external/skillcorner/<match id>/ and
never written to; arrays go to ARRAY_DIR. Broadcast tracking at 10 fps: players the camera does
not show are extrapolated by SkillCorner, and only some of them are given; the ball is kept
where SkillCorner detected it, NaN elsewhere -- noise as it is, not simulated, so these matches
are scored as they are, never degraded. Positions are interpolated to 25 fps (linearly, across
one 10 fps step at most) and speeds computed from them.

Labels, from SkillCorner's own automatic outputs (no human labels): alive inside a phase of play
(the phases file: "phases of play are only defined when the ball is in play"), dead outside one;
while alive, the team SkillCorner's tracking names in possession (the possession group), carried
over while it names none (a ball in flight or contested stays with the last team), and the
phase's team in possession before the first named one. A half runs from its first alive frame
to its last frame.
"""

import glob
import json
import os

import numpy as np
import pandas as pd

from harness import idsse
from mtah import atd, config
from evaluation import possession_metrics as pm

RAW_DIR = os.path.join(config.ROOT, "external", "skillcorner")
ARRAY_DIR = os.path.join(config.ROOT, "external", "skillcorner-arrays")
SOURCE_FPS, FPS = 10, 25
SPEED_LAG = 2  # frames (25 fps) either side of the central difference for speeds


def match_ids():
    return sorted(os.path.basename(path) for path in glob.glob(os.path.join(RAW_DIR, "*"))
                  if os.path.isdir(path))


def _resample(values, source_frames):
    """(k, ...) values at 10 fps frames source_frames (consecutive) -> 25 fps by linear
    interpolation between neighbouring 10 fps frames; NaN where either neighbour is NaN."""
    t = np.arange(int((len(source_frames) - 1) * FPS / SOURCE_FPS) + 1) * SOURCE_FPS / FPS
    low = np.floor(t).astype(int)
    high = np.minimum(low + 1, len(source_frames) - 1)
    weight = (t - low).reshape((-1,) + (1,) * (values.ndim - 1))
    return values[low] * (1 - weight) + values[high] * weight, np.round(t).astype(int)


def _speed(xy, pitch):
    speed = np.full(xy.shape[:-1], np.nan)
    step = (xy[2 * SPEED_LAG:] - xy[:-2 * SPEED_LAG]) * np.asarray(pitch)
    speed[SPEED_LAG:-SPEED_LAG] = np.hypot(step[..., 0], step[..., 1]) * FPS / (2 * SPEED_LAG)
    return speed


def parse(match_id):
    folder = os.path.join(RAW_DIR, match_id)
    with open(os.path.join(folder, match_id + "_match.json")) as handle:
        meta = json.load(handle)
    pitch = (float(meta["pitch_length"]), float(meta["pitch_width"]))
    home = meta["home_team"]["id"]
    persons = sorted((player["team_id"] != home, player["id"]) for player in meta["players"])
    slot = {person: i for i, (_, person) in enumerate(persons)}
    group_team = {"home team": atd.HOME, "away team": atd.AWAY}

    with open(os.path.join(folder, match_id + "_tracking_extrapolated.jsonl")) as handle:
        rows = [json.loads(line) for line in handle]
    rows = [row for row in rows if row["period"] is not None]
    phases = pd.read_csv(os.path.join(folder, match_id + "_phases_of_play.csv"))
    halves_out, parts, starts = [], [], []
    offset = 0
    for period in (1, 2):
        part = [row for row in rows if row["period"] == period]
        frames = np.array([row["frame"] for row in part])
        players = np.full((len(part), len(persons), 2), np.nan)
        ball = np.full((len(part), 2), np.nan)
        group = np.full(len(part), -1)
        for i, row in enumerate(part):
            for player in row["player_data"]:
                players[i, slot[player["player_id"]]] = (player["x"], player["y"])
            if row["ball_data"]["is_detected"]:
                ball[i] = (row["ball_data"]["x"], row["ball_data"]["y"])
            group[i] = group_team.get(row["possession"]["group"], -1)
        alive = np.zeros(len(part), dtype=bool)
        phase_team = np.full(len(part), -1)
        for _, phase in phases[phases.period == period].iterrows():
            inside = (frames >= phase.frame_start) & (frames <= phase.frame_end)
            alive |= inside
            phase_team[inside] = atd.HOME if phase.team_in_possession_id == home else atd.AWAY
        labels = np.full(len(part), pm.D, dtype=np.int8)
        team = -1
        for i in range(len(part)):
            if not alive[i]:
                team = -1
                continue
            team = group[i] if group[i] >= 0 else team if team >= 0 else phase_team[i]
            labels[i] = team
        scale = np.array(pitch)
        players25, nearest = _resample(players / scale + 0.5, frames)
        ball25, _ = _resample(ball / scale + 0.5, frames)
        labels25 = labels[nearest]
        first = int(np.argmax(labels25 != pm.D))
        parts.append((np.concatenate((players25, _speed(players25, pitch)[..., None]), axis=2),
                      np.concatenate((ball25, _speed(ball25, pitch)[:, None]), axis=1),
                      labels25))
        starts.append(offset)
        halves_out.append((offset + first, offset + len(labels25)))
        offset += len(labels25) + idsse.GAP
    n = offset - idsse.GAP
    players = np.full((n, len(persons), 3), np.nan)
    ball = np.full((n, 3), np.nan)
    labels = np.full(n, pm.X, dtype=np.int8)
    for start, (part_players, part_ball, part_labels), (first, stop) in zip(starts, parts,
                                                                           halves_out):
        players[start:stop], ball[start:stop] = part_players, part_ball
        labels[first:stop] = part_labels[first - start:]  # X before the kickoff
    return idsse.Match("skillcorner" + match_id, players,
                       np.array([away for away, _ in persons], dtype=np.int8), ball,
                       np.zeros(n), labels, np.array(halves_out), pitch)


def load(match_id):
    with np.load(os.path.join(ARRAY_DIR, "skillcorner%s.npz" % match_id)) as data:
        return idsse.Match("skillcorner" + match_id, data["players"], data["slot_team"],
                           data["ball"], data["ball_z"], data["labels"], data["halves"],
                           tuple(data["pitch"]))


if __name__ == "__main__":
    os.makedirs(ARRAY_DIR, exist_ok=True)
    for match_id in match_ids():
        match = parse(match_id)
        np.savez(os.path.join(ARRAY_DIR, match.name + ".npz"), players=match.players,
                 slot_team=match.slot_team, ball=match.ball, ball_z=match.ball_z,
                 labels=match.labels, halves=match.halves, pitch=np.array(match.pitch))
        print("parsed", match.name, flush=True)
