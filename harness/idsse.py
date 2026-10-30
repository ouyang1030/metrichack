"""IDSSE / DFL open data (Bassek et al. 2025, CC BY 4.0) in the BAR-ATH array format.

    python -m harness.idsse            # parse every match in external/idsse/ once

Seven Bundesliga matches of TRACAB tracking at 25 Hz. Every ball frame carries BallStatus (0 dead,
1 alive) and BallPossession (1 home, 2 away, "the player in control of the ball") -- the closest
analogue to how the BAR-ATH ground truth was derived from corrected tracking. Label: D where the
ball is dead, else the team in possession.

Per match, external/idsse/arrays/<match>.npz:
    players    (n, n_persons, 3)  x, y normalised 0-1, speed m/s; NaN while off the pitch
    slot_team  (n_persons,)       HOME / AWAY; one slot per person, identities intact (clean)
    ball       (n, 3)             x, y normalised 0-1, speed m/s
    ball_z     (n,)               height in metres
    labels     (n,)               official label codes H, A, D; X in the gap between halves
    halves     (2, 2)             [start, stop) of each half
    pitch      (2,)               length, width in metres
The two halves are laid end to end with GAP empty frames between them, about as long as
BAR-ATH's halftime, so that nothing estimated over a few minutes mixes the two halves.
Referees are left out; the ATD tracks players and the ball only.
"""

import glob
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from mtah import config  # noqa: F401  (config puts the repository root on sys.path)
from evaluation import possession_metrics as pm

RAW_DIR = os.path.join(config.ROOT, "external", "idsse")
ARRAY_DIR = os.path.join(RAW_DIR, "arrays")
SECTIONS = ("firstHalf", "secondHalf")
GAP = 15 * 60 * 25  # 15 minutes at 25 fps
KMH = 1 / 3.6


@dataclass
class Match:
    name: str
    players: np.ndarray
    slot_team: np.ndarray
    ball: np.ndarray
    ball_z: np.ndarray
    labels: np.ndarray
    halves: np.ndarray
    pitch: tuple


def _raw_path(kind, match_id):
    return glob.glob(os.path.join(RAW_DIR, "DFL_{}_*_DFL-MAT-{}.xml".format(kind, match_id)))[0]


def _framesets(path, team_ids):
    """(section, team, person, values) per FrameSet of the given teams and the ball.

    values is (k, 4) [N, X, Y, S] for a person, (k, 7) [N, X, Y, S, Z, possession, status] for
    the ball.
    """
    out = []
    for _, element in ET.iterparse(path):
        if element.tag != "FrameSet":
            continue
        team = element.get("TeamId")
        if team in team_ids or team == "BALL":
            keys = ("N", "X", "Y", "S") + (("Z", "BallPossession", "BallStatus")
                                           if team == "BALL" else ())
            values = np.array([[float(frame.get(key)) for key in keys] for frame in element])
            out.append((element.get("GameSection"), team, element.get("PersonId"), values))
        element.clear()
    return out


def parse(match_id):
    """Read one match's DFL XML files into a Match."""
    info = ET.parse(_raw_path("02_01_matchinformation", match_id)).getroot()
    general, environment = info.find(".//General"), info.find(".//Environment")
    home, guest = general.get("HomeTeamId"), general.get("GuestTeamId")
    length, width = float(environment.get("PitchX")), float(environment.get("PitchY"))

    framesets = _framesets(_raw_path("04_03_positions_raw_observed", match_id), (home, guest))
    ball = {section: values for section, team, _, values in framesets if team == "BALL"}
    first = {section: int(ball[section][0, 0]) for section in SECTIONS}
    sizes = [len(ball[section]) for section in SECTIONS]
    offsets = dict(zip(SECTIONS, (0, sizes[0] + GAP)))
    n = sizes[0] + GAP + sizes[1]

    persons = sorted({(team != home, person) for _, team, person, _ in framesets
                      if team != "BALL"})
    slot = {person: index for index, (_, person) in enumerate(persons)}
    players = np.full((n, len(persons), 3), np.nan)
    ball_xys = np.full((n, 3), np.nan)
    ball_z = np.full(n, np.nan)
    labels = np.full(n, pm.X, dtype=np.int8)

    for section, team, person, values in framesets:
        rows = values[:, 0].astype(int) - first[section] + offsets[section]
        xys = np.column_stack((values[:, 1] / length + 0.5, values[:, 2] / width + 0.5,
                               values[:, 3] * KMH))
        if team == "BALL":
            ball_xys[rows] = xys
            ball_z[rows] = values[:, 4]
            labels[rows] = np.where(values[:, 6] == 0, pm.D,
                                    np.where(values[:, 5] == 1, pm.H, pm.A))
        else:
            players[rows, slot[person]] = xys

    halves = np.array([(offsets[s], offsets[s] + size) for s, size in zip(SECTIONS, sizes)])
    return Match(match_id, players, np.array([away for away, _ in persons], dtype=np.int8),
                 ball_xys, ball_z, labels, halves, (length, width))


def match_ids():
    return sorted(path.rsplit("DFL-MAT-", 1)[1][:-4] for path in
                  glob.glob(os.path.join(RAW_DIR, "DFL_04_03_positions_raw_observed_*.xml")))


def save(match):
    os.makedirs(ARRAY_DIR, exist_ok=True)
    np.savez(os.path.join(ARRAY_DIR, match.name + ".npz"), players=match.players,
             slot_team=match.slot_team, ball=match.ball, ball_z=match.ball_z,
             labels=match.labels, halves=match.halves, pitch=np.array(match.pitch))


def load(match_id):
    with np.load(os.path.join(ARRAY_DIR, match_id + ".npz")) as data:
        return Match(match_id, data["players"], data["slot_team"], data["ball"], data["ball_z"],
                     data["labels"], data["halves"], tuple(data["pitch"]))


def load_all():
    return [load(match_id) for match_id in match_ids()]


if __name__ == "__main__":
    for match_id in match_ids():
        if not os.path.exists(os.path.join(ARRAY_DIR, match_id + ".npz")):
            save(parse(match_id))
        print("parsed", match_id)
