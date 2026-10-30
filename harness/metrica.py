"""Metrica Sports sample data (games 1-3) in the BAR-ATH array format, labels from the events.

    python -m harness.metrica          # parse the three games in external/metrica/ once

github.com/metrica-sports/sample-data (DATASETS.md): three anonymised matches, 25 fps tracking
synchronised with Metrica's event data. Games 1 and 2 come as CSV, game 3 in the EPTS FIFA format
the ATD uses (one DataFormatSpecification per stretch between substitutions) with JSON events.
Positions are normalised 0-1 on a 105 x 68 m pitch, players keep their identities; there is no
speed channel (computed from positions here) and no ball height (0).

No per-frame possession is provided, so the labels come from the events, by Metrica's written
rule that possession changes only when the other team gains control (DATA.md):
  * in possession is the team of the latest PASS, CARRY, SET PIECE, SHOT or RECOVERY to have
    started -- a RECOVERY is the moment a team takes control; a CHALLENGE or a BALL LOST never
    changes possession by itself (a duel or a touch without control);
  * the ball is dead from a stoppage to the start of the next restart -- a SET PIECE, or a goal
    kick, which Metrica codes as a PASS or BALL LOST with subtype GOAL KICK -- provided no event of
    play lies between the two (a foul may be followed by advantage; a card, the foul's own
    CHALLENGE, or a RECOVERY without subtype, which Metrica logs when a team gets the dead ball
    back for its restart, is not play): a BALL OUT or a SHOT ending OUT or
    in a GOAL from its end frame (the ball crossing the line), a FAULT RECEIVED, a CHALLENGE with a
    FAULT or a ball lost by HAND BALL from its start, an OFFSIDE from its end (when the offside
    player plays the ball);
  * a half is labelled from its KICK OFF to the end of its last event; X elsewhere.
The halves are laid end to end with idsse.GAP frames between them, as the DFL matches are.
"""

import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from harness import idsse
from mtah import atd, config
from evaluation import possession_metrics as pm

RAW_DIR = os.path.join(config.ROOT, "external", "metrica")
ARRAY_DIR = os.path.join(RAW_DIR, "arrays")
GAMES = ("1", "2", "3")
PITCH = (105.0, 68.0)
FPS = 25
SPEED_LAG = 2  # frames either side of the central difference the speed is computed from
POSSESSION = {"PASS", "CARRY", "SET PIECE", "SHOT", "RECOVERY"}


def _speed(xy):
    """(n, ...) m/s from (n, ..., 2) normalised positions, central differences; NaN at the ends
    and wherever either end is missing."""
    speed = np.full(xy.shape[:-1], np.nan)
    step = (xy[2 * SPEED_LAG:] - xy[:-2 * SPEED_LAG]) * np.asarray(PITCH)
    speed[SPEED_LAG:-SPEED_LAG] = np.hypot(step[..., 0], step[..., 1]) * FPS / (2 * SPEED_LAG)
    return speed


def _with_speed(xy):
    return np.concatenate((xy, _speed(xy)[..., None]), axis=-1)


def _csv_tracking(game):
    """(periods (n,), players (n, persons, 2), slot_team, ball (n, 2)) of a CSV game."""
    parts, teams = [], []
    for side, team in (("Home", atd.HOME), ("Away", atd.AWAY)):
        table = pd.read_csv(os.path.join(RAW_DIR, "Sample_Game_%s_RawTrackingData_%s_Team.csv"
                                         % (game, side)), skiprows=2)
        values = table.to_numpy(dtype=float)
        players = values[:, 3:-2].reshape(len(values), -1, 2)
        parts.append(players)
        teams += [team] * players.shape[1]
        periods, ball = values[:, 0].astype(int), values[:, -2:]
    return periods, np.concatenate(parts, axis=1), np.array(teams, dtype=np.int8), ball


def _csv_events(game):
    """[(team, type, subtypes, period, start, end)] of a CSV game; frames 1-based."""
    table = pd.read_csv(os.path.join(RAW_DIR, "Sample_Game_%s_RawEventsData.csv" % game))
    return [(atd.HOME if row.Team == "Home" else atd.AWAY, row.Type,
             set(row.Subtype.split("-")) if isinstance(row.Subtype, str) else set(),
             int(row.Period), int(row["Start Frame"]), int(row["End Frame"]))
            for _, row in table.iterrows()]


def _epts_tracking():
    """(periods, players, slot_team, ball) of game 3, from its EPTS metadata and raw file."""
    root = ET.parse(os.path.join(RAW_DIR, "Sample_Game_3_metadata.xml")).getroot()
    home = root.find(".//Team").get("id")
    team = {player.get("id"): player.get("teamId") for player in root.iter("Player")}
    owner = {channel.get("id"): channel.get("playerId") for channel in root.iter("PlayerChannel")}
    persons = sorted(team, key=lambda person: (team[person] != home, person))
    parameters = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}
    n = int(parameters["second_half_end"])
    periods = np.where(np.arange(1, n + 1) <= int(parameters["first_half_end"]), 1, 2)
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        groups = spec.findall("SplitRegister")[0].findall("SplitRegister")
        columns = [persons.index(owner[group[0].get("playerChannelId")]) for group in groups]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), columns))
    players = np.full((n, len(persons), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    with open(os.path.join(RAW_DIR, "Sample_Game_3_tracking.txt")) as handle:
        for line in handle:
            frame_id, groups, ball_xy = line.strip().split(":")
            frame = int(frame_id)
            columns = next(c for start, stop, c in specs if start <= frame <= stop)
            players[frame - 1, columns] = [list(map(float, group.split(",")))
                                           for group in groups.split(";")]
            ball[frame - 1] = list(map(float, ball_xy.split(",")))
    slot_team = np.array([atd.HOME if team[p] == home else atd.AWAY for p in persons],
                         dtype=np.int8)
    return periods, players, slot_team, ball


def _json_events():
    """Game 3's events as _csv_events gives them."""
    with open(os.path.join(RAW_DIR, "Sample_Game_3_events.json")) as handle:
        data = json.load(handle)["data"]
    with open(os.path.join(RAW_DIR, "Sample_Game_3_metadata.xml")) as handle:
        home = ET.parse(handle).getroot().find(".//Team").get("id")

    def subtypes(value):
        values = value if isinstance(value, list) else [value] if value else []
        return {name for item in values for name in item["name"].split("/")}

    return [(atd.HOME if event["team"]["id"] == home else atd.AWAY, event["type"]["name"],
             subtypes(event["subtypes"]), int(event["period"]), int(event["start"]["frame"]),
             int(event["end"]["frame"])) for event in data]


def _stoppage_frame(kind, subtypes, start, end):
    """The frame a stoppage begins at, or None if the event is not one."""
    if "OFFSIDE" in subtypes:
        return end
    if kind == "BALL OUT" or (kind == "SHOT" and subtypes & {"OUT", "GOAL"}):
        return end
    if (kind == "FAULT RECEIVED" or (kind == "CHALLENGE" and "FAULT" in subtypes)
            or (kind == "BALL LOST" and "HAND BALL" in subtypes)):
        return start
    return None


def _play(kind, subtypes):
    """Whether an event between a stoppage and a restart means play went on."""
    return not (kind == "CARD" or (kind == "RECOVERY" and not subtypes)
                or (kind == "CHALLENGE" and "FAULT" in subtypes))


def _restart(kind, subtypes):
    return kind == "SET PIECE" or "GOAL KICK" in subtypes


def event_labels(events, n):
    """(labels (n,) codes, halves [(start, stop)]) on 0-based frames from 1-based events."""
    labels = np.full(n, pm.X, dtype=np.int8)
    halves = []
    for period in sorted({event[3] for event in events}):
        events_p = sorted((e for e in events if e[3] == period), key=lambda e: e[4])
        kickoff = next(e[4] for e in events_p if e[1] == "SET PIECE" and "KICK OFF" in e[2]) - 1
        stop = min(max(e[5] for e in events_p), n)
        team = np.full(n, -1)
        for side, kind, _, _, start, _ in events_p:
            if kind in POSSESSION:
                team[start - 1:] = side
        half = np.where(team[kickoff:stop] >= 0, team[kickoff:stop], pm.D).astype(np.int8)
        for i, (_, kind, subtypes, _, start, end) in enumerate(events_p):
            frame = _stoppage_frame(kind, subtypes, start, end)
            if frame is None:
                continue
            later = [e for e in events_p[i + 1:] if e[4] > frame and _play(e[1], e[2])]
            if later and _restart(later[0][1], later[0][2]):
                half[max(frame - 1 - kickoff, 0):max(later[0][4] - 1 - kickoff, 0)] = pm.D
        labels[kickoff:stop] = half
        halves.append((kickoff, stop))
    return labels, halves


def parse(game):
    """One game as an idsse.Match, the second half moved idsse.GAP frames on."""
    periods, players, slot_team, ball = _csv_tracking(game) if game != "3" else _epts_tracking()
    labels, halves = event_labels(_csv_events(game) if game != "3" else _json_events(),
                                  len(periods))
    second = int(np.argmax(periods == 2))
    rows = np.where(np.arange(len(periods)) < second, np.arange(len(periods)),
                    np.arange(len(periods)) + idsse.GAP)
    n = len(periods) + idsse.GAP
    out_players = np.full((n, players.shape[1], 3), np.nan)
    out_ball = np.full((n, 3), np.nan)
    out_labels = np.full(n, pm.X, dtype=np.int8)
    for part in (slice(0, second), slice(second, len(periods))):
        out_players[rows[part]] = _with_speed(players[part])
        out_ball[rows[part]] = _with_speed(ball[part])
        out_labels[rows[part]] = labels[part]
    halves = [(rows[start], rows[stop - 1] + 1) for start, stop in halves]
    return idsse.Match("metrica" + game, out_players, slot_team, out_ball, np.zeros(n),
                       out_labels, np.array(halves), PITCH)


def load(game):
    with np.load(os.path.join(ARRAY_DIR, "metrica%s.npz" % game)) as data:
        return idsse.Match("metrica" + game, data["players"], data["slot_team"], data["ball"],
                           data["ball_z"], data["labels"], data["halves"], tuple(data["pitch"]))


if __name__ == "__main__":
    os.makedirs(ARRAY_DIR, exist_ok=True)
    for game in GAMES:
        match = parse(game)
        np.savez(os.path.join(ARRAY_DIR, match.name + ".npz"), players=match.players,
                 slot_team=match.slot_team, ball=match.ball, ball_z=match.ball_z,
                 labels=match.labels, halves=match.halves, pitch=np.array(match.pitch))
        print("parsed", match.name)
