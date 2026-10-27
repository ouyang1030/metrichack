"""Parse Metrica's FIFA/EPTS tracking export (the ATD) into dense arrays.

A raw line, as DATA.md documents it and starter/baseline.py reads it:

    <frame_id>:<x,y,speed>;<x,y,speed>;...;<x,y,speed>:<ball channels>

one x,y,speed group per track slot, NaN for a missing value. Array rows are video frames,
frame_id - 1. The metadata XML's DataFormatSpecification names the player behind each group.
"""

import xml.etree.ElementTree as ET

import numpy as np

HOME, AWAY = 0, 1
X, Y, SPEED = 0, 1, 2


def read_meta(meta_path):
    """(slot_team, n_ball_channels) from the metadata XML.

    slot_team is the team code of each group of a raw line, in the order the
    DataFormatSpecification declares the groups. The first <Team> is home (FIFA/EPTS).
    """
    root = ET.parse(meta_path).getroot()
    home = root.find(".//Team").get("id")
    team = {player.get("id"): player.get("teamId") for player in root.iter("Player")}
    owner = {channel.get("id"): channel.get("playerId") for channel in root.iter("PlayerChannel")}
    slots, ball = root.find(".//DataFormatSpecification").findall("SplitRegister")
    slot_team = [HOME if team[owner[group[0].get("playerChannelId")]] == home else AWAY
                 for group in slots]
    return np.array(slot_team, dtype=np.int8), len(ball)


def parse_raw(raw_path, n_slots, n_ball, n_frames):
    """(players (n_frames, n_slots, 3), ball (n_frames, n_ball)) as float64, NaN where missing.

    Every token goes through float(), so the doubles are exactly the ones starter/baseline.py
    computes with. Frames the file does not cover stay NaN.
    """
    players = np.full((n_frames, n_slots * 3), np.nan)
    ball = np.full((n_frames, n_ball), np.nan)
    with open(raw_path) as handle:
        for line in handle:
            frame_id, slots, ball_channels = line.split(":")
            row = int(frame_id) - 1
            players[row] = list(map(float, slots.replace(";", ",").split(",")))
            ball[row] = list(map(float, ball_channels.split(",")))
    return players.reshape(n_frames, n_slots, 3), ball
