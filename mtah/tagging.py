"""Smart Tagging as evidence for the decoder (AGENTS.md section 7).

Smart Tagging's phases name a team: attacking phases the team in possession, defending phases
the other one; our tracking-only predictions agree with both directions (EXPERIMENTS.md, 10-29).
votes() turns them into one vote per frame -- H, A or none -- leaving out the first and last
EDGE_S of each phase, where a phase is least certain, and frames on which phases disagree.

How far a vote can be trusted is estimated on the match itself: confusion() fits P(vote | H, A,
D) by expectation-maximisation, with the frame classifier's probabilities standing in for the
unknown states and the vote taken as a second observation, independent of the tracking given
the state. No label is involved; the estimate is symmetric in the two teams. log_evidence() is
log P(vote | state) per frame; the decoder adds it to its log emissions, tempered by a fixed
weight that was set on the harness with simulated taggers (harness/fuse.py), because a phase's
frames are no more independent than the classifier's.
"""

import re

import numpy as np

from mtah import frames

ATTACKING = ("build up", "progression", "creating chances", "ball in final third",
             "ball in the box", "players in the box", "long ball", "attacking transition")
DEFENDING = ("defending in", "defensive transition")  # prefixes
EDGE_S = 1.0
H, A, NONE = 0, 1, 2  # vote codes; H and A as the label codes
EM_ITERATIONS = 100
START = np.array([[0.6, 0.1, 0.3], [0.1, 0.6, 0.3], [0.1, 0.1, 0.8]])  # P(vote | state)


def _words(name):
    return {word for word in re.findall(r"[a-z]+", name.lower()) if len(word) > 2}


def team_sides(values, home, away):
    """{Team tag value: H or A}, by the words it shares with the metadata's club names; None if
    any value matches both or neither."""
    sides = {}
    for value in values:
        shared = [len(_words(value) & _words(home)), len(_words(value) & _words(away))]
        if shared[0] == shared[1]:
            return None
        sides[value] = H if shared[0] > shared[1] else A
    return sides


def tag_values(events):
    """The Team tag values that name a team."""
    return sorted({event["tags"].get("Team") for event in events} - {None, "N/A"})


def phases(events, sides):
    """[(start_frame, end_frame, team in possession)] of the attacking and defending phases."""
    out = []
    for event in events:
        code, team = (event["code"] or "").strip().lower(), event["tags"].get("Team")
        if team not in sides:
            continue
        if code in ATTACKING:
            out.append((event["start_frame"], event["end_frame"], sides[team]))
        elif code.startswith(DEFENDING):
            out.append((event["start_frame"], event["end_frame"], A - sides[team]))
    return out


def votes(spans, n):
    """(n,) vote codes from [(start, end, team)] spans: the team of the spans covering a frame,
    EDGE_S in from their ends; NONE where none covers it or they disagree."""
    edge = int(EDGE_S * frames.FPS)
    count = np.zeros((n + 1, 2), dtype=int)
    for start, end, team in spans:
        start, end = max(start + edge, 0), min(end - edge, n)
        if start < end:
            count[start, team] += 1
            count[end, team] -= 1
    covered = np.cumsum(count, axis=0)[:n] > 0
    return np.where(covered[:, 0] & ~covered[:, 1], H,
                    np.where(covered[:, 1] & ~covered[:, 0], A, NONE)).astype(np.int8)


def _symmetric(table):
    """table (states H, A, D x votes H, A, NONE) averaged with itself under the team swap."""
    swap = [A, H, 2]
    return (table + table[swap][:, swap]) / 2


def confusion(probs, vote, mask):
    """(3, 3) P(vote | state) for states H, A, D, by EM over the frames in mask: probs (n, 3) is
    the classifier's P(H, A, D | tracking) per frame."""
    p, v = probs[mask], vote[mask]
    one_hot = np.eye(3)[v]
    table = START.copy()
    for _ in range(EM_ITERATIONS):
        posterior = p * table[:, v].T
        posterior /= posterior.sum(axis=1, keepdims=True)
        table = _symmetric(posterior.T @ one_hot)
        table /= table.sum(axis=1, keepdims=True)
    return table


def log_evidence(table, vote):
    """(n, 3) log P(vote_t | state) for states H, A, D."""
    return np.log(np.clip(table[:, vote].T, 1e-12, None))
