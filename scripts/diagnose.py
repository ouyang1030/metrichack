"""Unsupervised diagnostics of our predictions on BAR-ATH (AGENTS.md 5.3), behind the firewall.

    python scripts/diagnose.py

No labels are involved: these catch bugs, they never choose between variants. For the
pipeline's output on the cached ATD:

  * the summary of the predictions (possessions, durations, short spells, dead share);
  * the graded stretch before the second-half kickoff, dead in the ground truth (DATA.md): the
    share we predict D;
  * agreement with Smart Tagging's team tags: inside attacking phases, the share of the frames we
    call alive that we give to the tagged team; inside defending phases, to the other team. The
    first and last second of each phase are left out (its edges are the least certain);
  * stability: the ball removed on about STABILITY_SHARE more of its seen graded frames, in runs
    of 0.5 s on average, and the pipeline run again: the share of graded frames whose label
    changes, the change in the number of possessions, and the spellF1 of the perturbed run scored
    against the original.
"""

import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah import cache, frames, pipeline, safe_run  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402
from starter import baseline as official_baseline  # noqa: E402
from starter import load_smart_tagging as official_tagging  # noqa: E402
import games  # noqa: E402

ATTACKING = ("build up", "progression", "creating chances", "ball in final third",
             "ball in the box", "players in the box", "long ball", "attacking transition")
DEFENDING = ("defending in", "defensive transition")  # prefixes
EDGE_S = 1.0
STABILITY_SHARE = 0.10
STABILITY_RUN_S = 0.5
SEED = 0


def _words(name):
    return {word for word in re.findall(r"[a-z]+", name.lower()) if len(word) > 2}


def team_sides(values, home, away):
    """{Team tag value: "H" or "A"}, by the words it shares with the metadata's club names; None
    if any value matches both or neither."""
    sides = {}
    for value in values:
        shared = [len(_words(value) & _words(home)), len(_words(value) & _words(away))]
        if shared[0] == shared[1]:
            return None
        sides[value] = "H" if shared[0] > shared[1] else "A"
    return sides


def tagging_agreement(labels, events, sides):
    """(attacking, defending) agreement of labels with the tagged team, as shares of frames."""
    edge = int(EDGE_S * frames.FPS)
    graded = frames.graded_mask(len(labels))
    other = {"H": "A", "A": "H"}
    hits = {"attacking": [0, 0], "defending": [0, 0]}
    for event in events:
        code, team = (event["code"] or "").strip().lower(), event["tags"].get("Team")
        if team not in sides:
            continue
        kind = ("attacking" if code in ATTACKING
                else "defending" if code.startswith(DEFENDING) else None)
        if kind is None:
            continue
        expected = sides[team] if kind == "attacking" else other[sides[team]]
        span = slice(event["start_frame"] + edge, event["end_frame"] - edge)
        alive = graded[span] & (labels[span] != "D")
        hits[kind][0] += int(np.sum(labels[span][alive] == expected))
        hits[kind][1] += int(np.sum(alive))
    return tuple(right / total for right, total in hits.values())


def perturbed(atd, share, seed):
    """The ATD with the ball removed on about share more of its seen graded frames, in runs."""
    ball = np.array(atd.ball)
    rng = np.random.default_rng(seed)
    seen = np.flatnonzero(~np.isnan(ball[:, 0]) & frames.graded_mask(len(ball)))
    mean = STABILITY_RUN_S * frames.FPS
    for start in rng.choice(seen, int(round(share * len(seen) / mean)), replace=False):
        ball[start:start + 1 + int(rng.exponential(mean - 1))] = np.nan
    return cache.Atd(atd.players, ball, atd.slot_team)


def main(report):
    cache.build(report)
    atd, events = cache.load_atd(), cache.load_smart_tagging()
    labels = pipeline.predict(atd, events)
    summary = report.predictions("pipeline {}".format(pipeline.VERSION), labels)

    second_half = frames.graded_runs()[1][0]
    kickoff = official_tagging.KICKOFF_VIDEO_FRAMES["2nd Half"]  # published (DATA.md, FAQ.md)
    report.diagnostic("share D before the second-half kickoff",
                      np.mean(labels[second_half:kickoff] == "D"))

    meta_path = games.require(["atd_meta"])[0]
    values = sorted({event["tags"].get("Team") for event in events} - {None, "N/A"})
    sides = team_sides(values, *official_baseline.team_names(meta_path))
    if report.check("Smart Tagging team names map one to one onto home and away",
                    sides is not None and sorted(sides.values()) == ["A", "H"]):
        attacking, defending = tagging_agreement(labels, events, sides)
        report.diagnostic("agreement with the tagged team in attacking phases", attacking)
        report.diagnostic("agreement with the other team in defending phases", defending)

    report.status("pipeline again with the ball removed on more frames")
    again = pipeline.predict(perturbed(atd, STABILITY_SHARE, SEED), events)
    graded = frames.graded_mask(len(labels))
    report.diagnostic("stability: share of graded frames whose label changes",
                      np.mean(again[graded] != labels[graded]))
    report.diagnostic("stability: possessions after / before",
                      report.predictions("perturbed", again)["n_possessions"]
                      / summary["n_possessions"])
    report.diagnostic("stability: spellF1 of the perturbed run against the original",
                      pm.spell_scores(pm.label_codes(labels), pm.label_codes(again),
                                      frames.graded_runs())["mean_f1"])


if __name__ == "__main__":
    sys.exit(safe_run.run(main, name="diagnose"))
