"""The submission pipeline: parsed cache in, one label per video frame out.

v2: a gradient-boosted classifier gives P(H, A, D) for every frame from a table of tracking
features (features.matrix, mtah/trees.py), and a second one P(no change, possession switch,
stop, restart within 0.2 s). A second stage re-estimates P(H, A, D) from the features and how
both outputs move over up to 8 s (features.second_stage). An input-output HMM decodes these into
piecewise-constant spells, with switches cheaper where the change detector expects them
(mtah/decoder.py). All trees in models/v2.json were fitted on off-game matches only
(harness/train_v2.py). The second half is graded from before its kickoff (DATA.md); graded
frames before a half's Smart Tagging "Kick Off (Start)" window are dead (before_kickoff()).

decode() is v1 (two logistic models, models/v1.json), which the harness keeps as the reference
v2 is measured against.
"""

import json
import os

import numpy as np

from mtah import config, decoder, emissions, features, frames, refine, trees
import games

VERSION = "v2.2"
MODEL = os.path.join(config.ROOT, "models", "v2.json")
LABELS = np.array(["H", "A", "D"])


def decode(f, model, kappa_dead, kappa_team):
    """H/A/D codes, the most likely path given the features."""
    log_trans, owner = decoder.chain(np.array(model["log_trans"]), model["mean_frames"],
                                     model["chain"])
    log_emit = emissions.log_likelihoods(f, model, kappa_dead, kappa_team)[:, owner]
    path = owner[decoder.viterbi(log_emit, log_trans)]
    return refine.boundaries(path, f) if model["refine"] else path


def decode_probs(probs, model, kappa, change=None, gamma=0.0):
    """H/A/D codes, the most likely path given a classifier's (n, 3) P(H, A, D | frame) and,
    if given, a change detector's (n, 4) probabilities (decoder.input_transitions)."""
    if change is None:
        log_trans, _ = decoder.chain(np.array(model["log_trans"]), model["mean_frames"], 1)
    else:
        log_trans = decoder.input_transitions(np.array(model["log_trans"]), model["mean_frames"],
                                              change, model["change_window"], gamma)
    return decoder.viterbi(emissions.tempered(probs, model["prior"], kappa), log_trans)


KICKOFF_CODE, KICKOFF_TYPE = "set pieces", "kick off (start)"  # as starter/load_smart_tagging.py
DEAD = np.array([0.0, 0.0, 1.0])  # P(H, A, D)


def before_kickoff(events, n_frames=frames.N_FRAMES):
    """Mask of the graded frames before their half's kickoff, where the ball is dead.

    The second half is graded from about two minutes before it kicks off (DATA.md). A half's
    kickoff is taken to lie in the earliest Smart Tagging "Kick Off (Start)" window that has not
    ended when the half's grading starts; the window is a few seconds around the restart, and
    inside it the decoder places the restart from the tracking. The first half is graded from
    its kickoff, so its window has begun already and nothing is masked (nor by a kickoff after a
    goal, should those be tagged alike).
    """
    windows = sorted((event["start_frame"], event["end_frame"]) for event in events
                     if (event["code"] or "").strip().lower() == KICKOFF_CODE
                     and (event["tags"].get("Type") or "").strip().lower() == KICKOFF_TYPE)
    mask = np.zeros(n_frames, dtype=bool)
    for start, stop in frames.graded_runs():
        first = next(begin for begin, end in windows if end > start and begin < stop)
        mask[start:max(start, first)] = True
    return mask


def predict(atd, events, pitch=games.GAMES[games.GAME]["pitch"]):
    """One label per frame from a cache.Atd and the Smart Tagging events."""
    with open(MODEL) as handle:
        model = json.load(handle)
    X, names = features.matrix(atd.players, atd.ball, atd.slot_team, pitch)
    X = X[:, [names.index(name) for name in model["names"]]]  # the columns the model was fitted on
    probs = trees.probabilities(X, model)
    change = trees.probabilities(X, model["detector"]) if "detector" in model else None
    if "second" in model:
        probs = trees.probabilities(features.second_stage(X, probs, change)[0], model["second"])
    probs[before_kickoff(events, len(probs))] = DEAD
    gamma = model.get("gamma", 0.0)
    return LABELS[decode_probs(probs, model, model["kappa"], change if gamma else None, gamma)]
