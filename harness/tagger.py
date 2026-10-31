"""A phase tagger like Smart Tagging, simulated on the harness, and its calibration (STRICT).

Smart Tagging exists only for BAR-ATH, so harness/fuse.py sets its weight in the decoder with a
simulated tagger on each held-out DFL match (phases()): one phase per possession of at least
min_s, missed with probability miss, each end moved out by extend_s (in, if negative: a phase
may spill past its possession or cover only part of it) and shifted by about LAG_S (normal, per
end), phases of one team less than merge_s apart merged into one, its team wrong with
probability flip.
Every draw is made per possession whatever the knobs (draws()), so that trials with different
knobs differ only by what the knobs do (common random numbers).

How coarse and how reliable Smart Tagging is, is measured without labels (statistics(), on the
graded frames after each kickoff): the share of frames with a vote, how long a run of one vote
lasts and the share of short runs, how often the voted team changes, and how often the
tracking-only path names the voted team. scripts/calibrate_tagger.py measures them on BAR-ATH
and fits the knobs (fit(), a grid search) until the simulated tagger on the DFL matches shows
the same; under STRICT the statistics and the knobs are written to cache/aggregates/ and never
shown, and calibrated() reads the knobs. The fit takes DFL's possessions to be like BAR-ATH's.
"""

import itertools
import json
import os
from dataclasses import dataclass, fields

import numpy as np

from mtah import config, frames, tagging
from evaluation import possession_metrics as pm

LAG_S = 1.0
SHORT_S = 5.0  # a run of one vote this short is short
GRID = {"min_s": (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0),
        "merge_s": (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, np.inf),  # inf: always
        "miss": (0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9),
        "extend_s": (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0),
        "flip": (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)}
CALIBRATED = os.path.join(config.CACHE_DIR, "aggregates", "tagger_params.json")


@dataclass(frozen=True)
class Knobs:
    min_s: float = 2.0
    merge_s: float = 2.0
    miss: float = 0.2
    extend_s: float = 0.0
    flip: float = 0.15


@dataclass(frozen=True)
class Draws:
    spells: list        # [(start, stop, team)] possessions
    miss: np.ndarray    # (n,) uniform
    shift: np.ndarray   # (n, 2) frames
    flip: np.ndarray    # (n,) uniform


def draws(labels, halves, rng):
    spells = pm.possession_spells(labels, halves)
    n = len(spells)
    return Draws(spells, rng.random(n),
                 np.round(rng.normal(0.0, LAG_S, (n, 2)) * frames.FPS).astype(int), rng.random(n))


def phases(d, knobs):
    """[(start, stop, team)] phases (see the module); a merged phase keeps its first flip."""
    out = []
    extend = int(round(knobs.extend_s * frames.FPS))
    for (start, stop, team), miss, shift, flip in zip(d.spells, d.miss, d.shift, d.flip):
        if stop - start < knobs.min_s * frames.FPS or miss < knobs.miss:
            continue
        start, stop = start - extend + shift[0], stop + extend + shift[1]
        if out and out[-1][2] == team and start - out[-1][1] < knobs.merge_s * frames.FPS:
            out[-1][1] = stop
        elif start < stop:
            out.append([start, stop, team, flip])
    return [(start, stop, tagging.A - team if flip < knobs.flip else team)
            for start, stop, team, flip in out]


def runs_of(mask):
    """Half-open [start, stop) spans of the True runs of mask."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2]))


def statistics(vote, path, runs):
    """{name: value} of a vote sequence over runs (see the module); path: the tracking-only
    H/A/D codes, for the agreement."""
    lengths, teams_changed, voted, total, right, counted = [], 0, 0, 0, 0, 0
    for start, stop in runs:
        v, p = vote[start:stop], path[start:stop]
        bounds = np.r_[0, np.flatnonzero(np.diff(v)) + 1, len(v)]
        values, sizes = v[bounds[:-1]], np.diff(bounds)
        teams = values[values != tagging.NONE]
        lengths.extend(sizes[values != tagging.NONE])
        teams_changed += int(np.sum(teams[1:] != teams[:-1]))
        voted += int(np.sum(v != tagging.NONE))
        total += stop - start
        alive = (v != tagging.NONE) & (p != pm.D)
        right += int(np.sum(p[alive] == v[alive]))
        counted += int(np.sum(alive))
    lengths = np.array(lengths) / frames.FPS
    return {"voted": voted / total, "run_s": float(lengths.mean()),
            "short_runs": float(np.mean(lengths < SHORT_S)),
            "switches_per_min": teams_changed / (total / frames.FPS / 60),
            "agreement": right / counted}


def mean_statistics(cases, knobs):
    """Mean over cases [(draws, n_frames, halves, paths)] of statistics(), the agreement
    averaged over each case's paths."""
    rows = []
    for d, n, halves, paths in cases:
        vote = tagging.votes(phases(d, knobs), n)
        per_path = [statistics(vote, path, halves) for path in paths]
        rows.append(dict(per_path[0], agreement=np.mean([s["agreement"] for s in per_path])))
    return {name: float(np.mean([row[name] for row in rows])) for name in rows[0]}


def fit(cases, target):
    """(Knobs, {knob: inside}): the point of GRID whose simulated statistics are nearest the
    target (sum of squared relative errors); inside is False for a knob at an end of its grid,
    unless that end is 0 or inf, the least or most the knob can be (never or always merge, no
    misses, no flips)."""
    best = None
    for values in itertools.product(*GRID.values()):
        knobs = Knobs(**dict(zip(GRID, values)))
        fitted = mean_statistics(cases, knobs)
        loss = sum(((fitted[name] - target[name]) / target[name]) ** 2 for name in target)
        if best is None or loss < best[0]:
            best = loss, knobs
    knobs = best[1]
    return knobs, {name: (GRID[name][0] < getattr(knobs, name) or GRID[name][0] == 0)
                   and (getattr(knobs, name) < GRID[name][-1] or GRID[name][-1] == np.inf)
                   for name in GRID}


def calibrated():
    """Knobs fitted to Smart Tagging by scripts/calibrate_tagger.py. STRICT: used, never
    printed."""
    with open(CALIBRATED) as handle:
        values = json.load(handle)
    return Knobs(**{field.name: values[field.name] for field in fields(Knobs)})
