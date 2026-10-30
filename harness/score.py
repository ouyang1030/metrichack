"""Score methods on the off-game harness with the official metric (AGENTS.md 5.1, 5.5).

A method maps a match -- players, ball, slot_team, pitch: the cache.Atd fields plus the pitch --
to one label per frame, the same contract mtah.pipeline.predict has on BAR-ATH. Each half is
scored on its own by possession_metrics.evaluate(). A method's result is the mean over halves
with a bootstrap interval; two methods are compared with a paired bootstrap over the same halves.

    python -m harness.score            # the four published baselines, clean and degraded
"""

import numpy as np

from harness import degrade, idsse
from mtah import baseline
from evaluation import possession_metrics as pm

SEEDS = (0, 1, 2)
N_BOOT = 2000
METRICS = ("spell_f1_mean", "macro_f1")


def score_halves(method, match):
    """(n_halves, 2) [spellF1, macroF1] of one method on one match."""
    return half_scores(match.labels, pm.label_codes(np.asarray(method(match))), match.halves)


def half_scores(labels, pred, halves):
    """(n_halves, 2) [spellF1, macroF1] of predicted codes against label codes."""
    return np.array([[pm.evaluate(labels[a:b], pred[a:b])[key] for key in METRICS]
                     for a, b in halves])


def run(methods, matches=None, seeds=SEEDS):
    """{name: (clean, degraded)} with one row per half; degraded rows average the seeds."""
    clean = {name: [] for name in methods}
    degraded = {name: [] for name in methods}
    for match_id in matches or idsse.match_ids():
        match = idsse.load(match_id)
        noisy = [degrade.degrade(match, seed) for seed in seeds]
        for name, method in methods.items():
            clean[name].append(score_halves(method, match))
            degraded[name].append(np.mean([score_halves(method, m) for m in noisy], axis=0))
    return {name: (np.concatenate(clean[name]), np.concatenate(degraded[name]))
            for name in methods}


def spell_recall(labels, pred, halves, short_s=4.0, iou=pm.SPELL_IOU_HEADLINE):
    """(2, 2) [[matched, total] of possessions shorter than short_s, [same] of the longer ones]."""
    counts = np.zeros((2, 2), dtype=int)
    for a, b in halves:
        gt = pm.possession_spells(labels[a:b], [(0, b - a)])
        overlaps = pm.spell_overlaps(gt, pm.possession_spells(pred[a:b], [(0, b - a)]))
        matched = pm.matched_gt_flags(overlaps, len(gt), iou)
        longer = np.array([stop - start >= short_s * pm.FPS for start, stop, _ in gt], dtype=int)
        np.add.at(counts, (longer, 0), matched)
        np.add.at(counts, (longer, 1), 1)
    return counts


def interval(rows, seed=0):
    """(mean, low, high) per column; the interval is the 5-95% bootstrap over rows (halves)."""
    draws = np.random.default_rng(seed).integers(0, len(rows), (N_BOOT, len(rows)))
    low, high = np.percentile(rows[draws].mean(axis=1), pm.BOOTSTRAP_PERCENTILES, axis=0)
    return rows.mean(axis=0), low, high


def table(results):
    lines = ["{:<22} {:>27} {:>27}".format("", "clean spellF1 / macroF1",
                                            "degraded spellF1 / macroF1")]
    for name, (clean, degraded) in results.items():
        cells = []
        for rows in (clean, degraded):
            mean, low, high = interval(rows)
            cells.append("{:.3f} [{:.3f},{:.3f}] / {:.3f}".format(mean[0], low[0], high[0],
                                                                   mean[1]))
        lines.append("{:<22} {:>27} {:>27}".format(name, *cells))
    return "\n".join(lines)


def baselines():
    def variant(window):
        return lambda m: baseline.smooth(
            baseline.nearest_player_codes(m.players, m.ball, m.slot_team, m.pitch), window)
    return {name: variant(window) for name, window in baseline.VARIANTS}


if __name__ == "__main__":
    print(table(run(baselines())))
