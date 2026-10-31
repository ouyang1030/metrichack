"""Where the decoded possessions go wrong, from the saved out-of-fold outputs (EXPERIMENTS.md).

    python -m harness.segments         # after python -m harness.train_v2

For the accepted v2 (train_v2.OOF_DIR), decoded at its setting (models/v2.json), on the degraded
copies: every true possession of each label set by length, and what became of it
(score.outcomes) --
  found     matched one to one at the headline IoU (possession_metrics.SPELL_IOU_HEADLINE);
  merged    a predicted spell of its team covers half of it and also half of another true
            possession of that team;
  split     two or more predicted spells of its team cover it, none of them half of it;
  missed    no predicted spell of its team covers half of it;
  shifted   the rest (covered, but not matched).
Each possession is also split by whether the degraded copy's ball is seen on at least half of
its frames: whether what is missed lacks the ball or lacks the reading of it.
Then the segments that separate two true possessions of one team -- a spell of the other team or
a stoppage, by length -- and the share of them the decoder reproduces (some frames of that other
team, or dead, inside them): one it misses merges the two. And the predicted / true possession
counts. Nothing here touches BAR-ATH.
"""

import json
import os

import numpy as np

from harness import degrade, idsse, score, semantics, train_v1, train_v2
from evaluation import possession_metrics as pm

COPIES = (1, 2, 3)  # the saved outputs of the degraded copies, seeds train_v1.SEEDS


def run():
    with open(train_v2.MODEL_PATH) as handle:
        exported = json.load(handle)
    setting = {key: exported[key] for key in ("kappa", "gamma")}
    results = {name: ([], [], [0, 0]) for name in train_v2.LABEL_SETS}
    for match_id in idsse.match_ids():
        match = idsse.load(match_id)
        with np.load(os.path.join(train_v2.OOF_DIR, match_id + ".npz")) as saved:
            probs, change, model = saved["probs"], saved["change"], json.loads(str(saved["model"]))
        for copy, seed in zip(COPIES, train_v1.SEEDS):
            pred = train_v2.decode(probs[copy], change[copy], model, **setting)
            seen = ~np.isnan(degrade.degrade(match, seed, train_v2.NOISE).ball[:, 0])
            for name, control_s in train_v2.LABEL_SETS.items():
                labels = match.labels if control_s is None else semantics.metrica(
                    match.labels, match.halves, control_s)
                rows, separators = score.outcomes(labels, pred, match.halves, seen)
                results[name][0].extend(rows)
                results[name][1].extend(separators)
                results[name][2][0] += len(pm.possession_spells(pred, match.halves))
                results[name][2][1] += len(pm.possession_spells(labels, match.halves))
    return results


def report(results):
    bins = score.BINS
    names = (["<%g s" % bins[1]] + ["%g-%g s" % pair for pair in zip(bins[1:-2], bins[2:-1])]
             + [">=%g s" % bins[-2]])
    for label_set, (rows, separators, (predicted, true)) in results.items():
        print("\n%s labels: predicted / true possessions %.2f" % (label_set, predicted / true))
        print("  %-9s %6s  %s" % ("true", "share", "  ".join("%7s" % o for o in score.OUTCOMES)))
        for b, name in enumerate(names):
            mine = [o for k, o, _ in rows if k == b]
            print("  %-9s %5.0f%%  %s" % (name, 100 * len(mine) / len(rows), "  ".join(
                "%7.2f" % (mine.count(o) / max(len(mine), 1)) for o in score.OUTCOMES)))
        for b, name in enumerate(names[:2]):
            cells = []
            for seen in (True, False):
                mine = [o for k, o, s in rows if k == b and s == seen]
                cells.append("ball seen %s: %.0f%% of them, found %.2f, missed %.2f" % (
                    "on half or more" if seen else "less", 100 * len(mine) / max(
                        sum(k == b for k, _, _ in rows), 1), mine.count("found") / max(
                        len(mine), 1), mine.count("missed") / max(len(mine), 1)))
            print("  %-9s %s" % (name, "; ".join(cells)))
        for kind in ("other", "dead"):
            cells = []
            for b, name in enumerate(names):
                mine = [r for k, n, r in separators if k == kind and n == b]
                if mine:
                    cells.append("%s %.2f (n %d)" % (name, np.mean(mine), len(mine)))
            print("  separators, %s: reproduced %s" % (kind, ", ".join(cells)))


if __name__ == "__main__":
    report(run())
