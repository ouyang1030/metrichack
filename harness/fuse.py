"""Smart Tagging's weight in the decoder, set on the harness with simulated taggers.

    python -m harness.fuse      # after harness.train_v2 and scripts/calibrate_tagger.py

Smart Tagging exists only for BAR-ATH, so its weight cannot be learnt from tagged matches. A
phase tagger is simulated on each held-out DFL match instead (harness/tagger.py), with the knobs
calibrated to Smart Tagging (tagger.calibrated(): how much of the match it votes on, how long
and how often its phases change team, how often the tracking agrees with them). Two kinds:
  * independent: phases from DFL's labels -- errors independent of the tracking, the best case;
  * echo: phases from v2's own decoded path -- a tagger that repeats what the tracking says and
    adds nothing, the worst case (it looks trustworthy, so the estimate of its reliability is
    high and its evidence counts the tracking twice).
Smart Tagging, computed by Metrica from the same uncorrected tracking, lies somewhere between.
For each kind the pipeline's fusion runs as on BAR-ATH (pipeline.tagging_evidence: votes, their
reliability estimated on the match itself, the weighted log evidence added to the decoder's),
at v2's decoder setting (models/v2.json), on the same copies and against the same label sets as
harness/train_v2.py. The weight kept is the one with the best mean spellF1 over both kinds, the
degraded and stress copies and train_v2.CHOICE -- chosen as the temperature is. Also printed:
predicted / true possessions per weight (DFL labels), to set beside the change in the number of
possessions Smart Tagging makes on BAR-ATH (scripts/diagnose.py). Nothing here touches BAR-ATH.
"""

import json
import os

import numpy as np

from harness import idsse, score, semantics, tagger, train_v2
from mtah import config, pipeline, tagging
from evaluation import possession_metrics as pm

KINDS = ("independent", "echo")
WEIGHTS = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
COPIES = (1, 2, 3, 4)  # of the saved outputs: seeds 0-2, stress (0 is the clean copy)
DEGRADED = (1, 2, 3)
SEED = 0


def _setting():
    with open(train_v2.MODEL_PATH) as handle:
        exported = json.load(handle)
    return {key: exported[key] for key in ("kappa", "gamma")}


def _outputs(match_id):
    with np.load(os.path.join(train_v2.OOF_DIR, match_id + ".npz")) as saved:
        return saved["probs"], saved["change"], json.loads(str(saved["model"]))


def cases():
    """[(draws, n_frames, halves, paths)] per DFL match for tagger.fit(): the independent
    tagger's draws and v2's decoded paths on the degraded copies."""
    setting = _setting()
    out = []
    for match_id in idsse.match_ids():
        match = idsse.load(match_id)
        probs, change, model = _outputs(match_id)
        rng = np.random.default_rng([SEED, int(match_id, 36)])
        out.append((tagger.draws(match.labels, match.halves, rng), len(match.labels),
                    match.halves, [train_v2.decode(probs[copy], change[copy], model, **setting)
                                   for copy in DEGRADED]))
    return out


def run():
    setting = _setting()
    knobs = tagger.calibrated()
    rows = {}    # (label set, kind, weight, copy) -> [(halves, 2) per match]
    counts = {}  # (kind, weight, copy) -> [predicted, true] possessions, DFL labels
    for match_id in idsse.match_ids():
        match = idsse.load(match_id)
        targets = {name: match.labels if control_s is None
                   else semantics.metrica(match.labels, match.halves, control_s)
                   for name, control_s in train_v2.LABEL_SETS.items()}
        probs, change, model = _outputs(match_id)
        mask = match.labels != pm.X
        true = len(pm.possession_spells(match.labels, match.halves))
        for copy in COPIES:
            plain = train_v2.decode(probs[copy], change[copy], model, **setting)
            kind_of_copy = "stress" if copy == 4 else "degraded"
            for kind in KINDS:
                rng = np.random.default_rng([SEED, copy, KINDS.index(kind), int(match_id, 36)])
                source = match.labels if kind == "independent" else plain
                vote = tagging.votes(tagger.phases(tagger.draws(source, match.halves, rng), knobs),
                                     len(mask))
                for weight in WEIGHTS:
                    path = plain if weight == 0 else pipeline.decode_probs(
                        probs[copy], model, setting["kappa"], change[copy], setting["gamma"],
                        pipeline.tagging_evidence(probs[copy], vote, weight, mask))
                    for name, labels in targets.items():
                        rows.setdefault((name, kind, weight, kind_of_copy), []).append(
                            score.half_scores(labels, path, match.halves))
                    count = counts.setdefault((kind, weight, kind_of_copy), [0, 0])
                    count[0] += len(pm.possession_spells(path, match.halves))
                    count[1] += true
        print("done", match_id, flush=True)
    return rows, counts


def report(rows, counts):
    def table(key):
        parts = rows[key]
        if key[-1] == "degraded":  # three seeds per match: their mean, as train_v2 does
            parts = [np.mean(parts[i:i + 3], axis=0) for i in range(0, len(parts), 3)]
        return np.concatenate(parts)

    for name in train_v2.LABEL_SETS:
        for kind in KINDS:
            for copy in ("degraded", "stress"):
                base = table((name, kind, 0.0, copy))
                cells = []
                for weight in WEIGHTS[1:]:
                    mean, low, high = score.interval(table((name, kind, weight, copy)) - base)
                    cells.append("%g: %+.3f [%+.3f,%+.3f]" % (weight, mean[0], low[0], high[0]))
                print("%-3s %-11s %-8s base %.3f | %s" % (name, kind, copy, base[:, 0].mean(),
                                                          "  ".join(cells)))
    for kind in KINDS:
        for copy in ("degraded", "stress"):
            print("%-11s %-8s predicted / true possessions by weight: %s" % (
                kind, copy, "  ".join("%g: %.3f" % (weight, np.divide(*counts[kind, weight, copy]))
                                      for weight in WEIGHTS)))
    best = max(WEIGHTS, key=lambda weight: np.mean([
        table((name, kind, weight, copy))[:, 0].mean()
        for name in train_v2.CHOICE for kind in KINDS for copy in ("degraded", "stress")]))
    print("weight with the best mean spellF1 over both kinds, degraded and stress, %s: %g" % (
        "/".join(train_v2.CHOICE), best))
    return best


if __name__ == "__main__":
    print("chosen weight -> set pipeline.TAGGING_WEIGHT (%s)" % os.path.relpath(
        os.path.join(config.ROOT, "mtah", "pipeline.py"), config.ROOT))
    report(*run())
