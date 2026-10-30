"""Fit and cross-validate v1 variants (mtah/emissions.py + mtah/decoder.py) on the harness.

    python -m harness.train_v1

Leave one match out: for every held-out match, each variant's emission models and the
transition probabilities are fitted on degraded seed-0 copies of the other six; the held-out
match is scored clean and on degraded seeds 0-2. Every variant runs at every kappa, next to
baseline_smooth50 on the same copies, and its best kappa is compared with REFERENCE's by a
paired bootstrap over halves. Then ACCEPTED -- set by hand after reading that comparison
(AGENTS.md 5.5) -- is refitted on all seven matches at its best kappa and written to
models/v1.json. v2 (harness/train_v2.py) has since replaced v1 in the pipeline; v1 stays as the
reference it is measured against. Nothing here touches BAR-ATH.
"""

import dataclasses
import json
import os

import numpy as np

from harness import degrade, idsse, score
from mtah import baseline, config, decoder, emissions, features, pipeline
from evaluation import possession_metrics as pm

# Variant name -> (team evidence terms (emissions.TEAM_TERMS), sub-states per state
# (decoder.chain), boundary refinement (mtah/refine.py)). The dead-ball model is shared.
# Rejected so far (EXPERIMENTS.md): "control" and "spread" terms, separate temperatures, chains of
# 2-3 sub-states, boundary refinement. Each stays available for re-testing once the simulator is
# calibrated on the ATD.
VARIANTS = {
    "near+flow": (("near", "flow"), 1, False),
    "near+flow refine": (("near", "flow"), 1, True),
}
REFERENCE = "near+flow"
ACCEPTED = "near+flow"
KAPPAS = (0.2, 0.25)  # one temperature for both kinds of evidence (EXPERIMENTS.md)
SEEDS = (0, 1, 2)
MODEL_PATH = os.path.join(config.ROOT, "models", "v1.json")


@dataclasses.dataclass
class Prepared:
    labels: np.ndarray
    halves: np.ndarray
    clean: features.Features
    degraded: list           # Features per seed
    smooth50: list           # baseline_smooth50 codes: clean, then per seed


def _smooth50(match):
    return pm.label_codes(baseline.smooth(baseline.nearest_player_codes(
        match.players, match.ball, match.slot_team, match.pitch), 50))


def prepare(match_id):
    match = idsse.load(match_id)
    noisy = [degrade.degrade(match, seed) for seed in SEEDS]
    return Prepared(
        labels=match.labels, halves=match.halves,
        clean=features.compute(match.players, match.ball, match.slot_team, match.pitch),
        degraded=[features.compute(m.players, m.ball, m.slot_team, m.pitch) for m in noisy],
        smooth50=[_smooth50(m) for m in [match] + noisy])


def _concat(parts):
    """Features of the in-play frames of several matches, end to end."""
    return features.Features(*(np.concatenate([getattr(f, field.name)[mask] for f, mask in parts])
                               for field in dataclasses.fields(features.Features)))


def transitions(prepared):
    """The decoder's transition probabilities and mean durations; each match weighs the same."""
    probs = np.mean([np.exp(decoder.transition_log_probs(p.labels, p.halves)) for p in prepared],
                    axis=0)
    return {"log_trans": np.log(probs).tolist(),
            "mean_frames": np.mean([decoder.mean_durations(p.labels, p.halves)
                                    for p in prepared], axis=0).tolist()}


def fit(prepared, terms, chain, refine):
    """Model from degraded seed-0 features of the given matches."""
    masks = [p.labels != pm.X for p in prepared]
    model = emissions.fit(_concat([(p.degraded[0], mask) for p, mask in zip(prepared, masks)]),
                          np.concatenate([p.labels[mask] for p, mask in zip(prepared, masks)]),
                          terms)
    model.update(transitions(prepared))
    model["chain"] = chain
    model["refine"] = refine
    return model


def predict(f, model, kappa):
    return pipeline.decode(f, model, kappa, kappa)


def _name(variant, kappa):
    return "%s k=%g" % (variant, kappa)


def cross_validate(prepared):
    """{name: (clean rows, degraded rows)} over all halves, one model per held-out match."""
    rows = {}

    def add(name, held_out, predictions):  # predictions: clean, then one per seed
        clean, noisy = rows.setdefault(name, ([], []))
        scores = [score.half_scores(held_out.labels, p, held_out.halves) for p in predictions]
        clean.append(scores[0])
        noisy.append(np.mean(scores[1:], axis=0))

    for i, held_out in enumerate(prepared):
        training = prepared[:i] + prepared[i + 1:]
        add("baseline_smooth50", held_out, held_out.smooth50)
        for variant, (terms, chain, refine) in VARIANTS.items():
            model = fit(training, terms, chain, refine)
            for kappa in KAPPAS:
                add(_name(variant, kappa), held_out,
                    [predict(f, model, kappa) for f in [held_out.clean] + held_out.degraded])
    return {name: (np.concatenate(clean), np.concatenate(noisy))
            for name, (clean, noisy) in rows.items()}


def compare(results):
    """Best kappa per variant, and paired degraded differences to smooth50 and to REFERENCE."""
    best = {variant: max(KAPPAS, key=lambda k: results[_name(variant, k)][1][:, 0].mean())
            for variant in VARIANTS}
    against = {"smooth50": results["baseline_smooth50"][1],
               REFERENCE: results[_name(REFERENCE, best[REFERENCE])][1]}
    for variant, kappa in best.items():
        for label, rows in against.items():
            if label == variant:
                continue
            mean, low, high = score.interval(results[_name(variant, kappa)][1] - rows)
            print("%-24s minus %-10s degraded: spellF1 %+.3f [%+.3f, %+.3f], macroF1 %+.3f "
                  "[%+.3f, %+.3f]" % (_name(variant, kappa), label, mean[0], low[0], high[0],
                                      mean[1], low[1], high[1]))
    return best


if __name__ == "__main__":
    prepared = [prepare(match_id) for match_id in idsse.match_ids()]
    results = cross_validate(prepared)
    print(score.table(results))
    best = compare(results)

    model = fit(prepared, *VARIANTS[ACCEPTED])
    model["kappa_dead"] = model["kappa_team"] = best[ACCEPTED]
    with open(MODEL_PATH, "w") as handle:
        json.dump(model, handle, indent=2)
        handle.write("\n")
    print("accepted", _name(ACCEPTED, best[ACCEPTED]), "-> wrote",
          os.path.relpath(MODEL_PATH, config.ROOT))
