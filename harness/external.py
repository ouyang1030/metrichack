"""The exported models on other providers' matches: Metrica's sample games and SkillCorner's.

    python -m harness.external         # after harness.metrica and harness.skillcorner

A check that what was chosen on DFL holds where tracking, label semantics and noise come from
elsewhere -- not a tuning set: nothing is fitted or chosen here. models/v1.json and
models/v2.json as exported (trained on DFL only), and baseline_smooth50, are scored with the
official metric per half, with the 5-95% bootstrap interval over halves and paired differences:
  * Metrica sample games 1-3 (labels from Metrica's events, harness/metrica.py): the tracking as
    provided (its ball is missing while dead and now and then in play), then degraded with the
    noise calibrated to the ATD (seeds 0-2) and the stress noise of harness/train_v2.py;
  * SkillCorner (labels from SkillCorner's automatic phases and possession,
    harness/skillcorner.py): broadcast tracking as provided, real noise, never degraded.
Each against its own labels and under Metrica's rule with a 1 s control time (M1,
harness/semantics.py). v2 runs as the pipeline does but without Smart Tagging, which exists
for BAR-ATH only. Nothing here touches BAR-ATH.
"""

import json
import os

import numpy as np

from harness import degrade, metrica, score, semantics, skillcorner, train_v1, train_v2
from mtah import config, features, pipeline
from evaluation import possession_metrics as pm

V1_MODEL = os.path.join(config.ROOT, "models", "v1.json")


def v2(match, model):
    probs, change = pipeline.outputs(match.players, match.ball, match.slot_team, match.pitch,
                                     model)
    return pipeline.decode_probs(probs, model, model["kappa"], change, model["gamma"])


def v1(match, model):
    return train_v1.predict(features.compute(match.players, match.ball, match.slot_team,
                                             match.pitch), model, train_v2.V1_KAPPA)


METHODS = {"baseline_smooth50": lambda match, _: train_v1._smooth50(match), "v1": v1, "v2": v2}


def copies(match, provided_only):
    """{copy name: matches} to score: as provided, and (unless provided_only) degraded."""
    if provided_only:
        return {"provided": [match]}
    return {"provided": [match],
            "degraded": [degrade.degrade(match, seed, train_v2.NOISE)
                         for seed in train_v1.SEEDS],
            "stress": [degrade.degrade(match, train_v2.STRESS_SEED, train_v2.STRESS)]}


def run(sets):
    """{(set, copy, label set, method): (halves, 2) rows}."""
    with open(train_v2.MODEL_PATH) as handle:
        models = {"v2": json.load(handle)}
    with open(V1_MODEL) as handle:
        models["v1"] = json.load(handle)
    rows = {}
    for set_name, (matches, provided_only) in sets.items():
        for match in matches:
            targets = {"own": match.labels, "M1": semantics.metrica(match.labels, match.halves,
                                                                    1.0)}
            for copy, versions in copies(match, provided_only).items():
                for method, predict in METHODS.items():
                    per_version = [[score.half_scores(labels, predict(m, models.get(method)),
                                                      match.halves)
                                    for labels in targets.values()] for m in versions]
                    for k, label_set in enumerate(targets):
                        rows.setdefault((set_name, copy, label_set, method), []).append(
                            np.mean([v[k] for v in per_version], axis=0))
            print("done", match.name, flush=True)
    return {key: np.concatenate(parts) for key, parts in rows.items()}


def report(rows):
    for key in sorted({key[:3] for key in rows}):
        print("\n%s, %s copy, %s labels" % key)
        for method in METHODS:
            mean, low, high = score.interval(rows[key + (method,)])
            print("  %-18s spellF1 %.3f [%.3f, %.3f]  macroF1 %.3f" % (method, mean[0], low[0],
                                                                       high[0], mean[1]))
        for a, b in (("v2", "v1"), ("v2", "baseline_smooth50")):
            mean, low, high = score.interval(rows[key + (a,)] - rows[key + (b,)])
            print("  %s minus %s: spellF1 %+.3f [%+.3f, %+.3f], macroF1 %+.3f" % (
                a, b, mean[0], low[0], high[0], mean[1]))


if __name__ == "__main__":
    report(run({"Metrica": ([metrica.load(game) for game in metrica.GAMES], False),
                "SkillCorner": ([skillcorner.load(m) for m in skillcorner.match_ids()], True)}))
