"""Cross-validate the v2 frame classifier on the harness, then fit models/v2.json.

    python -m harness.train_v2

v2 replaces v1's two logistic models by one gradient-boosted classifier P(H, A, D | frame) over
the wider feature table (features.matrix), decoded by the same HMM (pipeline.decode_probs).
Protocol as in harness/train_v1.py: leave one match out; fit on degraded seed-0 copies of the
other six -- every STRIDE-th in-play frame, plus the same frames with the teams swapped, so that
the classifier has no side to prefer; score the held-out match clean and on seeds 0-2. Next to
it run v1 as accepted and baseline_smooth50, refitted on the same folds, and count how many
short possessions each finds. The degraded copies carry the noise fitted to the ATD
(degrade.calibrated(); run scripts/calibrate_noise.py first, with the data bundle in place). A
stress copy (STRESS: degrade.stressed(), much noisier, and with the ball no more likely to vanish
off the pitch) checks that a gain does not rest on the particular noise the classifier was
trained under.

Every prediction is scored against each of LABEL_SETS: DFL's own labels, and the same labels
under Metrica's rule that a touch without control does not change possession
(harness/semantics.py) -- which of the two the ground truth follows is not known yet. Each
classifier variant (VARIANTS) runs at every temperature and keeps the one with the best mean
spellF1 over the degraded and stress copies and the CHOICE label sets: how noisy the ATD is is
not known yet, and a low temperature holds up better under heavy noise (EXPERIMENTS.md).
ACCEPTED is set by hand after reading the comparison with REFERENCE (AGENTS.md 5.5; a variant
must not lose on any label set); only then is it refitted on all seven matches and exported.
Nothing here touches BAR-ATH.
"""

import dataclasses
import json
import os
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from harness import degrade, idsse, score, semantics, train_v1
from mtah import config, decoder, emissions, features, pipeline, trees
from evaluation import possession_metrics as pm

STRIDE = 5  # neighbouring frames say nearly the same thing
CHANGE_WINDOW = 5  # frames (0.2 s) either side of a change that the change detector marks
GBM = dict(max_iter=100, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=100,
           early_stopping=False, random_state=0)
KAPPAS = (0.2, 0.3, 0.45, 0.6, 0.8, 1.0)
GAMMAS = (1.0, 1.5)
NOISE = degrade.calibrated()  # seeds 0-2, for training and scoring. STRICT: never print it
STRESS = degrade.stressed(NOISE)
STRESS_SEED = 100
RANDOM_SEEDS = (10, 11, 12)  # training copies with noise drawn between NOISE and STRESS
# Label set -> control time (s) of Metrica's rule; None: DFL's labels as they are.
LABEL_SETS = {"DFL": None, "M1": 1.0, "M2": 2.0}
CHOICE = ("DFL", "M1")  # the temperature is chosen on these; M2 is a check
# Training copies of a match -> their feature tables; each copy gives every STRIDE-th in-play
# frame, the n-th copy from the n-th frame on.
COPIES = {"seed 0": lambda p: p.tables[1:2], "seeds 0-2": lambda p: p.tables[1:4],
          "random 3": lambda p: p.random, "mixed": lambda p: p.tables[1:4] + p.random}
# Decoder -> the settings it is tried at. "hmm": constant transitions; "iohmm": transitions that
# follow a change detector (decoder.input_transitions).
DECODERS = {"hmm": [dict(kappa=k) for k in KAPPAS],
            "iohmm": [dict(kappa=k, gamma=g) for k in KAPPAS for g in GAMMAS]}


@dataclasses.dataclass(frozen=True)
class Variant:
    copies: str = "seeds 0-2"  # training copies (COPIES)
    decoder: str = "iohmm"     # DECODERS
    stacked: bool = False      # a second stage over the first stage's outputs (features.context)
    target: str = "DFL"        # the label set the classifiers are trained on
    drop: tuple = ()           # feature-name prefixes left out (an ablation)


TOUCH = ("impulse", "duel_", "touch_", "lead_")  # the touch and duel features (plan step 4)
RECEPTION = ("since_contact", "until_contact", "flight_", "contact_", "hold_")  # step 4b


# Accepted (EXPERIMENTS.md, plan steps 1-4): the change detector, the second stage, the touch
# features; mixed copies (by the team, as an exception to AGENTS.md 5.5: within noise on the
# calibrated copies, +0.09 under stress). Rejected: training on M1; a separate factor on
# possession switches; the reception features (step 4b, within noise, kept as an ablation
# group). Ablations: e.g. Variant(stacked=True, drop=TOUCH + RECEPTION), Variant(decoder="hmm").
VARIANTS = {"gbm": Variant(stacked=True, copies="mixed", drop=RECEPTION)}
REFERENCE = "gbm"
ACCEPTED = "gbm"
V1 = (("near", "flow"), 1, False)  # v1 as accepted (models/v1.json), at kappa V1_KAPPA
V1_KAPPA = 0.2
SWAP_LABELS = np.array([emissions.A, emissions.H, emissions.D])  # indexed by H, A, D
MODEL_PATH = os.path.join(config.ROOT, "models", "v2.json")


@dataclasses.dataclass
class Prepared:
    v1: train_v1.Prepared  # clean and per seed, DFL's labels
    targets: dict          # label set -> label codes
    tables: list           # features.matrix: clean, per seed, stress
    random: list           # features.matrix per RANDOM_SEEDS copy
    stress_v1: features.Features
    stress_smooth50: np.ndarray


@dataclasses.dataclass
class Scores:
    """One method against one label set, over the held-out matches."""
    clean: list = dataclasses.field(default_factory=list)     # (halves, 2) per match
    degraded: list = dataclasses.field(default_factory=list)  # the seeds' mean
    stress: list = dataclasses.field(default_factory=list)
    recall: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((2, 2), dtype=int))
    predicted: int = 0                                        # recall, predicted: all seeds

    def add(self, labels, halves, predictions):  # predictions: clean, one per seed, stress
        rows = [score.half_scores(labels, p, halves) for p in predictions]
        self.clean.append(rows[0])
        self.degraded.append(np.mean(rows[1:-1], axis=0))
        self.stress.append(rows[-1])
        for p in predictions[1:-1]:
            self.recall += score.spell_recall(labels, p, halves)
            self.predicted += len(pm.possession_spells(p, halves))

    def rows(self, copy):
        return np.concatenate(getattr(self, copy))


def prepare(match_id):
    match = idsse.load(match_id)
    versions = [match] + [degrade.degrade(match, seed, NOISE) for seed in train_v1.SEEDS]
    stress = degrade.degrade(match, STRESS_SEED, STRESS)
    random = [degrade.degrade(match, seed, degrade.drawn(NOISE, STRESS,
                                                         np.random.default_rng(seed)))
              for seed in RANDOM_SEEDS
              if any(v.copies in ("random 3", "mixed") for v in VARIANTS.values())]
    computed = [features.matrix(m.players, m.ball, m.slot_team, m.pitch)
                for m in versions + [stress] + random]
    return Prepared(
        v1=train_v1.Prepared(
            labels=match.labels, halves=match.halves,
            clean=features.compute(match.players, match.ball, match.slot_team, match.pitch),
            degraded=[features.compute(m.players, m.ball, m.slot_team, m.pitch)
                      for m in versions[1:]],
            smooth50=[train_v1._smooth50(m) for m in versions]),
        targets={name: match.labels if control_s is None
                 else semantics.metrica(match.labels, match.halves, control_s)
                 for name, control_s in LABEL_SETS.items()},
        tables=[table for table, _ in computed[:len(versions) + 1]],
        random=[table for table, _ in computed[len(versions) + 1:]],
        stress_v1=features.compute(stress.players, stress.ball, stress.slot_team, stress.pitch),
        stress_smooth50=train_v1._smooth50(stress)), computed[0][1]


def change_kinds(labels, halves):
    """decoder.NONE / SWITCH / STOP / RESTART per frame: the kind of label change within
    CHANGE_WINDOW frames of it (the later one where two are that close)."""
    kinds = np.full(len(labels), decoder.NONE, dtype=np.int8)
    for a, b in halves:
        for t in np.flatnonzero(np.diff(labels[a:b])) + a + 1:
            kinds[max(a, t - CHANGE_WINDOW):min(b, t + CHANGE_WINDOW + 1)] = (
                decoder.STOP if labels[t] == pm.D
                else decoder.RESTART if labels[t - 1] == pm.D else decoder.SWITCH)
    return kinds


def _sample(matches, names, targets, swapped, columns=slice(None)):
    """(X, y) from (match, feature tables of its training copies) pairs: every STRIDE-th in-play
    frame and the given columns, with targets(match) per frame, then the same frames with the
    teams swapped and y mapped through swapped."""
    X, y = [], []
    for p, tables in matches:
        codes = targets(p)
        in_play = np.flatnonzero(p.v1.labels != pm.X)
        for n, table in enumerate(tables):
            X.append(table[in_play[n % STRIDE::STRIDE]][:, columns])
            y.append(codes[in_play[n % STRIDE::STRIDE]])
    X, y = np.concatenate(X), np.concatenate(y)
    return np.concatenate((X, X[:, features.swap_index(names)])), np.concatenate((y, swapped[y]))


@dataclasses.dataclass
class Fitted:
    model: dict        # decoder model: names, prior, transitions
    columns: np.ndarray  # of the full feature table
    classifier: object
    detector: object   # P(decoder.NONE, SWITCH, STOP, RESTART | frame)
    second: object = None

    def outputs(self, X):
        """P(H, A, D | frame) and the change probabilities for a full feature table."""
        X = X[:, self.columns]
        probs, change = self.classifier.predict_proba(X), self.detector.predict_proba(X)
        if self.second is not None:
            probs = self.second.predict_proba(features.second_stage(X, probs, change)[0])
        return probs, change


def fit_first(prepared, names, variant):
    """Classifier and change detector from the variant's training copies of the given matches."""
    matches = [(p, COPIES[variant.copies](p)) for p in prepared]
    target = variant.target
    columns = np.array([i for i, name in enumerate(names) if not name.startswith(variant.drop)])
    names = [names[i] for i in columns]
    X, y = _sample(matches, names, lambda p: p.targets[target], SWAP_LABELS, columns)
    model = {"names": names, "prior": [float(np.mean(y == s)) for s in range(3)],
             "change_window": CHANGE_WINDOW}
    model.update(train_v1.transitions([dataclasses.replace(p.v1, labels=p.targets[target])
                                       for p in prepared]))
    X_change, kinds = _sample(matches, names, lambda p: change_kinds(p.targets[target],
                                                                     p.v1.halves),
                              np.arange(4), columns)
    return Fitted(model, columns, HistGradientBoostingClassifier(**GBM).fit(X, y),
                  HistGradientBoostingClassifier(**GBM).fit(X_change, kinds))


def _second_tables(first, tables):
    """Second-stage tables, one at a time (each is large)."""
    for X in tables:
        yield features.second_stage(X[:, first.columns], *first.outputs(X))[0]


def fit_second(prepared, names, variant):
    """Second-stage classifier from out-of-fold first-stage outputs: for each match, the first
    stage is fitted on the others (stacked sequential learning)."""
    matches = []
    for j, p in enumerate(prepared):
        first = fit_first(prepared[:j] + prepared[j + 1:], names, variant)
        matches.append((p, _second_tables(first, COPIES[variant.copies](p))))
    added = features.second_stage(np.zeros((1, 0)), np.zeros((1, 3)), np.zeros((1, 4)))[1]
    X, y = _sample(matches, first.model["names"] + added, lambda p: p.targets[variant.target],
                   SWAP_LABELS)
    return HistGradientBoostingClassifier(**GBM).fit(X, y)


def decode(probs, change, model, kappa, gamma=0.0):
    """H/A/D codes from the classifier's and the change detector's probabilities."""
    return pipeline.decode_probs(probs, model, kappa, change if gamma else None, gamma)


def _name(variant, setting):
    return variant + "".join(" %s=%g" % (key[0], value) for key, value in setting.items())


def cross_validate(prepared, names):
    """{label set: {method: Scores}}, one model per held-out match and variant."""
    scores = {label_set: {} for label_set in LABEL_SETS}

    def add(name, held_out, predictions):
        for label_set, labels in held_out.targets.items():
            scores[label_set].setdefault(name, Scores()).add(labels, held_out.v1.halves,
                                                             predictions)

    for i, held_out in enumerate(prepared):
        start = time.time()
        training = prepared[:i] + prepared[i + 1:]
        add("baseline_smooth50", held_out, held_out.v1.smooth50 + [held_out.stress_smooth50])
        v1_model = train_v1.fit([p.v1 for p in training], *V1)
        add("v1", held_out, [train_v1.predict(f, v1_model, V1_KAPPA) for f in
                             [held_out.v1.clean] + held_out.v1.degraded + [held_out.stress_v1]])
        first, second = {}, {}  # (label set, copies, drop) -> fitted on the training matches
        for name, variant in VARIANTS.items():
            key = variant.target, variant.copies, variant.drop
            if key not in first:
                first[key] = fit_first(training, names, variant)
            fitted = first[key]
            if variant.stacked:
                if key not in second:
                    second[key] = fit_second(training, names, variant)
                fitted = dataclasses.replace(fitted, second=second[key])
            outputs = [fitted.outputs(X) for X in held_out.tables]
            for setting in DECODERS[variant.decoder]:
                add(_name(name, setting), held_out,
                    [decode(probs, change, fitted.model, **setting) for probs, change in outputs])
        print("fold %d done in %.0f s" % (i, time.time() - start))
    return scores


def best_settings(scores):
    """{variant: decoder setting} with the best mean spellF1 over degraded / stress and CHOICE."""
    return {name: max(DECODERS[variant.decoder], key=lambda setting: np.mean([
        scores[label_set][_name(name, setting)].rows(copy)[:, 0].mean()
        for label_set in CHOICE for copy in ("degraded", "stress")]))
        for name, variant in VARIANTS.items()}


def report(scores):
    best = best_settings(scores)
    reference = _name(REFERENCE, best[REFERENCE])
    for label_set, by_name in scores.items():
        print("\n%-4s labels                clean        degraded [5-95%%]          stress       "
              "found <4 s  >=4 s  predicted/true" % label_set)
        for name, s in by_name.items():
            mean, low, high = score.interval(s.rows("degraded"))
            print("%-26s %.3f/%.3f  %.3f [%.3f,%.3f]/%.3f  %.3f/%.3f  %10.3f %6.3f %8.2f" % (
                name, *s.rows("clean").mean(axis=0), mean[0], low[0], high[0], mean[1],
                *s.rows("stress").mean(axis=0), *(s.recall[:, 0] / s.recall[:, 1]),
                s.predicted / s.recall[:, 1].sum()))
        pairs = [(reference, "v1")] + [(_name(v, k), reference) for v, k in best.items()
                                      if v != REFERENCE]
        for name, other in pairs:
            for copy in ("degraded", "stress"):
                mean, low, high = score.interval(by_name[name].rows(copy)
                                                 - by_name[other].rows(copy))
                print("  %s minus %s, %s: spellF1 %+.3f [%+.3f, %+.3f], macroF1 %+.3f "
                      "[%+.3f, %+.3f]" % (name, other, copy, mean[0], low[0], high[0], mean[1],
                                          low[1], high[1]))
    return best


def export(prepared, names, variant, setting):
    """Fit the variant on all the given matches and write it, with its decoder setting, to
    MODEL_PATH."""
    fitted = fit_first(prepared, names, variant)
    model = dict(fitted.model, **trees.export(fitted.classifier), **setting)
    if variant.decoder != "iohmm":
        del model["change_window"]
    if variant.decoder == "iohmm" or variant.stacked:
        model["detector"] = trees.export(fitted.detector)
    if variant.stacked:
        model["second"] = trees.export(fit_second(prepared, names, variant))
    with open(MODEL_PATH, "w") as handle:
        json.dump(model, handle, separators=(",", ":"))
        handle.write("\n")


if __name__ == "__main__":
    prepared = [prepare(match_id) for match_id in idsse.match_ids()]
    names = prepared[0][1]
    prepared = [p for p, _ in prepared]
    best = report(cross_validate(prepared, names))
    export(prepared, names, VARIANTS[ACCEPTED], best[ACCEPTED])
    print("accepted %s -> wrote %s" % (_name(ACCEPTED, best[ACCEPTED]),
                                      os.path.relpath(MODEL_PATH, config.ROOT)))
