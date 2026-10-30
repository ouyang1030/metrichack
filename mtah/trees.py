"""Gradient-boosted trees evaluated in numpy, so that the pipeline needs no scikit-learn.

harness/train_v2.py fits a scikit-learn HistGradientBoostingClassifier on off-game frames and
export() flattens it to JSON: per tree, the split feature and threshold, which way an unknown
(NaN) value goes, the children and the leaf values. probabilities() walks every frame down
every tree level by level and sums the leaves in the order scikit-learn does, so the raw scores
are the trained model's own.
"""

import numpy as np

FIELDS = ("feature_idx", "num_threshold", "missing_go_to_left", "left", "right", "is_leaf",
          "value")


def export(classifier):
    """JSON-ready dict of a fitted HistGradientBoostingClassifier (numeric features only)."""
    return {
        "baseline": classifier._baseline_prediction.ravel().tolist(),
        "trees": [[{field: predictor.nodes[field].tolist() for field in FIELDS}
                   for predictor in iteration] for iteration in classifier._predictors],
    }


def _leaves(X, tree, rows):
    feature, threshold, nan_left, left, right, leaf, value = (np.array(tree[field])
                                                              for field in FIELDS)
    node = np.zeros(len(X), dtype=np.int64)
    while not leaf[node].all():
        x = X[rows, feature[node]]
        go_left = np.where(np.isnan(x), nan_left[node] == 1, x <= threshold[node])
        node = np.where(leaf[node] == 1, node, np.where(go_left, left[node], right[node]))
    return value[node]


def probabilities(X, model):
    """(n, n_classes) class probabilities of the exported model for the rows of X."""
    raw = np.zeros((len(X), len(model["baseline"])))
    raw += np.array(model["baseline"])
    rows = np.arange(len(X))
    for iteration in model["trees"]:
        for k, tree in enumerate(iteration):
            raw[:, k] += _leaves(X, tree, rows)
    raw -= raw.max(axis=1, keepdims=True)
    return np.exp(raw) / np.exp(raw).sum(axis=1, keepdims=True)
