"""Structured decoding: one most likely H/A/D path, so that the output is piecewise constant.

An HMM over the three states. Each state can be a chain of k sub-states (`chain`), which gives
it a minimum duration of k frames and a duration distribution that peaks above zero instead of
the geometric one. Where to go when a state ends comes from how often possession changes and
play stops in off-game matches; per-frame evidence is scaled by a temperature `kappa` < 1,
because neighbouring frames are far from independent and taking them at face value makes every
brief contrary signal a switch.
"""

import numpy as np


def transition_log_probs(labels, runs):
    """(3, 3) log P(next | previous) from consecutive frames inside the given runs."""
    counts = np.ones((3, 3))  # add-one, so that no transition is impossible
    for start, stop in runs:
        np.add.at(counts, (labels[start:stop - 1], labels[start + 1:stop]), 1)
    return np.log(counts / counts.sum(axis=1, keepdims=True))


def mean_durations(labels, runs, codes=(0, 1, 2)):
    """Mean length in frames of the constant runs of each code inside the given runs."""
    lengths = {code: [] for code in codes}
    for start, stop in runs:
        values = labels[start:stop]
        edges = np.flatnonzero(np.diff(values)) + 1
        for a, b in zip(np.concatenate(([0], edges)), np.concatenate((edges, [len(values)]))):
            lengths[values[a]].append(b - a)
    return [float(np.mean(lengths[code])) for code in codes]


def chain(log_trans, mean_frames, k):
    """(log transitions over k sub-states per state, owner state of each sub-state).

    Sub-states of a state are visited in a row, each held with the same probability, so that
    the state lasts mean_frames on average; the last one leaves to the other states in the
    proportions of log_trans.
    """
    n = len(log_trans)
    probs = np.zeros((n * k, n * k))
    for state in range(n):
        stay = 1 - k / mean_frames[state]
        others = np.flatnonzero(np.arange(n) != state)
        leave = np.exp(log_trans[state, others])
        for j in range(k):
            sub = state * k + j
            probs[sub, sub] = stay
            if j < k - 1:
                probs[sub, sub + 1] = 1 - stay
            else:
                probs[sub, others * k] = (1 - stay) * leave / leave.sum()
    with np.errstate(divide="ignore"):
        return np.log(probs), np.repeat(np.arange(n), k)


def viterbi(log_emit, log_trans):
    """Most likely state per frame given (n, k) log emissions and log transitions, either (k, k)
    or (n, k, k): row t, from t - 1 to t, may depend on the frame (an input-output HMM)."""
    n, k = log_emit.shape
    back = np.empty((n, k), dtype=np.int8)
    score = log_emit[0].copy()
    for t in range(1, n):
        candidates = score[:, None] + (log_trans[t] if log_trans.ndim == 3 else log_trans)
        back[t] = candidates.argmax(axis=0)
        score = candidates.max(axis=0) + log_emit[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


# Kinds of change a change detector tells apart (an input-output HMM, input_transitions()).
NONE, SWITCH, STOP, RESTART = range(4)


def input_transitions(log_trans, mean_frames, change, window, gamma):
    """(n, 3, 3) log transitions over H, A, D that follow a change detector.

    change: (n, 4) P(no change, team switch, stop, restart within +-window frames | frame). A
    change that happens within +-window frames of t happens at t with probability 1/(2 window
    + 1), so P(kind) / (2 window + 1) is that kind's hazard at t. gamma blends it with the
    plain HMM's constant hazard, in log space: 0 is the plain HMM, 1 trusts the detector alone.
    A restart goes to either team alike; the evidence decides which.
    """
    base = np.exp(chain(log_trans, mean_frames, 1)[0])
    hazard = np.zeros((len(change), 3, 3))
    hazard[:, 0, 1] = hazard[:, 1, 0] = change[:, SWITCH]
    hazard[:, 0, 2] = hazard[:, 1, 2] = change[:, STOP]
    hazard[:, 2, 0] = hazard[:, 2, 1] = change[:, RESTART] / 2
    off = ~np.eye(3, dtype=bool)
    probs = np.zeros_like(hazard)
    probs[:, off] = base[off] ** (1 - gamma) * (hazard[:, off] / (2 * window + 1)) ** gamma
    probs[:, ~off] = 1 - probs.sum(axis=2)
    return np.log(probs)
