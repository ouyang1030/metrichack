"""Per-frame evidence for H, A and D: two small logistic models fitted on off-game data only.

    P(dead | x)          from the collective speed, whether and where the ball was last seen
    P(home | alive, x)   from team contrasts (TEAM_TERMS): how much closer the nearest home slot
                         is to the ball than the nearest away slot, how much more it moves with
                         the ball, and which way and how stretched the teams are -- the last two
                         still speak when the ball is not located

The decoder needs log P(x | state) up to a constant, which is log P(state | x) - log P(state).
"""

import numpy as np

H, A, D = 0, 1, 2
CONTROL_RADIUS = 1.5  # m: scale of the closeness contrast exp(-d / r)
CONTROL_SPEED = 2.0   # m/s: scale of the moving-together contrast exp(-|v_ball - v_player| / s)


def dead_design(f):
    return np.column_stack((np.ones(len(f.speed)), f.speed, f.speed ** 2, f.seen, f.last_out,
                            np.log1p(f.gap)))


def _closeness(f, team):
    return np.exp(-getattr(f, "near_" + team) / CONTROL_RADIUS)


def _control(f, team):
    return _closeness(f, team) * np.exp(-getattr(f, "rel_" + team) / CONTROL_SPEED)


TEAM_TERMS = {
    "near": lambda f: _closeness(f, "home") - _closeness(f, "away"),
    "control": lambda f: _control(f, "home") - _control(f, "away"),
    "flow": lambda f: f.flow,
    "spread": lambda f: f.spread,
}


def team_design(f, terms):
    return np.column_stack([TEAM_TERMS[term](f) for term in terms])


def fit_logistic(X, y, iterations=30, ridge=1e-6):
    """Logistic regression by Newton's method; deterministic."""
    w = np.zeros(X.shape[1])
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-X @ w))
        hessian = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(len(w))
        w += np.linalg.solve(hessian, X.T @ (y - p))
    return w


def fit(features, labels, team_terms):
    """Weights and class priors from frames of off-game matches (labels: H/A/D codes)."""
    dead = labels == D
    team_w = fit_logistic(team_design(features, team_terms)[~dead],
                          (labels[~dead] == H).astype(float))
    return {
        "dead_w": fit_logistic(dead_design(features), dead.astype(float)).tolist(),
        "team_terms": list(team_terms),
        "team_w": team_w.tolist(),
        "prior": [float(np.mean(labels == state)) for state in (H, A, D)],
    }


def log_likelihoods(features, model, kappa_dead, kappa_team):
    """(n, 3) tempered log P(x | state) + constant, columns H, A, D.

    The evidence factorises into alive-vs-dead and, once alive, home-vs-away; each part gets its
    own temperature, because play stops rarely and possession changes often.
    """
    p_dead = 1 / (1 + np.exp(-dead_design(features) @ np.array(model["dead_w"])))
    p_home = 1 / (1 + np.exp(-team_design(features, model["team_terms"]) @ model["team_w"]))
    prior_h, prior_a, prior_d = model["prior"]
    alive = kappa_dead * (np.log(np.clip(1 - p_dead, 1e-12, None)) - np.log(1 - prior_d))
    dead = kappa_dead * (np.log(np.clip(p_dead, 1e-12, None)) - np.log(prior_d))
    home = kappa_team * (np.log(np.clip(p_home, 1e-12, None)) - np.log(prior_h / (1 - prior_d)))
    away = kappa_team * (np.log(np.clip(1 - p_home, 1e-12, None))
                         - np.log(prior_a / (1 - prior_d)))
    return np.column_stack((alive + home, alive + away, dead))


def tempered(probs, prior, kappa):
    """(n, 3) tempered log P(x | state) + constant from a classifier's (n, 3) P(state | x)."""
    return kappa * (np.log(np.clip(probs, 1e-12, None)) - np.log(prior))
