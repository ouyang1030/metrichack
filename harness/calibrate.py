"""Calibrate the ATD noise simulator to the ATD itself, under STRICT (AGENTS.md 4 and 5.2).

statistics() summarises what tracking noise does, in numbers that need no labels: how much of the
ball is missing and in how long gaps, how many of those gaps are flickers of a frame or two and
how much of the missing time falls in gaps of 5 s or more (a ball lost, typically while it is
dead); whether it goes missing more in a crowd or at speed, and how fast it moves just after it
is found again (a placed ball before a restart is still; one found after it is not); how much
the players slow down after the ball is lost for 5 s or more and speed up after it is found
again (they slow down after a stoppage starts and speed up after the restart, so these tell
when a tracker loses and finds a dead ball); how
often it moves further in one frame than a ball can fly (a false detection, or the jump back);
how much the ball and the players jitter; how many players are tracked, and for how long at a
time. Where the crowd is counted on a missing frame, the ball is put on the straight line
between where it was last seen and where it is seen next. It is computed the same way on the
ATD's graded frames and on degraded harness matches.
fit() sets the simulator's knobs until the degraded harness matches show the ATD's statistics:
the player knobs one at a time by bisection; the knobs of where and when the ball goes missing
(lost stoppages and the dropout score's share) jointly by least squares; the rate of false
detections and the ball's jitter by bisection, each on a statistic of its own.
scripts/calibrate_noise.py runs both behind the firewall: the statistics and the knobs go to
cache/aggregates/ and are never shown, and degrade.calibrated() reads the knobs.

Not calibrated, because nothing in the ATD alone measures them: the weight of ball height (the
ATD has none) and of the ball being off the pitch, the share of tracklets in the other team's
slots, identity swaps, and where a false detection sits. The harness matches' ball stands still
while it is dead, so a ball seen during a stoppage moves less on them than on the ATD; the
players' speeds, which are not frozen, carry most of what tells when a stoppage is lost.
"""

from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares

from harness import degrade

FPS = degrade.FPS
JUMP_M = 1.5          # m in one frame (37.5 m/s): further than a ball flies
SPEED_LAG = 5         # frames the ball's speed is measured over (0.2 s)
CROWD_RADIUS = 3.0    # m
BALL_WINDOW = 5       # frames: the ball's jitter is measured against its centred mean over these
PLAYER_WINDOW = 25    # frames, for the players
FLICKER_FRAMES = 2   # a gap this short is a flicker
LONG_GAP_S = 5.0      # a gap this long is a ball lost, not overlooked
EDGE_WINDOW = 50      # frames (2 s) either side of a long gap's edges, for the players' speed
FIT_STEPS = 12       # bisection steps per knob
FIT_PASSES = 3        # passes over the player knobs
BALL_PASSES = 2       # passes over the dropout knobs, then the detection knobs
FIT_DIFF_STEP = 0.05  # finite-difference step of the dropout fit, as a share of each knob's range
FIT_RESTARTS = 3      # the statistics are a little rough at small scales, which can stop the
                      # solver early; each restart from where it stopped resets its trust region
UNDEFINED = 10.0      # relative error of a statistic a trial cannot show (no gap, no long gap):
                      # such a trial loses almost no ball and is far from any target
SEED = 0

# statistic -> (the knob that moves it, the range the knob is searched in).
PLAYER_KNOBS = {"tracked": ("person_missing", 0.01, 0.6),
                "tracklet_s": ("gap_mean_s", 0.1, 60.0),
                "jitter_m": ("jitter_m", 0.0, 3.0)}
# Where and when the ball goes missing: lost stoppages (how many; when the ball is lost and
# found again; how often it is seen meanwhile) and the dropout score's share of the other frames,
# knob -> range. They act together and are fitted together, on more statistics than knobs
# (fit_dropout). The score's weights and noise keep their defaults: on these statistics they
# would trade off against the stoppage knobs. The fit starts from a tracker that loses the ball
# in every stoppage and seldom otherwise.
DROPOUT_KNOBS = {"ball_dead_lost": (0.0, 1.0),
                 "ball_dead_seen_per_min": (0.0, 6.0),
                 "ball_missing": (0.0, 0.45),
                 "ball_dead_on_s": (-3.0, 3.0),
                 "ball_dead_off_s": (-3.0, 3.0)}
DROPOUT_STATISTICS = ("ball_missing", "ball_gap_s", "ball_flicker", "ball_long",
                      "ball_speed_lift", "ball_return_lift", "loss_slowdown", "return_speedup")
DROPOUT_START = dict(ball_dead_lost=1.0, ball_dead_seen_per_min=1.0, ball_missing=0.02)
# Each has a statistic of its own.
DETECTION_KNOBS = {"ball_jumps": ("ball_false", 0.0, 0.3),
                   "ball_jitter_m": ("ball_jitter_m", 0.0, 3.0)}


def _inside(n, runs):
    inside = np.zeros(n, dtype=bool)
    for start, stop in runs:
        inside[start:stop] = True
    return inside


def _run_lengths(flags, runs):
    """Lengths of the runs of True in flags, within each [start, stop) of runs."""
    lengths = [np.zeros(0, dtype=int)]
    for start, stop in runs:
        edges = np.flatnonzero(np.diff(np.r_[0, flags[start:stop].astype(np.int8), 0]))
        lengths.append(edges[1::2] - edges[::2])
    return np.concatenate(lengths)


def _runs_of(flags, runs):
    """(starts, stops) of the stretches of flags inside the runs."""
    starts, stops = [], []
    for start, stop in runs:
        edges = np.flatnonzero(np.diff(np.r_[0, flags[start:stop].astype(int), 0])) + start
        starts.append(edges[::2])
        stops.append(edges[1::2])
    return np.concatenate(starts), np.concatenate(stops)


def _team_speed(players, pitch):
    """Mean speed of the tracked players per frame, in m/s over the last second, from their
    positions: measured alike on the ATD and on degraded copies, whatever their speed channel."""
    xy = players[:, :, :2] * np.asarray(pitch)
    speed = np.full(xy.shape[:2], np.nan)
    speed[FPS:] = np.hypot(*(xy[FPS:] - xy[:-FPS]).transpose(2, 0, 1))
    return np.nanmean(speed, axis=1)


def _edge_ratio(values, edges, runs):
    """Mean of values over the EDGE_WINDOW frames from each edge on, over the mean over the
    EDGE_WINDOW frames before it; edges whose windows leave their run are left out."""
    inside = [any(start + EDGE_WINDOW <= e <= stop - EDGE_WINDOW for start, stop in runs)
              for e in edges]
    edges = edges[np.array(inside, dtype=bool)]
    after = np.nanmean([values[e:e + EDGE_WINDOW] for e in edges])
    before = np.nanmean([values[e - EDGE_WINDOW:e] for e in edges])
    return float(after / before)


def _residual(xy, window):
    """xy minus its centred mean over window (odd) frames along axis 0; NaN where the window
    misses a value."""
    half = window // 2
    missing = np.isnan(xy)
    zeros = np.zeros((1,) + xy.shape[1:])
    sums = np.cumsum(np.concatenate((zeros, np.where(missing, 0.0, xy))), axis=0)
    gaps = np.cumsum(np.concatenate((zeros, missing)), axis=0)
    out = np.full(xy.shape, np.nan)
    centre = xy[half:len(xy) - half]
    out[half:len(xy) - half] = np.where(gaps[window:] == gaps[:-window],
                                        centre - (sums[window:] - sums[:-window]) / window, np.nan)
    return out


def _robust_sd(residual, inside):
    values = np.abs(residual[inside])
    return float(1.4826 * np.median(values[~np.isnan(values)]))


def player_statistics(players, pitch, runs):
    """players (n, slots, 3) in normalised coordinates; runs: [start, stop) frame spans."""
    inside = _inside(len(players), runs)
    xy = players[:, :, :2] * np.asarray(pitch)
    tracked = ~np.isnan(xy[:, :, 0])
    tracklets = np.concatenate([_run_lengths(tracked[:, slot], runs)
                                for slot in range(tracked.shape[1])])
    return {"tracked": float(tracked[inside].sum(axis=1).mean()),
            "tracklet_s": float(tracklets.mean() / FPS),
            "jitter_m": _robust_sd(_residual(xy, PLAYER_WINDOW), inside)}


def _filled(xy, seen, runs):
    """xy with every gap between two seen frames of a run filled linearly; NaN before the first
    and after the last seen frame of each run, and throughout a run without one (the fit may try
    knobs that lose the ball for a whole half)."""
    out = np.full(xy.shape, np.nan)
    for start, stop in runs:
        frames = np.flatnonzero(seen[start:stop]) + start
        if len(frames) == 0:
            continue
        span = np.arange(frames[0], frames[-1] + 1)
        for axis in range(xy.shape[1]):
            out[span, axis] = np.interp(span, frames, xy[frames, axis])
    return out


def ball_statistics(players, ball, pitch, runs):
    """ball (n, channels) and players (n, slots, 3) in normalised coordinates."""
    inside = _inside(len(ball), runs)
    xy = ball[:, :2] * np.asarray(pitch)
    seen = ~np.isnan(xy[:, 0])
    visible = seen & inside
    step = np.r_[np.nan, np.hypot(*np.diff(xy, axis=0).T)]  # from the frame before
    pairs = visible & np.r_[False, visible[:-1]]
    speed = np.r_[np.full(SPEED_LAG, np.nan),
                  np.hypot(*(xy[SPEED_LAG:] - xy[:-SPEED_LAG]).T) * FPS / SPEED_LAG]
    plausible = speed < JUMP_M * FPS  # False where either end is missing
    onsets = visible & np.r_[~seen[1:] & inside[1:], False]  # the last frame before a gap
    returns = np.flatnonzero(visible & np.r_[False, ~seen[:-1] & inside[:-1]]) + SPEED_LAG
    returns = returns[returns < len(ball)]  # SPEED_LAG frames after the first one after a gap
    filled = _filled(xy, seen, runs)
    people = players[:, :, :2] * np.asarray(pitch)
    crowd = (np.hypot(people[:, :, 0] - filled[:, None, 0], people[:, :, 1] - filled[:, None, 1])
             < CROWD_RADIUS).sum(axis=1)
    hidden = ~seen & inside & ~np.isnan(filled[:, 0])
    gaps = _run_lengths(~seen, runs)
    lost, found = _runs_of(~seen, runs)
    long = found - lost >= LONG_GAP_S * FPS
    team = _team_speed(players, pitch)
    return {"ball_missing": float(1 - seen[inside].mean()),
            "ball_gap_s": float(gaps.mean() / FPS),
            "ball_flicker": float(np.mean(gaps <= FLICKER_FRAMES)),
            "ball_long": float(gaps[gaps >= LONG_GAP_S * FPS].sum() / gaps.sum()),
            "ball_crowd_lift": float(crowd[hidden].mean() / crowd[visible].mean()),
            "ball_speed_lift": float(speed[onsets & plausible].mean()
                                     / speed[visible & plausible].mean()),
            "ball_return_lift": float(speed[returns][plausible[returns]].mean()
                                      / speed[visible & plausible].mean()),
            "loss_slowdown": _edge_ratio(team, lost[long], runs),
            "return_speedup": _edge_ratio(team, found[long], runs),
            "ball_jumps": float(np.mean(step[pairs] > JUMP_M)),
            "ball_jitter_m": _robust_sd(_residual(xy, BALL_WINDOW), inside)}


def statistics(players, ball, pitch, runs):
    return dict(player_statistics(players, pitch, runs),
                **ball_statistics(players, ball, pitch, runs))


def degraded_players(match, params, seed=SEED):
    rng = np.random.default_rng(seed)
    players, slot_of = degrade.fragment_players(match, params, rng)
    degrade.swap_tracklets(players, slot_of, match, params, rng)
    degrade.jitter(players, match.pitch, params, rng)
    return players


def degraded_ball(match, params, seed=SEED):
    return degrade.drop_ball(match, params, np.random.default_rng(seed + 1))


def mean_statistics(matches, params, seed=SEED):
    """The statistics averaged over degraded copies of the harness matches."""
    rows = [statistics(degraded_players(m, params, seed), degraded_ball(m, params, seed),
                       m.pitch, m.halves) for m in matches]
    return {name: float(np.mean([row[name] for row in rows])) for name in rows[0]}


def solve(statistic, target, low, high, steps=FIT_STEPS):
    """(knob value, reached): the value in [low, high] at which statistic(value), monotone in it,
    meets target, by bisection; the nearer end, and reached False, if the target lies beyond."""
    at_low, at_high = statistic(low), statistic(high)
    if (target - at_low) * (target - at_high) > 0:
        return (low if abs(target - at_low) < abs(target - at_high) else high), False
    rising = at_high > at_low
    for _ in range(steps):
        middle = (low + high) / 2
        if (statistic(middle) < target) == rising:
            low = middle
        else:
            high = middle
    return (low + high) / 2, True


def _knob_value(unit, low, high):
    """A knob from its position in [0, 1] along its range: geometric where the range spans
    orders of magnitude, so that a step means the same relative change anywhere along it."""
    return low * (high / low) ** unit if low > 0 and high / low > 10 else low + unit * (high - low)


def fit_dropout(matches, players, target, params, knobs=DROPOUT_KNOBS,
                statistics=DROPOUT_STATISTICS, seed=SEED):
    """(Params, {knob: reached}): the dropout knobs fitted jointly, by bounded least squares on
    the statistics' relative errors (one at a time does not converge). reached is False for a
    knob that ended at an end of its range."""
    names = list(knobs)
    scale = np.array([max(abs(target[name]), 1e-3) for name in statistics])

    def trial(units):
        return replace(params, **{knob: _knob_value(unit, *knobs[knob])
                                  for knob, unit in zip(names, units)})

    def residuals(units):
        rows = [ball_statistics(p, degraded_ball(m, trial(units), seed), m.pitch, m.halves)
                for m, p in zip(matches, players)]
        errors = np.array([np.mean([row[name] for row in rows]) - target[name]
                           for name in statistics]) / scale
        return np.where(np.isnan(errors), UNDEFINED, errors)

    start = []
    for knob in names:
        low, high = knobs[knob]
        value = min(max(getattr(params, knob), low), high)
        start.append(np.log(value / low) / np.log(high / low) if low > 0 and high / low > 10
                     else (value - low) / (high - low))
    for _ in range(FIT_RESTARTS):
        result = least_squares(residuals, start, bounds=(0.0, 1.0), diff_step=FIT_DIFF_STEP)
        start = result.x
    return trial(result.x), {knob: bool(active == 0)
                             for knob, active in zip(names, result.active_mask)}


def _one_at_a_time(params, knobs, statistic, target):
    """(Params, {knob: reached}) with each knob solved by bisection in turn; statistic(trial,
    name) is the statistic name under the trial Params."""
    reached = {}
    for name, (knob, low, high) in knobs.items():
        value, reached[knob] = solve(
            lambda value: statistic(replace(params, **{knob: value}), name), target[name],
            low, high)
        params = replace(params, **{knob: value})
    return params, reached


def fit(matches, target, seed=SEED, dropout_knobs=DROPOUT_KNOBS,
        dropout_statistics=DROPOUT_STATISTICS):
    """(Params, {knob: reached}): knobs set so that degraded copies of the harness matches show
    the target statistics. The player knobs barely interact and go one at a time; then the
    dropout knobs (together, fit_dropout, from DROPOUT_START) and the detection knobs (one at a
    time) take turns. The other knobs keep their defaults."""
    def player_statistic(trial, name):
        return np.mean([player_statistics(degraded_players(m, trial, seed), m.pitch,
                                          m.halves)[name] for m in matches])

    params = degrade.Params()
    for _ in range(FIT_PASSES):
        params, reached = _one_at_a_time(params, PLAYER_KNOBS, player_statistic, target)
    players = [degraded_players(m, params, seed) for m in matches]

    def ball_statistic(trial, name):
        return np.mean([ball_statistics(p, degraded_ball(m, trial, seed), m.pitch,
                                        m.halves)[name] for m, p in zip(matches, players)])

    params = replace(params, **DROPOUT_START)
    for _ in range(BALL_PASSES):
        params, dropout = fit_dropout(matches, players, target, params, dropout_knobs,
                                      dropout_statistics, seed)
        params, detection = _one_at_a_time(params, DETECTION_KNOBS, ball_statistic, target)
    return params, dict(reached, **dropout, **detection)
