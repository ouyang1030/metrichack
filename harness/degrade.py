"""ATD noise simulator: clean tracking -> tracking that looks like Metrica's ATD (AGENTS.md 5.2).

What the ATD does to the truth, per DATA.md:
  * the ball is missing on 41% of graded frames, in runs, disproportionately when it is in the
    air, in a crowd, moving fast or off the pitch;
  * there are no identities: 46 track slots, each holding tracklets that appear, vanish and are
    reused; people go untracked for stretches (20.5 tracked objects on average for 22 on the
    pitch), and now and then two tracklets swap people when players cross;
  * positions jitter, the ball's included.
Not in DATA.md, but what a ball detector does: now and then it reports something that is not the
ball -- a player's foot or head, a spot on or next to the pitch -- for a moment (false
detections; off by default until the calibration measures them). And what a tracker may do
when play stops: lose the ball for the whole stoppage, from about when it goes dead to about the
restart, but for brief sightings while it is placed or carried (lost stoppages; off by default,
the calibration decides how much of the missing ball they explain).

Params holds every knob. The defaults are guesses anchored on the numbers DATA.md publishes.
scripts/calibrate_noise.py fits the knobs the ATD can measure to the ATD (harness/calibrate.py)
and writes them to cache/aggregates/ under STRICT; calibrated() reads them for the harness, and
nobody looks at them. stressed() makes much noisier copies from either. Not modelled yet:
duplicated objects.
"""

import json
import os
from dataclasses import dataclass, fields, replace

import numpy as np

from mtah import atd, config
from evaluation import possession_metrics as pm

FPS = 25
CALIBRATED = os.path.join(config.CACHE_DIR, "aggregates", "noise_params.json")
SIGHTINGS = 64  # brief sightings drawn per stoppage; the rate decides how many fall inside it


@dataclass(frozen=True)
class Params:
    n_slots: int = 46
    ball_missing: float = 0.41          # share of in-play frames the dropout score takes the
                                        # ball from (DATA.md), besides lost stoppages
    ball_corr_s: float = 1.0            # correlation time of the dropout process
    ball_rough: float = 1.0             # its share of rough (exponential) noise; the rest smooth
    ball_weight_z: float = 1.0          # per metre of ball height
    ball_weight_crowd: float = 0.3      # per player within crowd_radius of the ball
    ball_weight_speed: float = 0.05     # per m/s of ball speed
    ball_weight_outside: float = 2.0    # ball off the pitch
    ball_weight_dead: float = 0.0       # ball dead (out of play, a stoppage)
    ball_dead_lost: float = 0.0         # share of stoppages in which the ball is lost throughout
    ball_dead_on_s: float = 0.0         # mean delay from the ball going dead to its loss
    ball_dead_off_s: float = 0.0        # mean delay from the restart to the ball found again
    ball_dead_lag_sd_s: float = 0.5     # spread of both delays from one stoppage to the next
    ball_dead_seen_per_min: float = 0.0  # brief sightings per minute of a lost stoppage
    ball_dead_seen_s: float = 1.0       # their mean length
    crowd_radius: float = 3.0           # m
    person_missing: float = 0.07        # share of a person's frames untracked (20.5 of 22)
    gap_mean_s: float = 1.5             # mean untracked stretch between two tracklets
    team_error: float = 0.01            # share of tracklets put in a slot of the other team
    swap_radius: float = 1.0            # m: players of opposite teams this close may swap
    swap_prob: float = 0.05             # per such encounter
    jitter_m: float = 0.2               # positional noise, standard deviation in metres
    jitter_corr_s: float = 0.5
    ball_jitter_m: float = 0.3          # the ball is small and fast: noisier, less correlated
    ball_jitter_corr_s: float = 0.2
    ball_false: float = 0.0             # share of in-play frames whose "ball" is something else
    ball_false_s: float = 0.5           # mean length of such a run
    ball_false_player: float = 0.5      # share of the runs on a player; the rest at a fixed spot
    ball_false_margin_m: float = 3.0    # how far beyond the lines a spot may lie


def smooth_noise(rng, shape, corr_frames):
    """Unit-variance Gaussian noise along axis 0, exponentially correlated over corr_frames."""
    kernel = np.exp(-np.arange(int(5 * corr_frames) + 1) / corr_frames)
    kernel /= np.sqrt((kernel ** 2).sum())
    white = rng.standard_normal((shape[0] + len(kernel) - 1,) + tuple(shape[1:]))
    return np.apply_along_axis(np.convolve, 0, white, kernel, "valid")


def smooth_gaussian(rng, n, corr_frames):
    """Unit-variance Gaussian noise of length n with a Gaussian kernel of corr_frames: unlike
    smooth_noise, its paths are smooth, so a threshold is crossed once, not in a burst."""
    half = int(4 * corr_frames) + 1
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / corr_frames) ** 2)
    return np.convolve(rng.standard_normal(n + 2 * half), kernel / np.sqrt((kernel ** 2).sum()),
                       "valid")


def drop_ball(match, params, rng):
    """The ball with params.ball_missing of the in-play frames set to NaN, in runs, besides the
    lost stoppages.

    False detections go first (false_detections()), then the lost stoppages
    (lost_in_stoppages()); of the other in-play frames, those whose difficulty plus correlated
    noise is highest lose the ball, as many as make params.ball_missing of all in-play frames.
    The noise mixes a rough process, which crosses the
    threshold in bursts of short gaps (a flickering detector), and a smooth one, in which a gap
    lasts as long as the noise stays high (a ball lost until it is found again). Each part draws
    from its own stream (rng.spawn), so that changing one knob changes only what that knob does
    (common random numbers, which the calibration's least squares relies on)."""
    dropout_rng, smooth_rng, jitter_rng, false_rng, stoppage_rng = rng.spawn(5)
    length, width = match.pitch
    ball_x, ball_y = match.ball[:, atd.X] * length, match.ball[:, atd.Y] * width
    near = np.hypot(match.players[:, :, atd.X] * length - ball_x[:, None],
                    match.players[:, :, atd.Y] * width - ball_y[:, None]) < params.crowd_radius
    outside = ((match.ball[:, atd.X] < 0) | (match.ball[:, atd.X] > 1)
               | (match.ball[:, atd.Y] < 0) | (match.ball[:, atd.Y] > 1))
    difficulty = (params.ball_weight_z * match.ball_z
                  + params.ball_weight_crowd * near.sum(axis=1)
                  + params.ball_weight_speed * match.ball[:, atd.SPEED]
                  + params.ball_weight_outside * outside
                  + params.ball_weight_dead * (match.labels == pm.D))
    noise = smooth_noise(dropout_rng, difficulty.shape, params.ball_corr_s * FPS)
    if params.ball_rough < 1:
        noise = (np.sqrt(params.ball_rough) * noise + np.sqrt(1 - params.ball_rough)
                 * smooth_gaussian(smooth_rng, len(noise), params.ball_corr_s * FPS))
    score = difficulty + noise
    in_play = match.labels != pm.X
    ball = match.ball.copy()
    false = false_detections(match, ball, params, false_rng)
    lost = lost_in_stoppages(match, params, stoppage_rng)
    candidates = in_play & ~false & ~lost
    share = min(params.ball_missing * in_play.sum() / candidates.sum(), 1.0)
    ball[((score > np.quantile(score[candidates], 1 - share)) | lost) & ~false] = np.nan
    noise = smooth_noise(jitter_rng, (len(ball), 2), params.ball_jitter_corr_s * FPS)
    ball[:, atd.X] += noise[:, 0] * params.ball_jitter_m / length
    ball[:, atd.Y] += noise[:, 1] * params.ball_jitter_m / width
    return ball


def lost_in_stoppages(match, params, rng):
    """The in-play frames on which the tracker has lost a dead ball: in params.ball_dead_lost of
    the stoppages, from about params.ball_dead_on_s after the ball goes dead to about
    params.ball_dead_off_s after the restart, but for brief sightings (the ball placed, picked up,
    carried) at params.ball_dead_seen_per_min. Every draw is made per stoppage whatever the
    knobs, so that a knob moves only what it controls (common random numbers)."""
    dead = np.r_[False, match.labels == pm.D, False]
    edges = np.flatnonzero(dead[1:] != dead[:-1])
    starts, stops = edges[::2], edges[1::2]
    chosen = rng.random(len(starts)) < params.ball_dead_lost
    lags = params.ball_dead_lag_sd_s * rng.standard_normal((len(starts), 2))
    first = starts + np.round((params.ball_dead_on_s + lags[:, 0]) * FPS).astype(int)
    last = stops + np.round((params.ball_dead_off_s + lags[:, 1]) * FPS).astype(int)
    arrivals = np.cumsum(rng.exponential(1.0, (len(starts), SIGHTINGS)), axis=1)
    lengths = np.maximum(1, np.round(rng.exponential(params.ball_dead_seen_s * FPS,
                                                     (len(starts), SIGHTINGS)))).astype(int)
    lost = np.zeros(len(match.labels), dtype=bool)
    per_frame = params.ball_dead_seen_per_min / (60 * FPS)
    for i in np.flatnonzero(chosen & (last > first)):
        lost[max(first[i], 0):last[i]] = True
        if per_frame > 0:
            for begin, length in zip(first[i] + (arrivals[i] / per_frame).astype(int), lengths[i]):
                if begin >= last[i]:
                    break
                lost[begin:begin + length] = False
    return lost & (match.labels != pm.X)


def false_detections(match, ball, params, rng):
    """Overwrite about params.ball_false of the in-play frames of ball, in runs, with something
    that is not the ball: a player (the ball sits at his feet and follows him) or a fixed spot
    on the pitch or just beyond its lines. Returns the mask of the frames overwritten.

    The runs are the first ones of a fixed pool, as many as would cover every in-play frame, so
    that more false detections add runs rather than move them."""
    length, width = match.pitch
    margin = params.ball_false_margin_m
    in_play = np.flatnonzero(match.labels != pm.X)
    mean = params.ball_false_s * FPS
    pool = int(len(in_play) / mean)
    starts = rng.choice(in_play, pool)
    stops = np.minimum(starts + 1 + rng.exponential(mean - 1, pool).astype(int), len(ball))
    on_player, pick = rng.random(pool) < params.ball_false_player, rng.random(pool)
    offsets = rng.normal(0, 0.3, (pool, 2)) / match.pitch
    spots = (rng.random((pool, 2)) * (np.array(match.pitch) + 2 * margin) - margin) / match.pitch
    false = np.zeros(len(ball), dtype=bool)
    for i in range(int(round(params.ball_false * len(in_play) / mean))):
        start, stop = starts[i], stops[i]
        false[start:stop] = True
        people = np.flatnonzero(~np.isnan(match.players[start, :, atd.X]))
        if on_player[i] and len(people):
            ball[start:stop] = match.players[start:stop, people[int(pick[i] * len(people))]]
            ball[start:stop, :2] += offsets[i]
        else:
            ball[start:stop, :2] = spots[i]
            ball[start:stop, atd.SPEED] = 0.0
    return false


def fragment_players(match, params, rng):
    """(players (n, n_slots, 3), slot_of (n, n_persons)): every person cut into tracklets.

    Tracklets alternate with untracked gaps so that params.person_missing of a person's frames
    are lost. Each tracklet goes to the first free slot of its team's half of the slots (of the
    other team's, with probability params.team_error). slot_of is -1 while untracked.
    """
    n, n_persons, _ = match.players.shape
    per_team = params.n_slots // 2
    gap = params.gap_mean_s * FPS
    run = gap * (1 - params.person_missing) / params.person_missing

    pieces = []
    for person in range(n_persons):
        on = np.concatenate(([0], ~np.isnan(match.players[:, person, atd.X]), [0]))
        edges = np.flatnonzero(np.diff(on.astype(np.int8)))
        for start, stop in zip(edges[::2], edges[1::2]):
            t = start + (int(rng.exponential(gap)) if rng.random() < params.person_missing else 0)
            while t < stop:
                end = min(t + 1 + int(rng.exponential(run)), stop)
                pieces.append((t, end, person))
                t = end + int(rng.exponential(gap))
    pieces.sort()

    players = np.full((n, params.n_slots, 3), np.nan)
    slot_of = np.full((n, n_persons), -1, dtype=np.int16)
    busy_until = np.zeros(params.n_slots, dtype=int)
    for start, stop, person in pieces:
        team = match.slot_team[person] ^ (rng.random() < params.team_error)
        candidates = np.arange(team * per_team, (team + 1) * per_team)
        slot = candidates[np.argmax(busy_until[candidates] <= start)]
        busy_until[slot] = stop
        players[start:stop, slot] = match.players[start:stop, person]
        slot_of[start:stop, person] = slot
    return players, slot_of


def _tracklet_end(column):
    """Frames until the first NaN of a slot's x, or its length."""
    missing = np.isnan(column)
    return int(np.argmax(missing)) if missing.any() else len(column)


def swap_tracklets(players, slot_of, match, params, rng):
    """At close encounters of opposite-team players, their tracklets swap slots until one ends."""
    length, width = match.pitch
    for p in np.flatnonzero(match.slot_team == atd.HOME):
        for q in np.flatnonzero(match.slot_team == atd.AWAY):
            close = np.hypot(
                (match.players[:, p, atd.X] - match.players[:, q, atd.X]) * length,
                (match.players[:, p, atd.Y] - match.players[:, q, atd.Y]) * width,
            ) < params.swap_radius
            onsets = np.flatnonzero(close[1:] & ~close[:-1]) + 1
            for t in onsets[rng.random(len(onsets)) < params.swap_prob]:
                a, b = slot_of[t, p], slot_of[t, q]
                if a < 0 or b < 0:
                    continue  # one of the two is untracked at that moment
                end = t + min(_tracklet_end(players[t:, a, atd.X]),
                              _tracklet_end(players[t:, b, atd.X]))
                players[t:end, [a, b]] = players[t:end, [b, a]]


def jitter(players, pitch, params, rng):
    noise = smooth_noise(rng, players.shape[:2] + (2,), params.jitter_corr_s * FPS)
    players[:, :, atd.X] += noise[..., 0] * params.jitter_m / pitch[0]
    players[:, :, atd.Y] += noise[..., 1] * params.jitter_m / pitch[1]


def degrade(match, seed, params=Params()):
    """An ATD-like copy of a clean harness match. Labels are untouched; ball height is dropped."""
    rng = np.random.default_rng(seed)
    ball = drop_ball(match, params, rng)
    players, slot_of = fragment_players(match, params, rng)
    swap_tracklets(players, slot_of, match, params, rng)
    jitter(players, match.pitch, params, rng)
    slot_team = np.repeat(np.array([atd.HOME, atd.AWAY], dtype=np.int8), params.n_slots // 2)
    return replace(match, players=players, slot_team=slot_team, ball=ball, ball_z=None)


def drawn(low, high, rng):
    """Params with every field that differs between low and high drawn uniformly between the two
    (domain randomisation: train under a range of noise, not one guess of it)."""
    return replace(low, **{field.name: a + (b - a) * rng.random() for field in fields(Params)
                           for a, b in [(getattr(low, field.name), getattr(high, field.name))]
                           if a != b})


def calibrated():
    """Params fitted to the ATD by scripts/calibrate_noise.py. STRICT: used, never printed."""
    with open(CALIBRATED) as handle:
        return Params(**json.load(handle))


def stressed(params):
    """Much noisier than params (about the ratios of the first STRESS copy to the defaults): a
    check that a gain does not rest on the exact noise a classifier was trained under. The ball
    is no more likely to vanish off the pitch than elsewhere; where stoppages are lost, it is
    lost and found again at less regular times."""
    return replace(params, ball_missing=min(params.ball_missing + 0.14, 0.9),
                   ball_dead_lag_sd_s=2 * params.ball_dead_lag_sd_s,
                   ball_dead_seen_per_min=2 * params.ball_dead_seen_per_min,
                   ball_weight_outside=0.0, ball_weight_crowd=2 * params.ball_weight_crowd,
                   person_missing=min(2 * params.person_missing, 0.5),
                   gap_mean_s=2 * params.gap_mean_s, team_error=5 * params.team_error,
                   swap_radius=2 * params.swap_radius, swap_prob=4 * params.swap_prob,
                   jitter_m=2 * params.jitter_m, ball_jitter_m=2 * params.ball_jitter_m,
                   ball_false=3 * params.ball_false)
