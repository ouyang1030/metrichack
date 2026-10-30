"""Per-frame features from tracking, computed the same way on BAR-ATH and on the harness.

Only what the ATD provides is used: slot positions and speeds, slot teams, ball x and y. Ball
speed and height are not used (the ATD's ball channels beyond x, y are not documented).

compute() feeds the v1 logistic models; matrix() is the wider table the v2 frame classifier
reads.
"""

from dataclasses import dataclass

import numpy as np

from mtah import atd

FPS = 25
OUT_TOLERANCE = 0.5  # m beyond a line before a ball counts as out (AGENTS.md section 7)
SPEED_WINDOW = 25    # frames each side: the collective speed is a centred 2 s mean
VELOCITY_LAG = 2     # frames each side of a central difference
SIDE_WINDOW = 25 * 120  # frames each side: which way each team attacks, from a 4 min window
CLOSE_RADIUS = 1.5   # m: scale of the closeness exp(-d / r), as in emissions.CONTROL_RADIUS
COUNT_RADII = (2.0, 5.0, 10.0)  # m: players of a team this close to the ball
WINDOWS = (12, 25, 50)  # frames each side of the centred means: about 0.5, 1 and 2 s
STILL_SPEED = 1.0    # m/s: below this a player or the ball counts as still
TOUCH_RADIUS = 1.5   # m: a change of the ball's velocity this close to a player is his touch
TOUCH_LAG = 5        # frames: velocity before vs after, over 0.2 s each
HOLD_WINDOW = 25     # frames: how firmly a team holds the ball before it lets go, after it gets it
CONTEXT_WINDOWS = (12, 25, 50, 100, 200)  # frames before / after, for context(): 0.5 to 8 s


@dataclass
class Features:
    speed: np.ndarray      # median speed of the tracked players, m/s, centred 2 s mean
    seen: np.ndarray       # ball located on this frame
    last_out: np.ndarray   # the last located ball was beyond a line
    gap: np.ndarray        # seconds since the ball was last located
    near_home: np.ndarray  # distance from the ball to the nearest home slot, m (inf if unknown)
    near_away: np.ndarray
    rel_home: np.ndarray   # |ball velocity - that slot's velocity|, m/s (inf if unknown)
    rel_away: np.ndarray
    ball_speed: np.ndarray  # m/s from ball positions (NaN if unknown)
    flow: np.ndarray       # mean player velocity towards the away goal, m/s, 2 s mean
    spread: np.ndarray     # home minus away spread along the pitch, m, 2 s mean


def window_mean(values, before, after):
    """Mean over frames t - before ... t + after, the ends padded with the edge values."""
    kernel = np.full(before + after + 1, 1.0 / (before + after + 1))
    return np.convolve(np.pad(values, (before, after), mode="edge"), kernel, "valid")


def centred_mean(values, window):
    return window_mean(values, window, window)


def window_nanmean(values, before, after):
    """window_mean over the known values only; NaN where a window has none."""
    known = ~np.isnan(values)
    count = window_mean(known.astype(float), before, after)
    total = window_mean(np.where(known, values, 0.0), before, after)
    return np.where(count > 0, total / np.maximum(count, 1e-9), np.nan)


def centred_nanmean(values, window):
    """centred_mean over the known values only; 0 where a window has none."""
    known = ~np.isnan(values)
    total = centred_mean(np.where(known, values, 0.0), window)
    return total / np.maximum(centred_mean(known.astype(float), window), 1e-9)


def _row_mean(values):
    """Mean of each row over its known values; NaN for an empty row."""
    known = ~np.isnan(values)
    count = known.sum(axis=1)
    mean = np.where(known, values, 0.0).sum(axis=1) / np.maximum(count, 1)
    return np.where(count > 0, mean, np.nan)


def _row_std(values):
    mean = _row_mean(values)
    return np.sqrt(np.maximum(_row_mean(values ** 2) - mean ** 2, 0.0))


def velocity(x, y):
    """(vx, vy) in m/s along axis 0 by central differences; NaN next to gaps and at the ends."""
    vx, vy = np.full(x.shape, np.nan), np.full(y.shape, np.nan)
    lag = VELOCITY_LAG
    vx[lag:-lag] = (x[2 * lag:] - x[:-2 * lag]) * FPS / (2 * lag)
    vy[lag:-lag] = (y[2 * lag:] - y[:-2 * lag]) * FPS / (2 * lag)
    return vx, vy


def _nearest(distance, values, mask):
    """Per frame, values of the slot in mask that is nearest the ball."""
    index = np.where(mask, distance, np.inf).argmin(axis=1)
    return values[np.arange(len(values)), index]


def compute(players, ball, slot_team, pitch):
    length, width = pitch
    n = len(ball)
    speeds = players[:, :, atd.SPEED]
    tracked = ~np.isnan(speeds).all(axis=1)
    median = np.zeros(n)
    median[tracked] = np.nanmedian(speeds[tracked], axis=1)

    ball_x, ball_y = ball[:, atd.X] * length, ball[:, atd.Y] * width
    seen = ~np.isnan(ball_x)
    out = seen & ((ball_x < -OUT_TOLERANCE) | (ball_x > length + OUT_TOLERANCE)
                  | (ball_y < -OUT_TOLERANCE) | (ball_y > width + OUT_TOLERANCE))
    last_seen = np.maximum.accumulate(np.where(seen, np.arange(n), 0))

    player_x, player_y = players[:, :, atd.X] * length, players[:, :, atd.Y] * width
    distance = np.hypot(player_x - ball_x[:, None], player_y - ball_y[:, None])
    distance[np.isnan(distance)] = np.inf
    ball_vx, ball_vy = velocity(ball_x, ball_y)
    player_vx, player_vy = velocity(player_x, player_y)
    relative = np.hypot(player_vx - ball_vx[:, None], player_vy - ball_vy[:, None])
    relative[np.isnan(relative)] = np.inf
    home, away = slot_team == atd.HOME, slot_team == atd.AWAY
    # Home attacks towards larger x when, over a few minutes, it sits on the smaller-x side.
    towards_away = np.sign(centred_nanmean(_row_mean(player_x[:, away])
                                           - _row_mean(player_x[:, home]), SIDE_WINDOW))
    return Features(
        speed=centred_mean(median, SPEED_WINDOW),
        seen=seen,
        last_out=out[last_seen],
        gap=(np.arange(n) - last_seen) / FPS,
        near_home=np.where(home, distance, np.inf).min(axis=1),
        near_away=np.where(away, distance, np.inf).min(axis=1),
        rel_home=_nearest(distance, relative, home),
        rel_away=_nearest(distance, relative, away),
        ball_speed=np.hypot(ball_vx, ball_vy),
        flow=towards_away * centred_nanmean(_row_mean(player_vx), SPEED_WINDOW),
        spread=centred_nanmean(_row_std(player_x[:, home]) - _row_std(player_x[:, away]),
                               SPEED_WINDOW),
    )


def run_length(flags):
    """(frames the flag has held up to and including t, frames it holds from t on); 0 where off."""
    frames = np.arange(len(flags))
    past = frames - np.maximum.accumulate(np.where(flags, -1, frames))
    reverse = flags[::-1]
    future = (frames - np.maximum.accumulate(np.where(reverse, -1, frames)))[::-1]
    return np.where(flags, past, 0), np.where(flags, future, 0)


def _team_min(values, mask):
    """Per frame, the smallest value over the slots in mask; NaN if none is known."""
    least = np.where(mask & ~np.isnan(values), values, np.inf).min(axis=1)
    return np.where(np.isinf(least), np.nan, least)


def matrix(players, ball, slot_team, pitch):
    """((n, k) float32 features for the frame classifier, k names).

    Each team is measured towards the goal it attacks and every team quantity comes as a home /
    away pair, so that swapping the teams only permutes the columns (swap_index). Unknown is
    NaN. Windows look ahead as well as back: the pipeline runs offline.
    """
    length, width = pitch
    n = len(ball)
    frames = np.arange(n)
    player_x, player_y = players[:, :, atd.X] * length, players[:, :, atd.Y] * width
    player_vx, player_vy = velocity(player_x, player_y)
    speeds = players[:, :, atd.SPEED]
    tracked = ~np.isnan(speeds)
    median = np.zeros(n)
    median[tracked.any(axis=1)] = np.nanmedian(speeds[tracked.any(axis=1)], axis=1)

    ball_x, ball_y = ball[:, atd.X] * length, ball[:, atd.Y] * width
    seen = ~np.isnan(ball_x)
    out = seen & ((ball_x < -OUT_TOLERANCE) | (ball_x > length + OUT_TOLERANCE)
                  | (ball_y < -OUT_TOLERANCE) | (ball_y > width + OUT_TOLERANCE))
    last = np.maximum.accumulate(np.where(seen, frames, 0))
    following = np.minimum.accumulate(np.where(seen, frames, n - 1)[::-1])[::-1]
    last_x, last_y = ball_x[last], ball_y[last]  # where the ball was last located
    ball_vx, ball_vy = velocity(ball_x, ball_y)
    ball_speed = np.hypot(ball_vx, ball_vy)
    distance = np.hypot(player_x - ball_x[:, None], player_y - ball_y[:, None])
    last_distance = np.hypot(player_x - last_x[:, None], player_y - last_y[:, None])
    relative = np.hypot(player_vx - ball_vx[:, None], player_vy - ball_vy[:, None])
    home, away = slot_team == atd.HOME, slot_team == atd.AWAY
    towards_away = np.sign(centred_nanmean(_row_mean(player_x[:, away])
                                           - _row_mean(player_x[:, home]), SIDE_WINDOW))

    columns = {}

    def windows(name, values):
        for window in WINDOWS:
            columns["%s_c%d" % (name, window)] = window_nanmean(values, window, window)
        columns[name + "_past"] = window_nanmean(values, WINDOWS[-1], 0)
        columns[name + "_next"] = window_nanmean(values, 0, WINDOWS[-1])

    columns["speed"] = median
    windows("speed", median)
    columns["still_share"] = window_mean((tracked & (speeds < STILL_SPEED)).sum(axis=1)
                                         / np.maximum(tracked.sum(axis=1), 1), 25, 25)
    columns["tracked"] = tracked.sum(axis=1)
    columns["seen"] = seen
    columns["seen_c25"] = window_mean(seen.astype(float), 25, 25)
    columns["since_seen"] = (frames - last) / FPS
    columns["until_seen"] = (following - frames) / FPS
    columns["last_out"] = out[last]
    columns["next_out"] = out[following]
    columns["ball_speed"] = ball_speed
    columns["ball_speed_c12"] = window_nanmean(ball_speed, 12, 12)
    columns["ball_still"] = window_nanmean(np.where(np.isnan(ball_speed), np.nan,
                                                    ball_speed < STILL_SPEED), 25, 0)
    columns["touchline"] = np.minimum(last_y, width - last_y)
    columns["goal_line"] = np.minimum(last_x, length - last_x)
    columns["nearest"] = _team_min(distance, home | away)
    # Touches (Vidal-Codina et al. 2022; Link & Hoernig 2017): control and deflection both change
    # the ball's velocity next to a player; who touches it, how often, and whether an opponent is
    # as close (a duel) is what tells one from the other.
    def change(v):
        return window_nanmean(v, 0, TOUCH_LAG) - window_nanmean(v, TOUCH_LAG, 0)

    impulse = np.hypot(change(ball_vx), change(ball_vy))
    columns["impulse"] = impulse
    teams = (("home", home), ("away", away))
    nears = {team: _team_min(distance, mask) for team, mask in teams}
    lasts = {team: _team_min(last_distance, mask) for team, mask in teams}
    duel = (nears["home"] < TOUCH_RADIUS) & (nears["away"] < TOUCH_RADIUS)
    for window in WINDOWS[:2]:
        columns["duel_c%d" % window] = window_mean(duel.astype(float), window, window)

    for team, mask, sign in (("home", home, towards_away), ("away", away, -towards_away)):
        near = _team_min(distance, mask)
        columns["near_" + team] = near
        columns["near_last_" + team] = _team_min(last_distance, mask)
        windows("close_" + team, np.exp(-near / CLOSE_RADIUS))
        for radius in COUNT_RADII:
            columns["count%g_%s" % (radius, team)] = np.where(
                seen, (np.where(mask, distance, np.inf) < radius).sum(axis=1), np.nan)
        rel = _nearest(np.where(np.isnan(distance), np.inf, distance), relative, mask)
        columns["rel_" + team] = np.where(np.isnan(near), np.nan, rel)
        columns["team_speed_" + team] = window_nanmean(_row_mean(speeds[:, mask]), 25, 25)
        columns["depth_" + team] = window_nanmean(
            sign * (_row_mean(player_x[:, mask]) - length / 2), 25, 25)
        windows("push_" + team, sign * _row_mean(player_vx[:, mask]))
        columns["length_" + team] = window_nanmean(_row_std(player_x[:, mask]), 25, 25)
        columns["width_" + team] = window_nanmean(_row_std(player_y[:, mask]), 25, 25)
        columns["goal_dist_" + team] = np.hypot(last_x - length / 2 * (1 + sign),
                                                last_y - width / 2)
        columns["ball_v_" + team] = sign * ball_vx
        columns["ball_v_c12_" + team] = window_nanmean(sign * ball_vx, 12, 12)
        other = nears["away" if team == "home" else "home"]
        touch = np.where((near < TOUCH_RADIUS) & ~(other < near), np.nan_to_num(impulse), 0.0)
        for window in WINDOWS:
            columns["touch_b%d_%s" % (window, team)] = window_mean(touch, window, 0)
            columns["touch_a%d_%s" % (window, team)] = window_mean(touch, 0, window)
        lead = lasts[team] < lasts["away" if team == "home" else "home"]
        past, future = run_length(lead)
        columns["lead_past_" + team] = past / FPS
        columns["lead_next_" + team] = future / FPS

    # Receptions and interceptions (plan step 4b): which team touched the ball last and which
    # touches it next (a player within TOUCH_RADIUS, no opponent nearer), how firmly each held it
    # before letting go and holds it after getting it, and how far the ball went in between. A
    # pass to a team-mate keeps the possession; an opponent who gets the ball and holds it takes it.
    contacts = {team: seen & (nears[team] < TOUCH_RADIUS)
                & ~(nears["away" if team == "home" else "home"] < nears[team])
                for team, _ in teams}
    contact = contacts["home"] | contacts["away"]
    previous = np.maximum.accumulate(np.where(contact, frames, -1))
    upcoming = np.minimum.accumulate(np.where(contact, frames, n)[::-1])[::-1]
    known_previous, known_upcoming = previous >= 0, upcoming < n
    previous, upcoming = np.maximum(previous, 0), np.minimum(upcoming, n - 1)
    columns["since_contact"] = np.where(known_previous, (frames - previous) / FPS, np.nan)
    columns["until_contact"] = np.where(known_upcoming, (upcoming - frames) / FPS, np.nan)
    columns["flight_m"] = np.where(known_previous & known_upcoming,
                                   np.hypot(ball_x[upcoming] - ball_x[previous],
                                            ball_y[upcoming] - ball_y[previous]), np.nan)
    for team, _ in teams:
        held = contacts[team].astype(float)
        columns["contact_last_" + team] = np.where(known_previous, held[previous], np.nan)
        columns["contact_next_" + team] = np.where(known_upcoming, held[upcoming], np.nan)
        columns["hold_last_" + team] = np.where(
            known_previous, window_mean(held, HOLD_WINDOW, 0)[previous], np.nan)
        columns["hold_next_" + team] = np.where(
            known_upcoming, window_mean(held, 0, HOLD_WINDOW)[upcoming], np.nan)

    names = list(columns)
    return np.column_stack([np.asarray(columns[name], float) for name in names]).astype(
        np.float32), names


def swap_index(names):
    """Column order of the same frames with home and away exchanged."""
    swapped = [name.replace("home", "#").replace("away", "home").replace("#", "away")
               for name in names]
    return np.array([names.index(name) for name in swapped])


def context(columns):
    """(X, names) for a second stage: each named per-frame column (first-stage outputs) as it is,
    its means over the CONTEXT_WINDOWS frames before and after, and after minus before -- what
    changes around the frame, up to 8 s away."""
    out, names = [], []
    for name, values in columns.items():
        out.append(values)
        names.append(name)
        for window in CONTEXT_WINDOWS:
            before, after = window_mean(values, window, 0), window_mean(values, 0, window)
            out += [before, after, after - before]
            names += ["%s_b%d" % (name, window), "%s_a%d" % (name, window),
                      "%s_d%d" % (name, window)]
    return np.stack(out, axis=1).astype(np.float32), names


FIRST_STAGE = ("p_home", "p_away", "p_dead")  # classifier outputs, then the change detector's
CHANGES = ("change_switch", "change_stop", "change_restart")


def second_stage(X, probs, change):
    """(table, names of the added columns): the frame's own features, then context() of the
    first stage's P(H, A, D) and of the change detector's P(switch, stop, restart)."""
    added, names = context(dict(zip(FIRST_STAGE, probs.T)) | dict(zip(CHANGES, change[:, 1:].T)))
    return np.hstack((X, added)), names
