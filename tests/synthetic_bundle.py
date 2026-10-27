"""Synthetic data bundles in the exact file formats of the BAR-ATH bundle, for tests and dry runs.

Nothing here comes from the match: every position is random. The formats follow DATA.md, the
starter loaders and the FIFA/EPTS metadata structure kloppy reads.

    python tests/synthetic_bundle.py <dir>        # full length (199,319 frames), for a dry run

Values are drawn as k / 10**d so that writing them with d decimals and reading them back with
float() returns exactly the drawn double; tests can then compare parsed arrays for equality.
"""

import argparse
import os

import numpy as np

STEM = "BAR-ATH_tactical"
N_SLOTS = 46
FPS = 25
TEAM_IDS = ("1001", "1002")
CHANNELS = ("x", "y", "speed")
FULL_LENGTH = 199319


def write_meta(path, n_frames, slot_team, spec_order=None):
    n_slots = len(slot_team)
    spec_order = list(range(n_slots)) if spec_order is None else list(spec_order)
    players = "\n".join(
        '      <Player id="P{0}" teamId="{1}"><Name>Track {0}</Name>'
        '<ShirtNumber>{0}</ShirtNumber></Player>'.format(i + 1, TEAM_IDS[team])
        for i, team in enumerate(slot_team))
    player_channels = "\n".join(
        '      <PlayerChannel id="player{0}_{1}" playerId="P{0}" channelId="{1}"/>'.format(
            i + 1, channel) for i in range(n_slots) for channel in CHANNELS)
    groups = "\n".join(
        '        <SplitRegister separator=",">' + "".join(
            '<PlayerChannelRef playerChannelId="player{}_{}"/>'.format(i + 1, channel)
            for channel in CHANNELS) + "</SplitRegister>" for i in spec_order)
    ball = '      <SplitRegister separator=",">' + "".join(
        '<BallChannelRef channelId="{}"/>'.format(c) for c in CHANNELS) + "</SplitRegister>"
    with open(path, "w") as handle:
        handle.write("""<?xml version="1.0" encoding="utf-8"?>
<main>
  <Metadata>
    <GlobalConfig><FrameRate>{fps}</FrameRate><ProviderName>Metrica Sports</ProviderName></GlobalConfig>
    <Sessions>
      <Session id="1">
        <FieldSize><Width>105</Width><Height>68</Height></FieldSize>
        <MatchParameters><Score idLocalTeam="{home}" idVisitingTeam="{away}"><LocalTeamScore>0</LocalTeamScore><VisitingTeamScore>0</VisitingTeamScore></Score></MatchParameters>
      </Session>
    </Sessions>
    <Teams>
      <Team id="{home}"><Name>Home Synthetic</Name></Team>
      <Team id="{away}"><Name>Away Synthetic</Name></Team>
    </Teams>
    <Players>
{players}
    </Players>
    <Devices><Device id="device1"><Name>Device 1</Name><Sensors>
      <Sensor id="position"><Name>Position</Name><Channels>
        <Channel id="x"><Name>x</Name><Unit>normalized</Unit></Channel>
        <Channel id="y"><Name>y</Name><Unit>normalized</Unit></Channel>
      </Channels></Sensor>
      <Sensor id="speed"><Name>Speed</Name><Channels>
        <Channel id="speed"><Name>speed</Name><Unit>meters per second</Unit></Channel>
      </Channels></Sensor>
    </Sensors></Device></Devices>
    <PlayerChannels>
{player_channels}
    </PlayerChannels>
  </Metadata>
  <DataFormatSpecifications>
    <DataFormatSpecification separator=":" startFrame="1" endFrame="{n_frames}">
      <StringRegister name="frameCount"/>
      <SplitRegister separator=";">
{groups}
      </SplitRegister>
{ball}
    </DataFormatSpecification>
  </DataFormatSpecifications>
</main>
""".format(fps=FPS, home=TEAM_IDS[0], away=TEAM_IDS[1], players=players,
           player_channels=player_channels, n_frames=n_frames, groups=groups, ball=ball))


def random_tracking(n_frames, n_slots=N_SLOTS, seed=0, p_slot=0.45, p_ball=0.57):
    """(players (n, slots, 3), ball (n, 3)) with NaN for missing, exactly text-representable."""
    rng = np.random.default_rng(seed)
    players = np.stack([rng.integers(0, 100001, (n_frames, n_slots)) / 100000.0,
                        rng.integers(0, 100001, (n_frames, n_slots)) / 100000.0,
                        rng.integers(0, 1001, (n_frames, n_slots)) / 100.0], axis=2)
    players[rng.random((n_frames, n_slots)) > p_slot] = np.nan
    ball = np.stack([rng.integers(0, 100001, n_frames) / 100000.0,
                     rng.integers(0, 100001, n_frames) / 100000.0,
                     rng.integers(0, 3001, n_frames) / 100.0], axis=1)
    ball[rng.random(n_frames) > p_ball] = np.nan
    return players, ball


def add_edge_cases(players, ball):
    """Plant every case where starter/baseline.py's loop has a non-obvious answer."""
    ball[:60] = [0.5, 0.5, 1.0]
    players[:60] = np.nan
    for f in range(0, 10):            # exact tie between slots 3 and 7 -> the first one
        players[f, 3] = players[f, 7] = [0.4, 0.4, 1.0]
        players[f, 12] = [0.9, 0.9, 1.0]
    for f in range(10, 20):           # first usable slot has x but no y -> it wins
        players[f, 2] = [0.1, np.nan, 1.0]
        players[f, 5] = [0.5, 0.5, 1.0]
    for f in range(20, 30):           # a later slot with x but no y -> it never wins
        players[f, 2] = [0.45, 0.45, 1.0]
        players[f, 9] = [0.5, np.nan, 1.0]
        players[f, 11] = [0.49, 0.49, 1.0]
    for f in range(30, 40):           # ball x without y -> the first usable slot wins
        ball[f, 1] = np.nan
        players[f, 4] = [0.9, 0.9, 1.0]
        players[f, 8] = [0.5, 0.5, 1.0]
    # 40..49: ball present, no usable slot at all -> nobody
    for f in range(50, 60):           # slot with y but no x is skipped
        players[f, 1] = [np.nan, 0.5, 1.0]
        players[f, 6] = [0.6, 0.6, 1.0]
    return players, ball


def raw_lines(players, ball, spec_order=None):
    """One raw ATD line per frame; slot groups in spec order, NaN written as 'NaN'."""
    n_frames, n_slots, _ = players.shape
    order = np.arange(n_slots) if spec_order is None else np.asarray(spec_order)
    template = "%d:" + ";".join(["%.5f,%.5f,%.2f"] * n_slots) + ":%.5f,%.5f,%.2f\n"
    ordered = players[:, order, :].reshape(n_frames, -1)
    return [(template % ((f + 1,) + tuple(ordered[f]) + tuple(ball[f]))).replace("nan", "NaN")
            for f in range(n_frames)]


def write_smart_tagging(path, instances):
    """SportsCode XML: [(code, start_s, end_s, {group: text})]."""
    parts = ["<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<file>\n  <ALL_INSTANCES>\n"]
    for index, (code, start, end, tags) in enumerate(instances, start=1):
        labels = "".join("<label><group>{}</group><text>{}</text></label>".format(g, t)
                         for g, t in tags.items())
        parts.append("    <instance><ID>{}</ID><start>{:.2f}</start><end>{:.2f}</end>"
                     "<code>{}</code>{}</instance>\n".format(index, start, end, code, labels))
    parts.append("  </ALL_INSTANCES>\n</file>\n")
    with open(path, "w") as handle:
        handle.write("".join(parts))


def default_tags():
    return [
        ("SET PIECES", 11185 / FPS - 2, 11185 / FPS + 3,
         {"Team": "Barcelona", "Type": "Kick Off (Start)", "Half": "1st Half"}),
        ("BUILD UP", 11185 / FPS + 3, 11185 / FPS + 20, {"Team": "Barcelona", "Half": "1st Half"}),
        ("SET PIECES", 20000 / FPS, 20000 / FPS + 4,
         {"Team": "N/A", "Type": "Throw-in", "Half": "1st Half"}),
        ("SET PIECES", 113742 / FPS - 2, 113742 / FPS + 3,
         {"Team": "Atletic Club", "Type": "Kick Off (Start)", "Half": "2nd Half"}),
    ]


def write_homography(path, n_frames, seed=0):
    rng = np.random.default_rng(seed)
    base = np.array([1800.0, -500.0, 45.0, 1.7, 170.0, 255.0, -0.02, -0.5, 1.0])
    with open(path, "w") as handle:
        handle.write("frame," + ",".join("h{}{}".format(r, c) for r in range(3)
                                         for c in range(3)) + ",reprojection_error,valid\n")
        noise = rng.normal(0, 0.01, (n_frames, 9))
        error = rng.random(n_frames)
        for f in range(n_frames):
            handle.write("%d,%s,%.4f,1\n" % (f, ",".join("%.6g" % v for v in base + noise[f]),
                                            error[f]))


def write_bundle(data_dir, n_frames, seed=0, edge_cases=False, spec_order=None):
    """Write data_dir/tactical/* and return what was written."""
    rng = np.random.default_rng(seed + 1)
    slot_team = rng.permutation(np.repeat(np.array([0, 1], dtype=np.int8), N_SLOTS // 2))
    players, ball = random_tracking(n_frames, seed=seed)
    if edge_cases:
        players, ball = add_edge_cases(players, ball)

    folder = os.path.join(data_dir, "tactical")
    os.makedirs(folder, exist_ok=True)
    paths = {name: os.path.join(folder, STEM + suffix) for name, suffix in (
        ("atd_meta", "_atd.xml"), ("atd_raw", "_atd.txt"), ("homography", "_homography.csv"),
        ("smart_tagging", "_smart_tagging.xml"))}

    write_meta(paths["atd_meta"], n_frames, slot_team, spec_order)
    with open(paths["atd_raw"], "w") as handle:
        handle.writelines(raw_lines(players, ball, spec_order))
    write_smart_tagging(paths["smart_tagging"], default_tags())
    write_homography(paths["homography"], n_frames, seed)
    return {"data_dir": str(data_dir), "paths": paths, "n_frames": n_frames,
            "slot_team": slot_team, "players": players, "ball": ball}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir")
    parser.add_argument("--frames", type=int, default=FULL_LENGTH)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    write_bundle(args.data_dir, args.frames, args.seed)
    print("wrote a synthetic bundle of {} frames to {}".format(args.frames, args.data_dir))
