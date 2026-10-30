# Datasets

Every dataset the pipeline or its validation touches, with source, licence, who labelled it and
how much (SUBMISSION.md asks for this in the technical document). Nothing about BAR-ATH beyond
the official bundle, ever (AGENTS.md section 2.5).

## In use

| dataset | source | licence | labels by | size | used for |
|---|---|---|---|---|---|
| BAR-ATH bundle | organisers, gated | competition terms | automatic (Metrica) | 1 match | input only |
| IDSSE / DFL open data (Bassek, Rein, Weber & Memmert 2025, *Scientific Data* 12:195) | figshare, doi:10.6084/m9.figshare.28196177, downloaded 2026-10-27, md5 checked | CC BY 4.0 | DFL / TRACAB: optical tracking with operator quality control; `BallPossession` ("the player in control of the ball") and `BallStatus` per frame — the paper does not say whether these two are set by operators or derived | 7 matches (2 Bundesliga, 5 2. Bundesliga), 14 halves, ~1.0M frames at 25 Hz | off-game harness: labels D where `BallStatus` = 0, else H/A from `BallPossession` (`harness/idsse.py`) |

| Metrica Sports sample data, games 1–3 | github.com/metrica-sports/sample-data (master, last updated 2021-04-15), downloaded 2026-10-30 into `external/metrica/` | no licence file; the README asks for responsible use and that the source be acknowledged in public work (done in docs/TECHNICAL.md) | Metrica: tracking (games 1–2 CSV, game 3 EPTS FIFA as the ATD) and human event data; our per-frame labels are derived from the events by a written rule (`harness/metrica.py`) | 3 anonymised matches, 6 halves, ~0.4M frames at 25 Hz | external check of the exported models (`harness/external.py`): as provided, degraded, stress; never trained on |
| SkillCorner open data | github.com/SkillCorner/opendata, the copy in `../opendata/data/matches` (a download, not a clone), read in place through the symlink `external/skillcorner` | MIT (repository licence); the README asks that SkillCorner be credited (done in docs/TECHNICAL.md) | SkillCorner, automatically: broadcast tracking at 10 fps (players off screen extrapolated), phases of play, per-frame possession group; no human labels. Ours: alive inside a phase of play, the possession group carried over (`harness/skillcorner.py`) | 20 A-League 2024/25 matches, 40 halves, ~3.0M frames at 25 Hz after interpolation | external check under real broadcast noise (`harness/external.py`), never degraded, never trained on |

## Candidates — not yet verified

| dataset | why | to verify |
|---|---|---|
| PFF FC World Cup 2022 | tracking + events, 64 matches | registration terms (needs a team member to register) |
