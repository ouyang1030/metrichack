# Datasets

Every dataset the pipeline or its validation touches, with source, licence, who labelled it and
how much (SUBMISSION.md asks for this in the technical document). Nothing about BAR-ATH beyond
the official bundle, ever (AGENTS.md section 2.5).

## In use

| dataset | source | licence | labels by | size | used for |
|---|---|---|---|---|---|
| BAR-ATH bundle | organisers, gated | competition terms | automatic (Metrica) | 1 match | input only |
| IDSSE / DFL open data (Bassek, Rein, Weber & Memmert 2025, *Scientific Data* 12:195) | figshare, doi:10.6084/m9.figshare.28196177, downloaded 2026-10-27, md5 checked | CC BY 4.0 | DFL / TRACAB: optical tracking with operator quality control; `BallPossession` ("the player in control of the ball") and `BallStatus` per frame — the paper does not say whether these two are set by operators or derived | 7 matches (2 Bundesliga, 5 2. Bundesliga), 14 halves, ~1.0M frames at 25 Hz | off-game harness: labels D where `BallStatus` = 0, else H/A from `BallPossession` (`harness/idsse.py`) |

## Candidates for the off-game harness (v1) — not yet verified

Availability, format and licence are unverified until a row moves to the table above.

| dataset | why | to verify |
|---|---|---|
| Metrica Sports sample data, games 1–3 | game 3 is in the same FIFA/EPTS format as our ATD | licence; labels must be derived from events — document the rule |
| SkillCorner open data | broadcast tracking with a per-frame possession field, genuinely noisy | licence, field semantics |
| PFF FC World Cup 2022 | tracking + events | registration terms |
