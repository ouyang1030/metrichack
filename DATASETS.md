# Datasets

Every dataset the pipeline or its validation touches, with source, licence, who labelled it and
how much (SUBMISSION.md asks for this in the technical document). Nothing about BAR-ATH beyond
the official bundle, ever (AGENTS.md section 2.5).

## In use

| dataset | source | licence | labels by | size | used for |
|---|---|---|---|---|---|
| BAR-ATH bundle | organisers, gated | competition terms | automatic (Metrica) | 1 match | input only |

## Candidates for the off-game harness (v1) — not yet verified

Availability, format and licence are unverified until a row moves to the table above.

| dataset | why | to verify |
|---|---|---|
| IDSSE / DFL open data (Bassek et al., *Scientific Data*) | per-frame ball possession and ball status from corrected tracking, closest to how our ground truth was made | licence, possession/status semantics, loader (`kloppy` sportec / `floodlight`) |
| Metrica Sports sample data, games 1–3 | game 3 is in the same FIFA/EPTS format as our ATD | licence; labels must be derived from events — document the rule |
| SkillCorner open data | broadcast tracking with a per-frame possession field, genuinely noisy | licence, field semantics |
| PFF FC World Cup 2022 | tracking + events | registration terms |
