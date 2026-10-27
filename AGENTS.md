# AGENTS.md — More than a Hack 2027, Phase 1: Ball Possession

> Standing instructions for the coding agent on this project. Read this file in full at the start
> of every session. Where it and the official repository disagree, the official repository wins:
> `RULES.md` for what is allowed, `evaluation/possession_metrics.py` for the metric, `DATA.md` for
> the data. If something here looks wrong, say so — do not silently work around it.

Team name: `<TEAM_NAME>` · Deadline: **2026-11-17** · Internal freeze: **2026-11-14**
Official repo: https://github.com/metrica-sports/more-than-a-hack-2027

---

## 1. Mission

For every frame of the tactical video of FC Barcelona (home, `H`) vs Athletic Club (away, `A`),
output `H`, `A`, `D` (dead ball) or `X` (no answer). One CSV, 199,319 rows, `frame,label`.

- **Ranked on `spellF1`** (possessions matched one-to-one by temporal IoU, F1 averaged over IoU
  0.3–0.7). **Tie-break: `macroF1`** (per-frame mean F1 over H/A/D).
- Floor to beat: `baseline_smooth50`, full-match spellF1 **0.2433**.
- Optimise for spellF1 first, macroF1 second. Never trade a real spellF1 gain for macroF1.

Three deliverables: the CSV (`<TEAM_NAME>.csv`), a **public** code repo that regenerates the
*identical* file, and a short technical document (`<TEAM_NAME>.pdf` or `.md`).

---

## 2. Non-negotiables (any breach can disqualify us)

1. **Data firewall — no provided data or extracts reach a cloud LLM.** The terms forbid making the
   videos, datasets "or any extracts thereof" available to any third-party hosted LLM. *You are a
   hosted LLM.* So you write and run code, but you never see the match data. Section 4 is the
   protocol. When in doubt, don't print it.
2. **No hand labels or hand corrections on this match**, in any amount, for any purpose (training,
   validation, post-editing, "just the transitions"). Every label comes out of an algorithm.
3. **No constants tied to specific frames of this match.** Thresholds must be generic and physical
   (metres, seconds, m/s, counts of players) and justified by football logic or off-game
   validation. The only frame numbers allowed in code are those published by the organisers
   (`games.half_video_blocks`, the 199,319 length) — and prefer detecting events algorithmically.
4. **The public 5-minute ground truth is not a training set and not a tuning set.** It may be used
   to check the pipeline is correct (alignment, file format, label semantics). Never use it to pick
   a hyperparameter, a threshold, a model, or between two variants. See section 5.
5. **No human-produced data about this match from any source** — no event data (Opta, StatsBomb,
   Wyscout, LaLiga…), no corrected tracking, no possession stats, no match reports used as data. Do
   not web-search for this match's events. Human-labelled data about *other* games is fine.
6. **No cloud APIs inside the pipeline that receive match data** (no hosted LLM, no hosted vision
   API on video frames). Local models only.
7. **Byte-identical reproducibility.** One command on a fresh environment with the data bundle must
   regenerate the exact submitted file (verify with sha256 across two clean runs).
8. **Never commit data or anything that reconstructs it**: `data/`, video, parsed caches,
   frame-level exports, debug renders, or BAR-ATH submission CSVs. Our repo will be public.
9. **Validate every file** with `python evaluation/validate_submission.py <file>` before it is
   called a candidate. A malformed file scores zero.

If an idea sits near any of these lines, stop and draft a question for the organisers (GitHub
issue text) for the team to post. Do not build on it until it is answered.

---

## 3. Facts you must internalise

### The task and the frames
- 199,319 frames, 25 fps. Row *i* = frame *i* of the **tactical** video. ATD `frame_id` is 1-based:
  `video_frame = frame_id - 1`. Smart Tagging: `video_frame = round(seconds * 25)`. Homography
  `frame` is already the video frame.
- Graded frames: **1st half 11,185–90,651**, **2nd half 110,808–190,197** (158,857 frames). All
  other frames are never scored.
- Trap: the 2nd half is graded from 110,808 but kicks off at ~113,742 — those frames are all `D`.
  The 1st half is graded from its kickoff. Detect kickoffs algorithmically (Smart Tagging
  `Kick Off (Start)` + players in own halves + stationary ball at the centre spot) and assert the
  result is consistent with the published boundaries.
- If you ever produce ~164k rows you indexed the broadcast video. The broadcast feed is **not
  synchronised** and has no mapping table; treat it as low priority.

### What the ground truth is
- Derived **mechanically from Metrica's corrected tracking**: `D` where the corrected tracking marks
  the ball dead; otherwise the team the corrected tracking names as owning the ball. No smoothing,
  no hand edits afterwards. Our job is to reproduce *that* notion of "dead" and "owning" from the
  *uncorrected* data.
- Possession changes only when an opponent gains **controlled** possession. A touch without control,
  a deflection that does not intercept, an aerial duel won without subsequent control — none of
  these change possession. Loose/contested ball → possession stays with the last team.
- A possession is a maximal run of one team with the ball alive. `H H D D H H` is **two**
  possessions. Dead ball is a stoppage, never a possession.

### What the data is like
- ATD: 46 track slots for ~22 players, no identities, slots fragment and swap; slot → team comes
  from the metadata (first team in the XML = home = Barcelona). Coordinates normalised 0–1; convert
  with pitch 105.3 × 68.0 m before any distance. Per object: x, y, speed (m/s). NaN = missing.
- **The ball is missing on 41% of graded frames**, disproportionately in crowded boxes, aerial
  duels and tackles — exactly where possession changes. Design for this from day one.
- Smart Tagging: 638 automatic (not human-reviewed) phase intervals with `Team` (36 are `N/A`,
  including **every throw-in**), set-piece `Type`, `Half`. Intervals are phases, not possessions.
  Strong prior, noisy, never an answer key. Tag columns are Title Case; match case-insensitively.
- Homography: valid on 100% of tactical frames; only needed if we touch the video.

---

## 4. Data firewall protocol (how to work without seeing the data)

Everything you need about formats is already documented in `DATA.md`, `starter/*.py` and
`games.py`. Work from those. Concretely:

**Never**
- Read, `cat`, `head`, `tail`, `less`, `grep` or open any file under `data/`, `cache/`, `debug/`, or
  any `.mp4`. Never view an image or plot rendered from BAR-ATH data. Never extract or look at video
  frames.
- Print raw rows, arrays, per-frame or per-event values, timestamps, coordinates, tag contents or
  homographies from BAR-ATH. Never print a slice of a DataFrame built from it (`df.head()`,
  `print(arr[:10])`, `describe()` with per-column detail).
- Dump BAR-ATH per-frame predictions to stdout.

**Allowed outputs from scripts that touch BAR-ATH data** (what you may see)
- Pass/fail checks, assertion messages, row counts, schema checks against the documented schema,
  runtimes, memory use.
- Official metric outputs (`run_evaluation.py`) and our own summary diagnostics of **our
  predictions**: number of possessions, duration quantiles, flicker rate, share of `D`, agreement
  rate with Smart Tagging team tags.
- `<MODE: AGGREGATES>` Low-dimensional, non-frame-indexed aggregates of the input data (e.g.
  ball-missing rate, gap-length quantiles) — at most ~20 numbers per report, nothing that could
  reconstruct the data.
  `<MODE: STRICT>` If the team sets STRICT, input aggregates are not shown to you either: write
  them to a gitignored JSON that downstream code consumes, and you only see "written OK".
  **Current mode: STRICT** (the team switches to AGGREGATES only after the organisers confirm it).

**Mechanics**
- Every entry point that touches BAR-ATH runs through one wrapper (e.g. `mtah/safe_run.py`) that:
  catches exceptions and prints only exception type, file and line, and a message with numbers
  masked; and refuses to print anything not produced by an approved report function.
- Parsed data is cached as `.npz`/`.parquet` under `cache/` (gitignored, never read by you).
- Visual debugging is for humans: scripts write overlays/plots to `debug/` (gitignored). Ask the
  team to look, and ask them to describe failure modes in generic terms ("restarts after throw-ins
  are detected ~1 s late"), never as frame-specific facts.
- If you need a fact about the data that the docs do not give you, ask the team; they decide what
  can be shared.
- External datasets about other games (section 5) are fine to inspect normally, subject to their
  licences.

---

## 5. Validation strategy — the core of doing this honestly and well

The public window has only 17 possessions; one missed possession moves spellF1 by ~6 points, and
the published baselines show it ranks variants in the wrong order. On the full match, spellF1
differences below **~0.04 are noise**. So we need a validation set that is (a) big, (b) not this
match, (c) noisy like the ATD.

### 5.1 Off-game harness (primary decision tool)
Build `harness/` that converts other matches into the exact BAR-ATH format (199k-style per-frame
arrays, 25 fps, normalised coords, slot-based tracklets, H/A/D labels) and scores predictions with
the **official** `possession_metrics.evaluate()` — import it, don't reimplement it.

Candidate sources (verify availability, format and licence yourself; record each in `DATASETS.md`):
- **IDSSE / DFL open data** (Bassek et al., *Scientific Data*): 7 Bundesliga matches, positional data
  with per-frame **ball possession and ball status** flags — the closest analogue to how our ground
  truth was derived (from corrected tracking). Loadable via `floodlight` or `kloppy` (sportec).
  Top priority.
- **Metrica Sports sample data** (games 1–3): game 3 is in the same FIFA/EPTS format as our ATD;
  labels must be derived from events (document the derivation rule).
- **SkillCorner open data**: broadcast tracking with a per-frame possession field; genuinely noisy.
- **PFF FC World Cup 2022**: tracking + events, registration required.

Label semantics differ across providers (what "possession" and "dead" mean). Use these sets to
compare *methods* under the same metric, not to import their absolute possession counts as truth.

### 5.2 ATD noise simulator
Clean tracking is too easy. Write `harness/degrade.py` that turns clean tracking into ATD-like
tracking: ball dropout (~41%, correlated with ball speed, crowding near the ball and, if available,
ball height), tracklet fragmentation and identity swaps across 46 slots, missing and duplicated
players, positional jitter. Calibrate to ATD aggregates via the firewall rules (STRICT: the
calibration script writes parameters to a gitignored JSON that the simulator reads). Report every
result on clean **and** degraded versions.

### 5.3 Unsupervised diagnostics on BAR-ATH (no labels involved)
Possession count and duration distribution of our output vs priors from the off-game sets,
flicker rate, agreement with Smart Tagging team tags, stability under perturbation (drop 10% more
ball frames — does the output change much?). These catch bugs; they do not pick winners.

### 5.4 The public 5-minute window — strictly limited use
- **Phase 0–1, semantics check (log each use in `COMPLIANCE_LOG.md`):** confirm alignment and
  label conventions — e.g. is a possession switch placed at the new team's *first touch* or at
  confirmed control? Where does `D` start relative to the ball crossing the line? This is
  explicitly what the organisers offer the window for.
- **After that, smoke test only:** after each major version, run `run_evaluation.py` once as an
  alarm. A sharp drop means "look for a bug" (alignment, label swap, broken half boundary). It
  never decides between variants, and you never iterate "look at errors → adjust → look again".
- Never train on it, never compute thresholds from it, never use its possession count or mean
  duration as a target.

### 5.5 Decision rule
A change is accepted when it improves **mean spellF1 on the degraded off-game harness** beyond its
bootstrap interval, without hurting macroF1 materially, and passes the BAR-ATH diagnostics. If two
variants are within noise, keep the simpler one.

---

## 6. What the metric rewards — design implications

- **Every flicker is fatal.** A spurious 1-frame `A` inside an `H` possession turns one possession
  into three. A spurious short `D` does the same. Output must be piecewise constant with realistic
  segment lengths.
- **Merges are fatal too.** Missing a short stoppage (throw-in, foul) between two possessions of the
  same team merges them into one long predicted spell that matches neither at IoU ≥ 0.3–0.7.
  Dead-ball detection matters as much as team assignment.
- **Timing matters at the frame scale:** perfect-but-3-frames-late ≈ 0.98, 8 frames ≈ 0.89,
  1 s ≈ 0.65. Confirm a switch with delay if you must, but **backdate the boundary** to the onset
  (first control / ball crossing the line / restart touch).
- **`X` is (weakly) dominated on graded frames.** spellF1 forward-fills `X` with our last label
  (back-fills at the start of a half); macroF1 counts `X` as a miss. Writing that same filled label
  explicitly gives identical spellF1 and equal-or-better macroF1. So: **never output `X` inside the
  graded halves**; write `X` only outside them (or write `D` there — either is free). If uncertain at
  a transition, guess; if uncertain mid-possession, keep the current label.

---

## 7. Technical roadmap

Treat the problem as **segmentation**, not frame classification:
(a) alive/dead segmentation, (b) team possession within alive segments with sparse, controlled
switches, (c) a structured decoder that enforces realistic durations, (d) boundary refinement.

### v0 — reproduce and scaffold (day 1)
Reproduce the four baselines exactly; wire the scorer, validator, sha256 check and the firewall
wrapper; build the parsed-data cache; make the full pipeline one command.

### v1 — physics/rules + structured decoding (week 1)
- **Ball state (dead/alive) evidence:** ball last seen beyond a touchline/goal line (tolerance
  ~0.5 m) then lost or stationary; collective deceleration of all players; ball stationary > ~1 s
  at a restart location (corner arc, goal area, centre spot, touchline, free-kick spot with a wall
  ~9 m away); kickoff formations; goals → dead until the next kickoff. Smart Tagging `SET PIECES`
  starts are restart evidence; `GOALS` imply a following stoppage.
- **Restart owner:** team of the player taking the restart (nearest to the stationary ball; for
  throw-ins, the player at/outside the touchline — Smart Tagging gives no team there).
- **Possession evidence while alive:** a player of team T within ~1–1.5 m of the ball for several
  frames with ball velocity ≈ player velocity (dribble) or ball decelerating into the player
  (reception) → control. Ball in flight → stays with the passer's team. Single touches /
  deflections → no switch. Treat these as *starting values* to be set on the off-game harness.
- **Ball gaps:** interpolate short gaps physically; across long gaps use team-level context
  (which team stretches its shape and moves toward the opposing goal, which team compacts and
  presses toward the last ball position), plus the continuity prior.
- **Smart Tagging as soft evidence:** attacking codes (`BUILD UP`, `PROGRESSION`,
  `CREATING CHANCES`, `BALL IN FINAL THIRD`, `BALL IN THE BOX`, `PLAYERS IN THE BOX`,
  `LONG BALL`, `ATTACKING TRANSITION`) → evidence for `Team`; defending codes
  (`DEFENDING IN … THIRD`, `DEFENSIVE TRANSITION`) → evidence for the *other* team. Verify these
  directions with unsupervised agreement against tracking-only estimates before trusting them.
  Down-weight interval edges; `N/A` carries no team evidence.
- **Decoder:** HMM/HSMM (or CRF) over {H, A, D} with explicit duration models and switch costs;
  Viterbi for the final path. Then boundary refinement: snap each switch to the most likely onset
  frame within a small window using local physics.

### v2 — learned emissions (week 2)
- Frame-level classifier on permutation-invariant features (per-team nearest distances to the
  ball, counts within radii, team centroids/spreads/velocities relative to the ball and goals,
  ball speed/acceleration, missingness indicators and time since last ball sighting). Start with
  gradient boosting; consider a temporal model (TCN/GRU) or a set/graph encoder over players if
  time allows.
- Train on the degraded off-game harness; feed calibrated probabilities to the same decoder.
  Smart Tagging exists only for BAR-ATH, so fuse it at decoding time with a principled, fixed weight
  (document how it was set; not from the public window).
- Optional: self-training on BAR-ATH with algorithmic pseudo-labels (allowed — no human decides
  any of them).

### v3 — only if time and GPU allow
Own ball detection on the tactical video (local models only) to fill ball gaps, projected to pitch
with the homography. Time-box it; it is the most expensive and least certain line.

At every version: ablate (with/without Smart Tagging, with/without decoder, clean vs degraded),
and write the result in `EXPERIMENTS.md`.

---

## 8. Repository & engineering standards

```
<our-public-repo>/
  README.md                 install, data placement, the one command, expected sha256
  AGENTS.md                 this file
  requirements.txt          exact pins (==); Python version stated
  games.py, evaluation/     copied unchanged from the official repo (MIT, keep the licence)
  mtah/                     our package: io, features, ball_state, possession, decoder, fuse
  mtah/safe_run.py          firewall wrapper (section 4)
  harness/                  off-game loaders, degrade.py, scoring
  scripts/make_submission.py   → evaluation/results/<TEAM_NAME>.csv
  models/                   small trained weights from OFF-GAME data only (with training script)
  docs/TECHNICAL.md         the technical document
  EXPERIMENTS.md, COMPLIANCE_LOG.md, DATASETS.md
  data/  cache/  debug/     gitignored
```
- `.gitignore` must cover `data/`, `cache/`, `debug/`, `*.mp4`, `evaluation/results/*.csv`
  (except the official baselines), `*.npz`, `*.parquet`. Add a pre-commit check that blocks files
  > 5 MB and any path under those folders.
- Determinism: fixed seeds everywhere, deterministic algorithms, CPU for the final inference if GPU
  nondeterminism appears, no time- or order-dependent behaviour. `make_submission` twice in a fresh
  venv → same sha256.
- Data location via `MTAH_DATA_DIR` / `games.asset_paths()`; never hard-code paths.
- The pure-Python ATD parse is slow — parse once into the cache, vectorise everything after.
- Small, typed, tested functions. Unit tests on synthetic sequences for the decoder and the
  possession rules (e.g. "a deflection does not switch", "H H D D H H is two spells").

---

## 9. Working protocol

- Plan before coding each version; keep a task list; one experiment = one hypothesis.
- `EXPERIMENTS.md`: date, hypothesis, change, off-game spellF1/macroF1 (clean and degraded) with
  intervals, BAR-ATH diagnostics, decision.
- `COMPLIANCE_LOG.md`: every look at the public window (why, what was checked, what was not
  changed because of it), every external dataset, every rule question.
- Stop and ask the team when: a rule is unclear; you need a fact only the data would give you; a
  decision is expensive to reverse; results look too good (suspect leakage or misalignment).
- Report progress tersely: what changed, the numbers, the next step.

## 10. Timeline

| when | milestone |
|---|---|
| Oct 27–28 | v0: scaffold, firewall wrapper, baselines reproduced, cache, one-command pipeline |
| Oct 29–Nov 2 | off-game harness + degradation simulator; v1 rules + HSMM decoder |
| Nov 3–9 | v2 learned emissions, Smart Tagging fusion, ablations |
| Nov 10–13 | robustness, boundary refinement, technical document draft |
| **Nov 14** | **freeze**: final file, two clean-room reproductions, sha256 recorded |
| Nov 15–16 | document final, repo public and checked for data, submit (one day of buffer) |

## 11. Final checklist

- [ ] `<TEAM_NAME>.csv`: 199,319 rows, frames 0…199318 in order, labels in {H, A, D, X}, no `X`
      inside graded halves, passes `validate_submission.py`
- [ ] Fresh clone + fresh venv + data bundle + one command → identical sha256
- [ ] Repo public; no data, caches, debug renders or BAR-ATH predictions in it or in its history
- [ ] Technical document covers: approach and why; assets used and dropped; every training dataset
      (source, who labelled it, how much — none from this match); possession rules implemented
      (loose balls, challenges, restarts); limitations; reproduction
- [ ] `COMPLIANCE_LOG.md` complete; public window never used for selection
- [ ] Off-game harness results reported with intervals

---

## Appendix — Claude Code guardrail (`.claude/settings.json`)

Keep the data in `./data` so these relative rules apply. They block the file tools and commands
that name these paths, but not every shell route (e.g. a recursive grep or a Python one-liner) —
the protocol in section 4 still applies.

```json
{
  "permissions": {
    "deny": [
      "Read(./data/**)",
      "Read(./cache/**)",
      "Read(./debug/**)",
      "Read(./**/*.mp4)"
    ]
  }
}
```
Do not enable the sandbox's filesystem read-deny for `data/`: the pipeline itself must read it.
