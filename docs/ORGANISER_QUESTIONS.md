# Questions for the organisers (drafts)

Ready to post. README.md routes terms-of-use questions to
hackathon.talentarena@mobileworldcapital.com and everything else to GitHub issues. Record the
answers in COMPLIANCE_LOG.md.

---

## 1. Hosted coding assistant behind a data firewall: what counts as an "extract"? (email; terms of use)

> The FAQ quotes the terms: participants shall not make the videos, datasets "or any extracts
> thereof" available to any third-party hosted LLM. We want to use a hosted LLM coding assistant
> (Claude Code) to write code, and we keep the data away from it:
>
> - the data stays on our machine; the assistant is technically blocked from opening the data
>   folder, our parsed caches and debug renders, and it never sees video frames or plots;
> - every script that reads the data runs locally behind a wrapper that sends the console output
>   to a local log and passes on only approved lines.
>
> Which of these approved outputs may the assistant see?
>
> (a) pass/fail checks, row counts and schema checks (e.g. "199,319 records, one per frame: PASS");
> (b) the output of the official `validate_submission.py` and `run_evaluation.py` for our own
>     submission files, including scores on the public 5-minute window;
> (c) summary statistics of our own predictions: number of possessions, duration quantiles, share
>     of dead ball;
> (d) a few low-dimensional aggregates of the input, never frame-indexed: ball-missing rate,
>     quantiles of ball-gap length, mean number of tracked players.
>
> Today we allow (a)–(c) and withhold (d). Is that within the terms? Is (d) allowed as well?

## 2. Public window: learning the label convention vs tuning (GitHub issue)

> CHALLENGE.md says to use the public ground truth "to understand what is expected" and that "if
> your reading of the definition disagrees with it, the ground truth is right". RULES.md and
> EVALUATION.md say it is not a tuning set. We would like to confirm where the line is.
>
> (a) Looking at the public window to learn a convention, and coding that convention as a general
>     rule. For example: is a change of possession placed at the new team's first touch or at
>     confirmed control? Does `D` start when the ball crosses the line or at the whistle?
> (b) Measuring a quantity in the window, such as the typical delay between the ball crossing the
>     line and the start of `D`, and using that number as a constant.
>
> We read (a) as allowed (it is the "labels meaning what you think" check) and (b) as tuning, so
> not allowed. Is that right?

## 3. Reproduction environment for "the same file" (GitHub issue)

> RULES.md requires running the repository to reproduce the submission exactly, not
> approximately. Which OS and CPU architecture will you re-run on? Is a pinned
> `requirements.txt` with a stated Python version enough, or would you rather have a Dockerfile?
> If a model trained only on other games is part of the pipeline, may we commit its trained
> weights and reproduce by inference only, with the training script included for inspection?
> Some numerical libraries do not give bit-identical results across platforms.

## 4. May the four baseline CSVs be redistributed in our public repository? (GitHub issue)

> LICENSE extends MIT to "the source code, the documentation, and the public 5-minute ground
> truth". The four `evaluation/results/baseline_*.csv` files are not named there, and they are
> derived from the ATD. May a team's public repository include copies of them, as this one does?
> Or should we leave them out? We use them only to check that we reproduce them byte for byte.

## 5. Kickoff frames: games.py anchors vs DATA.md / FAQ, one frame apart (GitHub issue, low priority)

> `anchor_video_frame(GAMES["BAR-ATH"]["kickoff"])` gives video frame 11186, and the
> `kickoff_2nd` anchor gives 113741. DATA.md says the first half is graded from its kickoff at
> 11185, and the FAQ (also `starter/load_smart_tagging.py`) gives the second-half kickoff as
> 113742. On which frame is the ball struck in each half? We want to assert that our algorithmic
> kickoff detection agrees with the published frames. A one-frame offset is free under spellF1,
> so this is a documentation consistency question.

## 6. Do touches without control ever form a spell in the ground truth? (GitHub issue)

> CHALLENGE.md says possession changes only when the other team gains controlled possession: a
> touch without control, a deflection that is not an interception, or an aerial duel won without
> control does not change it. In the corrected tracking the ground truth comes from, can such a
> touch still produce a short spell of the other team, e.g. a blocked pass that rebounds to the
> passer's team? Or does the corrected tracking assign the ball to a team only once a player has
> it under control, so that every spell in the ground truth is a controlled possession? We do not
> want to tune on the public window, so we would rather have the rule stated than infer it.
