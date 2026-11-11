# Pre-registration: bench-1, OpenHands with free models on exagium's past fixes

Written 2026-11-11 before any run. Author: Claude Opus (it chose the tasks, wrote the statements and the grader, and is not an arm). Changes after the first run need a new pre-registration.

## Questions
- **H1.** Is OpenHands, run headless, a workable base for the machine and harness?
- **H2.** Can free models drive it to fix real exagium bugs?
- **Budget.** How many model calls does a task take? That number decides whether the free allowance (1,000 OpenRouter requests a day) carries agent work.
- **H5.** What memory does a run need, and would it fit Oracle's free 12 GB?

## Tasks (`tasks.json`, statements in `tasks/`)
Each is a past fix in this repo that added its own test:

| Task | Fix commit | What it adds |
|---|---|---|
| T1 | `4711b86` | `check-anchors` reports a missing `.ots` proof |
| T2 | `06bc550` | `_orphan_entries_problem` |
| T3 | `bdd7ec5` | `check-anchors --max-pending-hours` |

How each task is set up and graded:
- The agent gets the tree at the commit **before** the fix, as a fresh git repo with no history, plus the task statement.
- **Validity, checked before registering:** each hidden test fails on the base and passes on the fix.
- **Grading, by script only (`grade.py`):** the fix commit's test file is copied over the agent's own. The task passes if the hidden test passes **and** no test in `p2p-<fix>.json` fails. Those are the tests that passed both before and after the real fix.

## Arms (each provider's default settings, one run per task, 9 runs)
| Arm | Model | Provider |
|---|---|---|
| A | `nvidia/nemotron-3-ultra-550b-a55b:free` | OpenRouter |
| B | `recursive/flint-v1.0:free` | OpenRouter, built for coding agents |
| C | `gemini-3.8-flash` | Google AI Studio free tier |

- The OpenHands CLI is the latest at run time; its version is recorded.
- Each run is capped at 25 minutes of wall time.
- A paid OpenRouter model makes the job fail before it runs.

## Measures, per run
- Pass or fail (hidden test and pass-to-pass).
- Steps and model calls, counted from OpenHands' JSON events.
- Wall time and peak memory (`/usr/bin/time -v`).
- Provider errors (429s and the like), recorded apart from model failures.
- Diff size.
- **Contamination flag:** if the event log mentions the fix commit's hash or this repo's GitHub URL, the run is reported separately. The repo is public, so an agent could look up the answer.

## Decisions, fixed now
- **H1 holds** if the harness completes at least 7 of 9 runs without a harness crash, whatever the model does.
- **H2 holds** if at least one arm passes 2 of 3 tasks without contamination. **Fails** if no arm passes any.
- **Budget:** report the median calls per task. If the median exceeds 100 calls, free OpenRouter supports at most about 10 tasks a day. That goes to the owner as a cost decision.
- With 9 runs, every result is **a direction, not evidence.** No ranking of models is claimed.

## Spend
$0. All arms are free. The expected budget use is under 900 OpenRouter requests (two arms × 3 tasks × at most 150 calls) and under the Gemini free daily limit (unknown; a quota 429 is reported as "unavailable", never as a model failure).

## Outside review before running (Nemotron 3 Ultra, 2026-11-11) and changes made
1. **"Fixes are in the models' training data."** Rebutted with dates. The fixes were committed 5–11 Nov 2026. Nemotron 3 Ultra was released 4 June 2026 and Flint on 21 Oct 2026, so neither could have seen them. Gemini 3.8 Flash's release date is unknown and is recorded as a caveat. Web lookup is still possible, so the contamination flag stays.
2. **The H2 rule didn't cover every result.** It is now:
   - **holds** if any arm passes 2 or more of 3, uncontaminated;
   - **fails** if 0 of 9 pass;
   - **inconclusive** otherwise.
3. **"Runners have 7 GB."** GitHub's standard public-repo Linux runners have 16 GB, so peak memory is measured below the cap. The cap is recorded with the result.
4. **Grader not pinned.** Every result now records the repo commit, which pins `grade.py`, `tasks.json` and the `p2p` lists.
5. **One run each.** Accepted: the result is a direction, as stated above.
6. **Correlated 429s.** At most 2 jobs run in parallel. 429s are counted per run and reported beside wall time and calls.
7. **OpenHands version.** Pinned to openhands-cli commit `283c6ac` (v1.16.0). Note: the CLI's README has marked it "no longer actively maintained" since Aug 2026, in favour of Agent Canvas. That is a finding for H1.
8. **No control.** Added arm **Z**: no agent, grade the untouched base. It must fail all 3 tasks, or the grading is broken and the run is void.
