# bench-1 results: void as a test of the models; useful as a test of the harness

Run 42974778781 (commit `fbf1f71`), 2026-11-11, 06:46–07:40 UTC. Raw data: `router-runs:bench/bench1-42974778781/`. Pre-registration: `PREREG.md`. Written by Claude Opus; numbers audited by a non-Claude model (below).

## Bottom line
- **0 of 9 agent runs passed.** The no-agent control failed all 3 as required. That shows untouched code doesn't pass; that each hidden test does pass on the real fix was checked separately before registering.
- **The run cannot answer H2** ("can free models fix these bugs") because **my setup leaked the answers**:
  - The full, current checkout (`src/`, which contains every fix) sat next to the agent's workspace (`ws/`), and the task file was handed to the agent from inside `src/`.
  - 5 of the 6 OpenRouter runs opened `src/record/`, which is the fixed code, and most of them worked there instead of in `ws/`.
  - Only one OpenRouter run (T1 with Flint) stayed clean. It edited the right file 11 times and still failed the hidden test.
- **The run does answer the harness, budget and host questions** (below).

## Per run
| Task | Arm | Pass | End | Wall | Actions | Clean? |
|---|---|---|---|---|---|---|
| T1 | A Nemotron Ultra | no | crash* | 607 s | 40 | no (worked in src) |
| T1 | B Flint | no | finished | 223 s | 34 | **yes** |
| T1 | C Gemini 3.8 Flash | no | timeout | 1,500 s | 5 | yes (stalled) |
| T2 | A | no | finished | 132 s | 13 | no |
| T2 | B | no | finished | 262 s | 40 | no (read src) |
| T2 | C | no | timeout | 1,500 s | 5 | yes (stalled) |
| T3 | A | no | finished | 120 s | 17 | no |
| T3 | B | no | crash* | 257 s | 40 | no |
| T3 | C | no | quota 429 | 533 s | 2 | yes (refused) |
| T1–T3 | Z control | no (as required) | – | – | – | – |

\* `TypeError: Object of type set is not JSON serializable` (2 runs), raised inside the OpenHands SDK (v1.28.1, CLI v1.16.0). This is a harness bug, not a model error.

## Pre-registered decisions, applied as written
- **H1 (≥ 7 of 9 runs complete without a harness crash): inconclusive.**
  - Two runs (T1-A and T3-B) crashed inside the SDK.
  - T3-C ended on a Gemini quota error. That is a provider error, which the pre-registration keeps apart from model and harness failures.
  - T1-C and T2-C hit the 25-minute cap while their model calls hung (300 s read timeouts).
  - **Lenient reading** (only SDK crashes count): 7 of 9, so H1 holds.
  - **Strict reading** (a run must end on its own): 5 of 9, so H1 fails.
  - The pre-registration didn't say which reading applies, so the honest answer is inconclusive.
  - Other caveats: the CLI is unmaintained, and it does not confine the agent to its working directory.
- **H2:** 0 of 9 meets the "fails" rule. **It is declared void, not failed,** because the setup leaked the answers and sent edits to the wrong tree. This override of the written rule is stated openly here.
- **Budget:**
  - The median is **37 actions per attempt**: OpenRouter runs used 13, 17, 34, 40, 40 and 40 actions. Each action is roughly one model call, but this is approximate: retries aren't counted.
  - At 1,000 free requests a day, that is about **27 attempts a day**, under the 100-call threshold. An attempt is one worker doing one task once, as in real use.
  - The auditor multiplied by the 3 arms. That's rejected: in real use, arms don't all run on the same task.
  - Several runs stopped at exactly 40 actions. That looks like a default cap, which would mean harder tasks get cut off.
- **H5 (memory):** peak memory was 342–367 MB per run, for the CLI process tree, with the models running remotely. That is far under Oracle's free 12 GB.

## Corrected measures (bugs disclosed)
These three corrections don't change any pass or fail result. They change only these measures:
- **Diff size** was always empty, because the diff was taken against a commit message instead of `HEAD`. The actual edits were recounted from the logged actions.
- **Contamination:** the URL check flagged everything, because the README contains the repo's URL. Recounted from the actions:
  - **No run fetched anything from the web.**
  - The real leak was the filesystem: reading `src/`.
- **Rate limits:** the regex also matched code text. Recounted from the error records:
  - **OpenRouter: none.**
  - **Gemini:** model calls timed out after 300 s in 2 runs (T1, T2), and the third run (T3) hit 429 quota errors.

## What this means for the stack map
- **Gemini's free tier can't sustain an agent loop through this path.** Its calls hung for 300 s each, and it ran out of quota within an hour. It stays useful only for single calls.
- **OpenRouter's free tier carried about 200 calls with no rate limiting.**
- The **OpenHands CLI is fragile.** It crashed in 2 of 9 runs, is unmaintained, and does not confine the agent to its directory. That pushes H1 towards trying Agent Canvas or the SDK directly, or towards a thinner harness of our own. That choice needs a test, not an assumption.

## Bench-2 (a new pre-registration is needed)
- **The agent sees only its workspace.** Copy the grader and hidden tests to a private place, then **delete `src/` before the agent starts**. The task text is passed inline.
- **Fix the three measures:** diff against `HEAD`, contamination from the agent's own actions, and rate limits from error records only.
- **Replace Gemini** with another free OpenRouter model from a different vendor.
- **Record the action cap**, and raise it if the harness allows.

## Outside audit (Nemotron 3 Ultra, 2026-11-11)
It checked every claim against the raw per-run data.
- **Applied:**
  - The median was 37, not about 35.
  - H1's count depended on an unstated reading, so H1 is now inconclusive.
  - "So the grading works" was stronger than the control alone shows.
- **Rejected:**
  - "Budget ×3 arms": in real use, one worker attempts a task once.
  - "3 crashes": T3-C's error was a provider quota error. The pre-registration keeps those apart from harness errors.
- **Already stated in the write-up:** H2 is declared void, overriding the "fails" rule. That override is not pre-registered and is disclosed as such.
