# bench-2 results: the seal held; free models fixed real exagium bugs; harness verdict inconclusive on wording

Run 42983345686 (commit `9cccb53`), 2026-11-11, 07:58–09:10 UTC. Raw data: `router-runs:bench/bench2-42983345686/`. Pre-registration: `PREREG.md`. Written by Claude Opus. The numbers were audited by gpt-oss-120b, from a family other than every arm (below).

## Bottom line
- **The seal held.** No attempt touched a path outside its own workspace.
- **H2 holds:**
  - **Nemotron 3 Ultra passed 3 of 3, cleanly.**
  - North Mini Code passed 3 of 3 by the grader, 2 of them clean. On T1 it fetched two GitHub pages from the sigstore-js repository, not exagium, and the pre-registration counts any fetch as unclean.
  - Flint passed 0 of 3; the harness judged it stuck each time.
- **H1′ is inconclusive under the literal rule:** 6 of 9 attempts, 67%. The three iteration-cap endings logged SDK error events, which the rule counts as crashes. Under the rule's intent (catching real crashes, like bench-1's `TypeError`), the result is 9 of 9. Both readings are given below.
- **Budget:** the median is 47 actions per attempt, about 21 attempts a day on free OpenRouter. The spend was $0.

## Per attempt
| Task | Arm | Pass | Clean | End | Actions | Wall | Real edit |
|---|---|---|---|---|---|---|---|
| T1 | A Nemotron 3 Ultra | **yes** | yes | finished | 64 | 689 s | record.py +10 |
| T1 | B Flint | no | yes | stuck | 35 | 260 s | none |
| T1 | D North Mini Code | yes (grader) | **no: 2 fetches** | iteration cap 80 | 78 | 326 s | record.py +10, plus a test fixture |
| T2 | A | **yes** | yes | finished | 33 | 355 s | record.py +80 −1 |
| T2 | B | no | yes | stuck | 14 | 76 s | none |
| T2 | D | **yes** | yes | iteration cap 80 | 78 | 258 s | record.py +30 |
| T3 | A | **yes** | yes | finished | 47 | 789 s | record.py +30 |
| T3 | B | no | yes | stuck | 17 | 132 s | none |
| T3 | D | **yes** | yes | iteration cap 80 | 78 | 280 s | record.py +26 −1 |
| T1–T3 | Z control | no (as required) | – | – | – | – | – |

No pass broke a pass-to-pass test. There were no rate-limit errors.

## Pre-registered decisions, applied as written
- **Seal: holds.** `outside_paths` is empty for all 9 attempts, so H2 is not void.
- **H1′: inconclusive under the literal rule.** The data:
  - No attempt was cut short by a provider, so all 9 count.
  - No exception occurred in `run_agent.py` outside the agent loop.
  - Flint's three "stuck" endings logged **no** SDK error event; they surfaced only as `ConversationRunError` from the run loop.
  - North Mini Code's three runs each logged a `ConversationErrorEvent` with code `MaxIterationsReached`, meaning they hit the pre-registered iteration cap of 80.
  - **Literal reading:** the rule counts "an SDK error event" as a harness crash. Then 6 of 9 attempts are crash-free (67%), which falls between 44% and 78%, so H1′ is **inconclusive**.
  - **Intended reading:** the rule was written to catch harness faults like bench-1's `TypeError`. Hitting a cap that the pre-registration set itself isn't such a fault. Then it's 9 of 9, and H1′ holds.
  - The literal reading is the headline. The gap is a defect in my pre-registration's wording. The next pre-registration will name which SDK error codes count as crashes.
- **H2: holds.** At least one arm passed 2 or more of 3 cleanly: A with 3 of 3, and D with 2 of 3.
- **Budget:**
  - Actions per attempt were 14, 17, 33, 35, 47, 64, 78, 78 and 78, so the median is **47**.
  - At about one model call per action, that is roughly **21 attempts a day** from 1,000 free requests, under the 100-call threshold.
  - Prompt tokens per attempt were 0.28M–2.24M, mostly cache reads. That costs nothing on free models but would matter on paid ones.

## Measurement corrections (disclosed; pass/fail unaffected)
- **Diff:** `--shortstat` counted file-mode changes made by the workflow's own `chmod` (110 files), plus the Agent Server's conversation state written into `/workspace/conversations/`. The real edits above were recounted from `agent.diff`, counting only content hunks outside `conversations/`.
- **Model calls:** `llm_calls` read 0 because the SDK's `token_usages` list was empty. Actions and token totals are used instead.
- **Control labels:** `measure.py` labelled the three Z controls `harness_crash` because a control writes no `run.json` by design. That label is wrong; the controls ran as intended and failed all 3 tasks, as required.

## What this says, as directions rather than evidence (9 attempts)
- **The harness itself, not only the model, explains much of bench-1's 0 of 9.** Nemotron failed every bench-1 attempt under the old CLI with a 40-action cap and an unsealed setup. Here it passed every task. Its passing runs used 33–64 actions, so a cap of 40 would have cut off at least one of them.
- **The models differ in kind, not just in degree:**
  - Nemotron finishes on its own.
  - North Mini Code solves the task but doesn't stop. All three runs hit the cap after the fix was already in place, which is wasted calls.
  - Flint stalls early.
  - That is a routing fact for the worker spec cards: the same task costs different amounts depending on the worker.
- **Web access matters:** one agent went looking on the web unprompted. A seal on the filesystem doesn't stop that. Only the clean-run rule caught it.

## Next (inbox, not decided)
- Fix the three measurement bugs in `measure.py` and `bench2.yml`:
  - set `core.fileMode=false` and exclude `conversations/` from the diff;
  - take model calls from a source that works;
  - give the control its own label.
- Repeat runs, so there are SEs and arms can be ranked, plus harder tasks: the counter and jail fixes.
- Write the first **worker spec cards** from this data: pass rate, actions, ending behaviour and web use, per model.
- **Live trial (H3):** Agent Canvas, opened from the owner's phone.

## Outside audit (gpt-oss-120b on Cloudflare, 2026-11-11; outside every arm's family)
- **It flagged the H1′ classification.** Checking the event logs showed part of its claim was right and part wrong:
  - **Right:** the three iteration-cap endings did log SDK error events, so the literal rule counts them. H1′ was changed from "holds" to "inconclusive".
  - **Wrong:** Flint's "stuck" endings logged no SDK error event. The auditor counted those, which isn't supported by the logs.
- It raised nothing else. Every other number and claim was accepted.
