# Working on exagium (for any agent or person)

## Before claiming anything
- Every number points to its evidence: a file, a run ID, or a ledger entry. If there is none, the claim is a hypothesis and is labelled as one.
- Say on which task, host and data split a result holds. Results on one task are not results about AI in general.
- A separate reviewer checks consequential work: another agent, or a free model through `review.yml`. Its points are verified, then applied or rejected with evidence.

## Limits, tests and new paths must fit the measured system (2026-11-19)
- **Every limit cites its source** (a published figure, a measurement, a run ID) next to its value: caps, timeouts,
  rates, retries. A limit without a source is a guess and is labelled as one. On 2026-11-19 an unsourced cap of 40 NVIDIA
  calls a day (the published figure is about 40 a minute) and a chain cap of 24 runs a day stopped the worker and the
  proposer while most of the real allowance was unused.
- **A test's rules must fit the system it tests.** No test may depend on a part already measured as unreliable
  (GitHub's scheduler here: 3 hourly runs in about 30 hours, daily runs 7-9 hours late); if the rules can only be met by
  breaking them, the test is redesigned and the failure recorded, not worked around.
- **A new or changed workflow path runs once with no cost before it runs for real** (a dry run, a fixture, a probe).
  On 2026-11-19 a one-character shell bug lost the proposer's first scoring run.

## Context is scarce: process in code, return what the decision needs
Run API calls, tool results and data through a script and return only the summary a decision needs; never paste raw
JSON, logs or whole files into a model's context. The same holds for the free models: give them queried rows from a
store, not whole documents (queue P7f). Evidence: code execution in place of direct tool calls cut one workflow from
150,000 tokens to 2,000 (Anthropic engineering, "Code execution with MCP", Nov 2025) and total tokens by 77% at the same
success rate in an independent benchmark (AIMultiple); direct calls still win for a single simple call.

## Check the world's evidence, not only our own (hypothesis under test, queue P7)
Before building or testing something, look for what is already known: published results, the current best number to beat, prior art and patents, known failure modes. Our own data shows only what we ran. Each outside claim cites its source, and its quote and numbers are checked against that source. Whether this becomes a required step depends on P7's test: would such a check have changed a past decision?

## Use the cheapest worker that does the job well
See the README's "Who does what". Measurement and scoring run on CI. Reviews and proposals go to free models. Scarce paid models do design, integration and final checks.

## Closing every session
1. **Update the public face**, honestly:
   - the README's "Where it is now", "Progress", "Best result so far" and "Next";
   - any part's README whose behaviour changed.

   Remove claims the evidence no longer supports. Say what failed or was dropped, not only what worked. Keep the direction clear: what comes next and why.
2. **Update the queue** (`audit/2026-11-17/queue.json` on the pilot branch, until it moves to `main`): done items with where, new items with a next step.
   - Every new item says what it writes (`touches`: files, paths, branches).
   - Every new item says where it came from (`origin`: `follow-up` if finishing or checking work produced it, `new` if it is a new direction). The weekly indicators use it to tell healthy growth (depth) from breadth outrunning what gets built.
   - `render.py` then lists any other item that writes the same place with no dependency between them. Put such items in one batch, or add a blocker, so related work lands together and nothing is edited twice.
3. **If the session added items with `origin: new`** (new directions, not follow-ups), send them to a free model through `review.yml` with every sign of whose idea each was removed, asking keep, defer, merge or cut for each. Record its calls and what was applied or rejected, with reasons, in the handoff. Why only these: on 2026-11-19, reviews of ideas and scope changed decisions (8 of 11 points of the whole-system review accepted at least in part; the blind review cut or deferred 3 of 16 new items), while reviews of code changed little that tests had not already caught (3 of 15 points held: 3 of 10, then 0 of 5). Revisit with more data.
4. **Write a handoff:** what was done, decided, tested or still hypothesis; the open list in order; anything left running; and spend.
