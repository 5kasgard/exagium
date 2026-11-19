# Worker unattended test (queue Q58), fixed before it starts

Written 2026-11-19 before the start. Release test 2's "carries itself without manual dispatch" criterion FAILED
(worker/RELEASE-TEST-2.md; stall at 06:09Z on a ledger fault, then the 24/day chain cap at 07:29Z, with GitHub's
scheduler measured as no usable backup). Its rules allowed at most 2 manual dispatches while the design depended on a
scheduler already known to be unreliable (Q57): the rules and the design did not fit. This test checks the fixes alone.

## What changed (main, this commit)
- `worker.yml`: the chain cap is a backstop (200 a day); the chain still runs only after an item lands, and the
  ledger's daily budget caps are the real limit. A run that stops for budget starts `keeper.yml`.
- `keeper.yml`: sleeps until the given time (handing over to a new keeper past 5h40m), then starts the worker unless
  it is paused or already running. No new secret; the job's own token dispatches.

## The test
- **Start:** one dispatch of `keeper.yml` (a start time about two minutes ahead), which also checks the keeper path end to
  end. No other manual dispatch of `worker.yml` or `keeper.yml` until the end.
- **End:** 48 hours after the start, or earlier when no item is ready.
- **Pass, all of:**
  1. 0 manual dispatches after the start (counted from the API: dispatches by a person, not by github-actions[bot]);
  2. every item that was ready during the window was claimed (no ready item left unclaimed for more than 2 hours
     while budget remained), from the queue's history;
  3. if any run stopped for budget, the worker started again within 30 minutes after 00:05 UTC by a keeper;
  4. no run started while `worker/PAUSE` existed, and never more than one worker run waiting at a time.
- **Fail:** any of these does not hold; the cause is written down before any fix.

Items released in this window are graded for acceptance with release test 2's other items, by its unchanged rules
(blind grading agent, acceptable-outcome column), and labelled as run under this test.

## Result (written 2026-11-19 20:40Z, from the Actions API and the queue record)
**PASS by its rules, on thin evidence.** Started by one keeper dispatch at 10:59Z; four chained runs by github-actions[bot]
(11:01, 11:04, 11:07, 11:11Z) claimed and landed R5–R8; the run at 11:11Z found no ready item, which ends the test early by its rule.
1. manual dispatches after the start: 0. 2. every ready item claimed within minutes. 3. no budget stop occurred, so the keeper's
budget path was not exercised. 4. no run while paused; never more than one waiting.
What it does not show: 11 minutes and 4 items; no budget stop, no keeper hand-over past 5h40m.
**Fault found after the end:** the worker then sat idle for about 8 h with budget left: nothing was ready until P8r was graded,
and only the unreliable cron could restart it (one scheduled run, 19:29Z). Owner, 2026-11-20: workers always run; a stop means
something is wrong. Fix (this commit): a run that finds nothing ready re-polls in 30 minutes through the keeper; keeper runs are
named by their target so the worker counts only its own waiting keepers.
