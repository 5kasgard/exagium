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
