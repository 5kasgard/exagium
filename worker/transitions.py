"""The queue item's state machine, explicit (queue Q59/Q60; owner 2026-11-20: a pass and a fail go down different pipelines).

One table, one function. Every writer of an item's status (the worker's release, lease expiry, the scheduled Claude
session, the owner's chat) goes through next_status(); a pair missing from the table is refused, never guessed.

Pipelines, from the 2026-11-19 trace (10 items sat in "waiting on the owner"; only 2 needed the owner):
  pass      merged ............................................ -> done
  check     unchanged (no edit proposed, nothing refused) ..... -> review     (a stronger tier confirms no change was right)
            pr / pr_not_created (a person or Claude must read) -> review
  fail      guard_blocked (a script guard or the reviewer refused every edit) -> escalated
            no_claims (research found nothing that survived both checks)     -> escalated
            failed (a step crashed) ............................ -> open, and escalated after MAX_FAILURES in a row
            lease_expired (the run was killed) ................. -> open, counted as a failure
  wait      budget, retry (target changed under the run) ........ -> open (not a failure)
Escalation goes one tier at a time and stops where a check passes: free worker -> scheduled Claude session -> owner.
Research basis: cascades escalate only when a check on the answer fails (FrugalGPT, AutoMix); another model's opinion is a
weak gate, so the gates here are scripts first (guard words, target lines, quote match), and a human is the last tier.

python3 -I worker/transitions.py next STATUS OUTCOME [FAILURES]   -> prints the next status, exit 2 if not allowed
"""
import sys

STATES = ("open", "in progress", "review", "escalated", "done", "waiting on the owner", "parked", "blocked", "scheduled")
MAX_FAILURES = 2   # same as queue_ops before this module (review M4): a second failure in a row climbs a tier

# (from, outcome) -> to. Outcomes by actor:
#   worker run:      merged, unchanged, pr, pr_not_created, guard_blocked, no_claims, failed, budget, lease_expired
#   claim:           claim
#   review tier:     confirm (no change was right, or the PR is good and merged), redo (back to the free workers with a
#                    note), escalate (beyond this tier)
#   escalated tier:  solved, redo, escalate
#   owner:           decide_done, decide_open, park
WORKER = {"merged": "done", "unchanged": "review", "pr": "review", "pr_not_created": "review",
          "guard_blocked": "escalated", "no_claims": "escalated", "failed": "open", "budget": "open", "lease_expired": "open",
          "retry": "open"}
TABLE = {("open", "claim"): "in progress"}
TABLE.update({("in progress", o): s for o, s in WORKER.items()})
TABLE.update({("review", "confirm"): "done", ("review", "redo"): "open", ("review", "escalate"): "waiting on the owner",
              ("escalated", "solved"): "done", ("escalated", "redo"): "open", ("escalated", "escalate"): "waiting on the owner"})
for s in STATES:   # the owner may settle any item, from any state; nothing else is implied
    TABLE.update({(s, "decide_done"): "done", (s, "decide_open"): "open", (s, "park"): "parked",
                  (s, "decide_blocked"): "blocked", (s, "decide_scheduled"): "scheduled", (s, "decide_owner"): "waiting on the owner"})


def classify(result, staged_empty, landed):
    """A worker run's outcome from the structured fields of its result.json (never from free text the model can steer;
    review of Q60, 3), whether anything was staged, and how it landed (merged | pr | pr_not_created | none). Pure."""
    if landed in ("merged", "pr", "pr_not_created"):
        return landed
    r = result.get("result", "")
    if r == "quota":
        return "budget"          # a provider's allowance is spent (HTTP 429): wait for the reset, not a failure (2026-11-20)
    if r == "failed_call":
        return "failed"          # a model call failed: infrastructure, counts toward escalation like any failure
    if r == "stale":
        return "retry"           # the target changed under the run: redo, not a failure
    if not staged_empty or r == "merge":
        return "failed"          # a change was ready but did not land, or nothing was staged for an accepted change
    if r == "duplicate":
        return "unchanged"       # the card exists already: the next tier confirms
    if r == "no claims":
        return "no_claims"
    if r == "rejected":
        return "guard_blocked"   # the reviewer refused every edit
    if r == "pr" and isinstance(result.get("proposed"), int):
        return "unchanged" if result["proposed"] == 0 else "guard_blocked"   # proposed none / proposed some, all refused
    return "failed"              # an unknown shape is not guessed


class NotAllowed(ValueError):
    pass


def next_status(status, outcome, failures=0):
    """The single transition function. failures: consecutive failed runs including this one (queue_ops.failures)."""
    key = (status, outcome)
    if key not in TABLE:
        raise NotAllowed(f"no transition from {status!r} on {outcome!r}")
    to = TABLE[key]
    if outcome == "failed" and failures >= MAX_FAILURES:   # guard: a crash loop climbs a tier instead of retrying forever
        to = "escalated"
    return to


def main(a):
    if len(a) == 4 and a[0] == "classify":
        import json
        print(classify(json.load(open(a[1])) if a[1] != "-" else {}, a[2] == "1", a[3]))
        return 0
    if len(a) >= 3 and a[0] == "next":
        try:
            print(next_status(a[1], a[2], int(a[3]) if len(a) > 3 else 0))
            return 0
        except NotAllowed as e:
            print(e, file=sys.stderr)
            return 2
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
