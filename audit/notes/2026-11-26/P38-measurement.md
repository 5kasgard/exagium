---
recorded_by: Claude Opus 5.5 (claude.ai chat session), from the scheduled P38 check-in
when: 2026-11-26T20:40Z (check-in fired 2026-11-25T11:37Z; read late)
source: audit/2026-11-17/queue.json runs (state branch); worker.yml idle-job logs (runs 45634431138..45693735129)
kind: measurement
item: P38
---
# P38: did #159 (literature items get research jobs) let free workers finish backlog items?

Window: 24 h after #159 merged (2026-11-25T05:34Z) against the 24 h before.

| | free-worker runs on P/Q items | finished P/Q items |
|---|---|---|
| 24 h before | 11 (claim 5, guard_blocked 2, failed 1, pr 1, no_claims 1, merged 1) | 1 (Q73, after its touches were narrowed by hand) |
| 24 h after | 2 (claim 1, merged 1) | 1 (Q77, also narrowed by hand) |

Generator attachments ("research jobs on literature items" commits) since #159: **0**.

Why: every idle-job generator run in the window logged `generate: cap reached (10 today)` and made no call.
#159 counts attachments toward the same daily cap of 10 research items, and the cap was already used up
for the UTC day, so the change never got to run. The measurement is therefore **inconclusive, blocked by the cap**,
not a negative result.

Next (hypothesis to test, not imposed): give attachments their own small daily allowance (or count them
separately from new R items), then measure again over 24 h from the next UTC midnight.
