# Handoff: scheduled shift 2026-11-26 08:40Z (Claude tier, claude-opus-5-5 configured)

## Health
- worker, propose, tier, review, dash-snapshot, schedules: green. tests.yml on main green (last 3 runs); its 5 failures in 40 runs were on worker PR branches (K79, K83), not main.
- **Worker idle on its last 7 runs (08:00-08:40Z)**, with no crash: the idle job ran cleanly (generator accepted 0, splitter "nothing passed the script check", classify "nothing judged", idle re-poll started the next worker). Cause: **no item was in the worker lane** (lanes before this shift: parked 94, claude 54, blocked 19, owner-only 7, review 5, worker 0). Fixed at the cause by making claude-lane work worker-sized (K84, K87 below): `queue_ops.py ready` now lists K84 and K87.
- Clock (#218) holds: schedules.yml dispatched hourly at :05 from 04:05 to 08:05Z, dash-snapshot.yml hourly at :41, sandbox-eval.yml at 03:23 and 06:17Z (Actions runs, event workflow_dispatch). README Progress row added (eaaac83).
- backlog.py after the shift: lanes parked 94, claude 54, blocked 22, owner-only 7, worker 2, splitter 1; orphans 0, stale 0, undated 0. Flow: last 7 days 218 created, 135 finished (queue.json, backlog.py at close; before the shift: 214 / 132).

## Review lane drained (5 items)
- K80 confirm -> done: PR #209 reviewed and merged (f136b0f). Runner/proposals tests 20/20 on main+PR. Note: staged_select already returned scored_items; the wrapper is thin.
- K76 redo -> open, now blocked by K75: PR #214 closed. Its qualifies_pair used thresholds 0.8/0.7 with no source, contradicting K75's pre-registered P31 rule (5/5 controls, >=27/30, two runs). Next step rewritten to call K75's function. tier.py is a guard path, so K76 also ends at the owner (guard-merge), like K75 (PR #208).
- Q38 solved -> done (main 77d1e34): VERIFY-anchors.md:10 says "a result changed after it was anchored can be detected"; record/README has no guarantee left.
- Q68 solved -> done (main 77d1e34): 11 numbers replaced, each re-counted from the files on 2026-11-26.
- K71 redo -> open, blocked by K86: the "invalid: no pilot result" (run 45708023768) came from running the rule before anything produced pilot data (K69, the producer, was parked after PR #152's rejection). Rule unchanged.

## Claude lane
- P5d producer chain filed (origin follow-up): K84 repair helper (worker lane), K85 propose.py --arm (blocked by K84), K86 pilot producer in propose.yml (guard path, Claude tier, then owner-only guard-merge; blocked by K85). K69 stays parked as superseded.
- P3 (top of claude lane, 5.5 days): first worker-sized piece K87, a drifting-stream label task generator (tasks/drift.py, fixed seeds, no download). Next after it: wire into tasks/build.py with a card, then a pre-registered online-vs-static experiment.
- Measured for P38 (free code throughput), queue.json runs since 11-19 on code items: 93 PRs, 20 confirms, 69 guard_blocked, 91 failed, 81 redo, 51 resplit. Reasons in tier redo/resplit notes: model reject 35, "files a person merges" 28, no accepted edit 26, over 80 lines 6. "Files a person merges" is by design (tier auto-merges only experiments/*.py; everything else waits for this tier), so approved non-experiment PRs depend on Claude shifts.
- Note: tasks/test_drift.py is not in tests.yml, so CI will not run it. Adding it needs a guard-path change; bundle it with K86's workflow PR.

## Owner-only (unchanged, nothing new escalated)
Q29 repo-settings; K75 guard-merge (PR #208); P11, P14, P16, P17, P18 owner-hold.

## Open, in order
1. Review K84/K87 PRs when they arrive (tasks/ and runner/ are not auto-merged).
2. K86 workflow change, after K85 lands: dry run first.
3. Report the E2B/CF sandbox cycles (chat handoff 11-26 open item 1).
4. Rest of claude lane by value then age: P33, P35 (all 5 sub-items parked: re-split due), Q84, P5e, P39, P40, P38.
5. Open PRs not in the queue's review lane: #219 (P28 cap-160 result), #157 (architecture docs).

## Spend
No paid API calls. Claude subscription usage for this shift only.
