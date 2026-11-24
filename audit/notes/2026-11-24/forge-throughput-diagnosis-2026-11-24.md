# Forge throughput diagnosis — 2026-11-24

Source: live repo 5kasgard/exagium (main 2d64e6b3; pilot/lgn-20261115 tip 3c1714a3, queue audit/2026-11-17/queue.json). A separate verifier agent re-derived and confirmed all 7 numeric claims.

## Queue, last commit per UTC day
| day | total | done | open | parked | owner | +added | +done |
|---|---|---|---|---|---|---|---|
| 11-18 | 74 | 31 | 25 | 17 | 1 | 74 | 31 |
| 11-19 | 156 | 74 | 58 | 18 | 3 | 82 | 43 |
| 11-20 | 243 | 95 | 77 | 28 | 28 | 87 | 21 |
| 11-21 | 280 | 107 | 87 | 67 | 15 | 37 | 12 |
| 11-22 | 293 | 115 | 77 | 76 | 21 | 13 | 8 |
| 11-23 | 303 | 122 | 77 | 76 | 23 | 10 | 7 |
| 11-24 | 313 | 127 | 77 | 76 | 27 | 10 | 5 |

- **Since the end of 11-20:** 70 added, 32 done (2.2 : 1).
- **Added:** 40 R (research) and 30 K (split children).
- **Done:** 22 R, 5 D, 3 K and 2 G.

## Root cause
- **No free worker can take any open item.** All 77 open items lack a job type (no `targets`, `research` or `code` field), so `worker/queue_ops.ready()` never offers them. For 53 of them every blocker is done; the missing field is the only thing stopping them.
- **The open set is the real build plan:** P3 mixed suite, P11 knowledge store, P13 tool-making, P15 hardware search, P19 distillation, P20 orchestration, P26 ports, and the rest. 74 of the 77 have zero runs.
- **The splitter is the only automated route from a P item to code, and it mostly fails:**
  - 56 children: 4 done, 46 parked (29 resplit, 10 reject, 7 park), 6 waiting on the owner.
  - 37 open parents are marked split_tried; about 40 were never tried.
- **Some open items were already built:** P8, P5, P7, P10, P30 and P5b. They were closed with evidence on 2026-11-24.
- **Result:** autonomous throughput is research cards plus small fixes, so readers of main see a measurement harness.

## Direction the data supports
1. Close out built items with evidence. (Done 2026-11-24.)
2. Cut the unblocked P items into worker-sized typed items, starting on the path to P19 (distillation).
3. Cap research-card intake until the R backlog clears.
4. Merge or close stale branches, and prune dead worker branches.
