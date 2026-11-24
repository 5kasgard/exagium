# Handoff: scheduled Claude shift, 2026-11-24 11:00–11:50Z

Copied from the owner's project so other agents can read it. Follows HANDOFF-chat-2026-11-22-research-filter.

## Blocker: this shift could not write the queue
- **Cause:** the session's GH_TOKEN was invalid. The GitHub connector can only write whole files, and queue.json is 740 KB.
- **Result:** the moves below were recorded but not written.
- **Owner:** refresh the token the shift runs with.
- **Update:** the Claude chat wrote these moves at 11:45Z.

## Health (runs since 01:50Z, public API)
- **worker.yml:** 53 succeeded.
- **propose.yml:** 27 succeeded.
- **tier and wiring:** all green.
- **tests.yml:** 4 failures, all on the 05:0xZ fix-branch PRs. Re-runs were green, except fix/tier-merge-preconditions-20261124 (run 45331365511).

## Fault found and fixed (PR #115)
- **What went wrong:** F038 (be136ee7) counts a missing reproduce.json or dev.json as "not scored". That also caught proposals whose own code crashed.
- **Size:** 59 of 63 "not scored" rows are the candidates' own crashes.
- **Effect on the valid rate:** 0.62 becomes 0.59 (767 of 1,297).
- **Fix:** `judge.own_fault()` reads the first FAILED line, which must name the candidate. Infrastructure failures stay "not scored".
- **Checks:** the full tests.yml command passes locally. A Sonnet reviewer approved after 3 real points were fixed.
- **After merge:** rescore the 59 rows, then refresh the README proposals row.

## Drain (decided by the shift, written by the Claude chat)
- **R26 and R32:** confirm (merged).
- **R30 (PR #92):** recommend merge, the distillation card.
- **R62 (PR #102):** recommend merge.
- **R43 (PR #96):** recommend close.
- **R36 (PR #94):** redo with queries on LLM API drift (Chen, Zaharia and Zou 2023, arXiv 2307.09009).
- **Q59:** its close rule was met.

## Open
1. Owner: fix the shift's GitHub token. Decide #115, #92, #102, #96 and #94.
2. P28 step 3: the base harness inside the 8 task images, dry run first.
