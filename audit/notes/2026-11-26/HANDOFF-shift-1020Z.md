# Handoff: scheduled shift 2026-11-26 (started 22:42Z 11-25, resumed 10:21Z 11-26 after a usage limit)

## Health
- **Proposer chain was dead 7 h** (15:51Z-22:42Z 11-25): run 45646733762's record step got `HTTP 500` from GitHub's dispatch API and exited; the KEEP chain (no PAUSE) had nothing to restart it. Restarted by hand (run 45715524383). Fixed at the cause in PR #213 (3 dispatch attempts, 20/40 s backoff; no-cost check: stub `gh` failing twice -> ok on try 3). Owner merged #213 at 23:12Z.
- Idle job (run 45709325792): clean; generator/splitter skipped because the claim job had work.
- Lanes at close (backlog.py): parked 94, claude 55, blocked 21, owner-only 7, worker 2 (K85, K88); orphans 0, stale 0. Flow, last 7 days: 218 created, 137 finished.

## Review lane drained
- K84 confirm -> done: PR #220 (repair helper) matches its spec; runner/proposals tests OK on main+PR; squash-merged 9c709bca3. Unblocked K85 (now in the worker lane).
- K87 confirm -> done: PR #221 (drift generator) matches its spec; tasks/test_drift.py OK on main+PR; squash-merged 62ad37095. tests.yml still doesn't run tasks/test_drift.py (guard path; bundle with K86, as the 08:40Z shift noted).
- The first half of this shift (before the limit) reviewed #209 (K80) and #210 (K81); the 08:40Z shift has since merged #209. #210 is closed. My review of it (still relevant if K81 comes back): its loader guessed at four record formats, never read the real `dev.json` `candidates[name].items`, turned any truthy score into 1, and lined items up by list index across files. A dev.json from another task would merge into the same candidate names without any error.

## Claude lane
- P3: added K88 (follow-up, worker lane). It wires `drift` into tasks/build.py with seed 2026 and n=8000, splits the stream in order (train blocks 0-1, dev block 2, test block 3), and pins the card digests. It needs a separate, pre-registered online-vs-static experiment item after it lands. render.py passed.

## Owner-only (unchanged)
Q29 repo-settings; K75 guard-merge (PR #208); P11, P14, P16, P17, P18 owner-hold.

## Open, in order
1. Review K85 and K88 PRs when they arrive.
2. K86 workflow change (after K85): dry run first; add tasks/test_drift.py to tests.yml in the same PR.
3. P3: file the online-vs-static experiment with its decision rule once K88 lands.
4. Claude lane by value then age: P33, P35, Q84, P5e, P5d.
5. Open PRs outside the review lane: #219 (P28 result), #157 (docs).

## Spend
No paid API calls; Claude subscription only.
