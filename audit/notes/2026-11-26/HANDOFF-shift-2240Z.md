# Handoff: scheduled shift 2026-11-26 22:40Z (Claude tier)

## Health
- Last 30 runs of worker, propose, review, tier and tests: all successful. make-work last ran 2026-11-20. It is dispatched by hand, so that gap is expected.
- The last idle job (run 45904409184) ran cleanly. The generator was at its cap (10 today). The splitter accepted 0 items ("nothing passed the script check").
- **Fault found: the free workers had no work** (worker lane 0) and K86 was an orphan. Cause: the tier had parked K85 and K88 "for a smaller split". Their parents (P5d, P3) are in the claude lane, so no automatic job would ever cut them.
  - K88 failed for a different reason than size. The prepare step refused it twice (runs 45832352658, 45833902349), because code_job.new_ok cannot create a file in a folder that does not exist yet (tasks/drift/). Fixed at the cause: render.py now refuses an open or proposed code item that the prepare step would refuse. It applies the same rule against origin/main's tree. Known-answer check: a bad spec is flagged and a well-formed K85 is not (pilot 223c3cfaf).
  - backlog.py put open experiment items in the claude lane ("no job type fits"), although dispatch_experiments.py starts them. Fixed in PR #247 (worker suite 169 OK).
- Lanes at close (backlog.py): parked 95, claude 56, blocked 20, owner-only 7, scheduled 1 (K89), worker 1. Orphans 0, stale 0. Flow over the last 7 days: 221 created, 131 finished (the previous shift recorded 218 and 137).

## Review and escalated lanes drained
- R83, R84 and R85 confirmed -> done. Cards checked: quotes are script-checked, the untrusted label is present, and the quoted text holds no instructions. The evidence is thin: 1-4 claims each, mostly experience reports or repository blurbs. PRs #225, #226 and #227 squash-merged.
- R86 parked, with a recorded reason: it asks for a reference catalogue, which is the wrong job type for the literature pipeline. The YES control is worded for empirical studies. It scored 0.29 and 0.22 against the 0.3 bar, while 15 empirical pools scored 0.717-1.000 (relevance-control/RESULT.md). Only 5 items hit 'controls wrong' in 7 days, and the other 4 finished, so no filter redesign. Reopen when there is a reference-lookup job.
- K88 done (decide_done) via PR #244: build_drift and tasks/drift/card.json with the digests pinned. Two builds gave identical digests, and tasks/test_drift.py passes 3 tests.
- K85 done (decide_done) via PR #245: propose.py takes main(argv), --arm baseline|packet and --packet (exit 2 if missing, no fallback), and makes one repair call through repair.run_arm. 28 tests OK; dry runs OK. K86 is unblocked.

## Claude lane
- P3 (top of the lane): filed **K89**, an experiment with origin follow-up. Its rule was committed first in experiments/drift_online.py (PR #246). It holds if online minus static is at least 0.10, the bootstrap p2.5 is above 0.05, and online beats majority by at least 0.10. One no-cost local run gave holds: online 0.915, static 0.449, majority 0.529. That is not the record. dispatch_experiments has already scheduled K89; the experiment.yml run decides.
- The README's Next step 3 now names the drift task and K89 (PR #248).
- No items with origin "new" were added this shift, so no blind review was needed.

## Owner-only (unchanged)
Q29 repo-settings; K75 guard-merge (PR #208); P11, P14, P16, P17, P18 owner-hold.

## Open, in order
1. K89 **held of record** (experiment.yml run 45907417078, 22:52Z): online 0.915, static 0.449, majority 0.529, difference 0.466 (bootstrap p2.5 0.441); K89 is done. I filed K90 (follow-up, Claude lane): design a labelled-feedback (prequential) mode in runner/CONTRACT.md, then cut it into worker-sized code items.
2. K86 (now unblocked): change propose.yml so it takes an arm input and writes packet-pilot records. Do the dry run first, and add tasks/test_drift.py to tests.yml in the same PR. It is a guard path, so it needs the owner's merge.
3. tests.yml still does not run tasks/test_drift.py (bundle this with K86).
4. Claude lane by value, then age: P33, P35, Q84, P5e, P39, P38, P40, Q85.
5. Open PRs outside the review lane: #237 (workbench graph view), #219 (P28 result), #208 (K75, owner), #157 (docs).

## Spend
No paid API calls; Claude subscription only. One local experiment run on this machine.
