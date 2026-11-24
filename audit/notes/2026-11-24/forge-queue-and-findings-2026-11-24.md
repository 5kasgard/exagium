# Forge: queue and findings re-analysis — 2026-11-24

Window: since the last Claude handoff (2026-11-22 ~08:30Z). Sources: live clone of 5kasgard/exagium (main 2d64e6b3, pilot 3c1714a3), the GLM round-1/round-2 reports, the Z.ai 22-Nov state report and upgrade path, the 23-Nov delta audit, and the 24-Nov hardening handoff. Numbers re-derived by script; checked by a separate agent (see end).

## 1. What changed since the last Claude session
- **main:** 39 non-merge commits: 16 worker research cards, 23 by chat sessions (GLM/GPT). Merged: #95, #68, #91 (cards), #97–#100 (14 audit fixes), #101 (README refresh).
- **Unmerged, from 24 Nov:** 7 fix branches + 1 handoff doc. PRs #106 (tier merge bound to reviewed head), #108 (ledger idempotency), #110 (stale OTS explorer = can't check) wait on an independent review. Issues #109 (workflow-only fixes) and #111 (loop/runner resource containment) are open. fix/audit-workflow-edges (F006/F009/F011/F033) is blocked by the 403 on `.github/workflows/*`.
- **Machine:** pilot +142 queue commits, worker-runs +2,239, proposals +175 (board 1,097 / 644 valid).

## 2. Findings: there are three review series and two of them share IDs
| Series | IDs | Total | Fixed (evidence) | Still open |
|---|---|---|---|---|
| GLM round 1 (20 Nov, 83f06fd) | F001–F111 | 111 | 47 by round 2 | carried 63 |
| GLM round 2 (21 Nov, 5301510d) | F112–F136 new + 63 carried | 88 open | since then: F054 F112 F113 F115 F117 F123 F124 (+F125 paid retry, F106/F082/F088/F032 overlap with 22-Nov fixes) | ~76 |
| Z.ai state audit (22 Nov, df7890aa) | its own F001–F070 | 70 | 13 merged 23 Nov (F001 F002 F014 F016 F017 F019 F024 F036 F038 F041 F065 F066 F067) | ~57 |
| 23-Nov delta | F001 (dead HN link in R32) | 1 | — | 1 |
| 24-Nov GPT hardening | PRs #106/#108/#110, issues #109/#111 | — | 0 merged | all |

- **The ID collision is a defect in its own right:** "F001" is a decisions_probe bug in the GLM series and the forgeable grader in the Z.ai series. The upgrade-path report already mismapped G1 (it is GLM F068, not Z.ai F068). Fix: prefix the ID with its series (G1-F001, Z1-F001).
- **Many findings overlap across series:** for example the seccomp denylist (GLM F104 = Z.ai F004 = Q61), the probe hostname identity (GLM F087 = Z.ai F060 = G12), README drift (GLM F106 = Z.ai F068), the UTC overdue check (GLM F088 = Z.ai F065, now fixed), and future 'at' in facts (GLM F082 = Z.ai F036, now fixed). The unique open total is therefore lower than ~133, probably ~100–110.
- **Mechanical check of the 63 carried GLM findings at main:** for 34, the file is byte-unchanged (so still open). For 22, the evidence quote is still present. For 7, the code changed: F106, F082, F088 and F032 are fixed via the 22-Nov work; F022, F035 and F038 are still open on reading.
- **Where findings live:** GLM round 1 is in the queue: 23 G items, plus bundles Q63–Q68 and Q76, which cover 38 of the 63 carried IDs. 25 carried IDs have no queue item: F001 F012 F015 F027 F030 F035 F046 F047 F049 F054 F060 F068 F085 F092–F094 F097 F098 F100–F106. Nothing from GLM round 2 (F112–F136) or the Z.ai series is in the queue. Those were fixed straight from chat sessions or not at all. GLM F112–F136 exist only in the project's docx/zip, not in the repo.

## 3. The queue (313 items: 127 done, 77 open, 76 parked, 27 owner, 6 review)
**Open 77: no worker can take any of them** (no `targets`/`research`/`code` field). By what they need:
- **Already built: close as done with evidence:** P5 proposer, P7 research worker v1, P8 queue worker, P10 (tier 1 only: split tiers 2–4 into a new item), P27 continuous operation, P30 facts register, P5b confirmation, Q58 stall (keeper #78), Q25 (classifier: already_done), Q59 (handoff 22 Nov: rule met).
- **Duplicates of findings:** Q61 (= the seccomp finding) and Q63–Q68 (GLM bundles; Q63 partly fixed via F088). Merge with the findings register below rather than keeping two copies.
- **Owner-held ("add to the queue only, no work yet", 19 Nov):** P11, P12, P13, P14, P15, P17, P18 and P19 (distillation). Nothing will ever start these unless the hold is lifted.
- **The real plan, unblocked but untyped:** P3, P6, P9, P20 family, P21, P22, P23, P24, P25, P26 family, P28, P29, P31, P32, Q54, Q69–Q81. These need cutting into worker-sized code, doc or research items. The splitter's record is 4 done of 56.
- **Owner actions:** Q09, Q28, Q74, Q80, P7b (YouTube key is now added per memory).

**Waiting on the owner (27):**
- 15 research cards that found nothing after being narrowed. Close them as "no evidence found", with the search log as the result.
- 7 code PRs (#24, #38, #40, #47, #66, #67, #80) refused by the tier only because they touch files a person merges. Each needs a merge or a close.
- The rest are 3 decisions (Q29, Q38, P16) and 2 research cards with claims (R10, R15) that need a read.

**Review 6:** R26 and R32 are already merged on main (the queue is stale). R30 is the distillation literature card, the one closest to the stated purpose.

**Parked 76:** 46 failed split children (29 too big again), 18 Q, 10 G, 1 R, 1 P.

## 4. Why the system looks like a benchmark tool, and why work piles up
1. Plan items can't be picked up by any worker, and the purpose items are on hold by the owner's own 19-Nov rule.
2. The only self-contained job type that works is research cards, so the generator keeps making them: 40 of the 70 items added since 21 Nov are research cards.
3. Findings are fixed from chat sessions outside the queue, in separate ID spaces, with no single register. Each review rediscovers parts of the last one.
4. Finished work isn't closed out. Queue state lags the repo, so every outside reader sees a bigger backlog than is real.

## Verification
A separate agent re-derived sections 2 and 3 from the repo and confirmed every count:
- 20 fixed F-IDs named on main since 5301510d.
- 34 / 22 / 7 split for the carried GLM findings. GLM F038 is still live at engines.py:235.
- P5, P8, P30, P5b and Q58 built; P10 partly built.
- R26 and R32 cards are on main.
- 15 empty research escalations and 7 merge-blocked PRs.
- No Z.ai-series or round-2 items in the queue.

Caveats it raised:
- The 7 quote mismatches need a manual read, not an assumed fix.
- F124's commit carries no series label.
- It could not check live PR state.

## 5. Proposed changes (not yet applied)
1. **Close-out pass on the queue:** close the ~10 built items, R26 and R32, and the 15 empty research escalations; record evidence on each.
2. **One findings register:** all three series in one file, series-prefixed IDs, duplicates merged, status from code. Retire Q61 and Q63–Q68 into it.
3. **Lift the hold on the path to distillation:** P11 (or test Gnomon as the store, P25), then P19's first test. Cut each into typed items by hand rather than through the splitter.
4. **Merge or close the 7 stuck PRs** and the 3 reviewed 24-Nov PRs once their independent review is recorded. Apply the 4 workflow fixes yourself, or give the tool workflow scope.
5. **Cap research-card generation** until the R backlog clears.
6. **Delete the 76 dead worker branches.**
