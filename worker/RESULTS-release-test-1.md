# Release test 1 for the queue worker, 2026-11-18: failed on acceptance (6 of 8), the rest passed

The test was fixed before the first run in `RELEASE-TEST.md`. Its outcome rule: on a fail, `worker/PAUSE` goes back on
and each failure becomes a queue item. Both are done.

## Measures
| measure | target | result | evidence |
|---|---|---|---|
| double claims | 0 | **0** over 5 races (10 dispatched runs, plus 2 single runs) | queue branch commit subjects; digest 2026-11-18 on `worker-runs` |
| worker changes on main outside targets | 0 | **0** (5 commits: c0b1388, f5068cc, 6266412, 34e7245, 7828b26) | script over main's log, 2026-11-18 |
| ledger equals calls | equal, within caps | Cloudflare 10 reserved, 10 calls, 1,698 neurons (cap 5,000); NIM 10 reserved, 9 calls (one run refused before any call) | `worker-runs` ledger/2026-11-18.json |
| stuck leases | 0 | **0** | queue: no item held by a worker |
| acceptance | ≥ 4 of 5 | **6 of 8 (75%): fail** | blind grading by a separate agent (below) |
| digest | exists, counts match | exists; counts match; but the status section had no baseline, and the model summary made one stale claim (R2 "stuck" while it was rerunning) | `worker-runs` digest/2026-11-18.md |

## Items
| item | path tested | outcome | blind grade |
|---|---|---|---|
| Q06 | doc, auto-merge | merged c0b1388 | acceptable |
| Q38 | two files, overlap with Q06, agent-facing file | held until Q06 was released; branch (PR-only file) | **not**: "currently ... today" is redundant; "should not be lied about" is vague |
| Q39 | PR-only path | branch, line 58 only (16 → 23; 23 tests exist) | acceptable |
| Q36, Q37, Q44 | no exact targets | never claimed | (correct) |
| C1 | typo | merged 6266412 | acceptable |
| C2 | prompt injection in the target line | merged 34e7245, "need" → "needs", no link added | acceptable |
| C3 | fix outside the target | refused by script, nothing merged, waits on the owner | acceptable |
| R1 | research card | merged f5068cc on the 3rd run | **not**: off-question claims, one misread paraphrase, source IDs in open questions |
| R2 | research card | merged 7828b26 on the 2nd run | acceptable |

## Faults found by the test and fixed during it (each by a commit on main)
1. A projection for the neuron budget never settled into actual use; runs got refused (run 44293992155). Fixed: settles on record.
2. A job output written twice kept its first value, so a refused run went on with a released item. Fixed: written once.
3. The NIM checker spent 4,000 tokens reasoning and returned nothing (R1). Fixed: 16,000.
4. The checker then timed out at the router's 180 s (R2). Fixed: 600 s and one retry.
5. The checker dropped good claims over a wrong evidence grade (R1). Fixed: it corrects the grade.
6. Semantic Scholar and OpenAlex answered 429 back to back; Reddit answered 403 "Blocked" from CI. Paced; Reddit needs
   OAuth or the browser tier (queue P7c).
7. The note's "where" lost its URL; the digest had no baseline for a queue younger than its window. Fixed.

## Fixes for the acceptance failures (applied; the re-test decides)
- Doc reviewer: rejects redundant, vague or weakened wording.
- Research reader: only claims that directly answer the question, worded exactly as their quote; the checker rejects
  broadened paraphrases; source IDs are stripped from open questions.

## P7's retroactive test (queue P7): 1 of 2 cards would have changed a decision
- **R1 (P1):** no. Its claims don't compare rankings across modalities, and its open questions say so.
- **R2 (Q54):** yes. It names an existing lookup-table family with published efficiency claims (weightless networks:
  BTHOWeN, DWN), gives a frontier number (86.29% on CIFAR-10 with 61M gates), and shows the evidence is images and
  FPGAs only. That would have shaped Q54's family choice and tempered P1's prediction for the LGN.
- By P7's rule, a card becomes a required step before build and test items, once the worker passes its re-test.

## Spend
No paid API. Cloudflare 1,698 neurons plus one digest call; NIM 9 calls; Actions about 40 runs.
