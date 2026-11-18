# Transfer step 1, first run (2026-11-18): rankings differ

Plan registered at `c5949a5` (`../../PLAN.md`); both runs were made from that commit, no file amended.
- **Main run:** Actions run 44237908056 → `main/`. Hosts: optdigits and text on AMD EPYC 9V45, adult on EPYC 9V74.
- **Budget check** (4× epochs, the families of the accuracy flips): run 44238804472 → `budget-x4/`. Hosts: Xeon 6975P-C
  (optdigits, text), EPYC 7763 (adult).

An independent agent re-applied the rule from the raw per-item data with its own script and got the same
flips and verdict for both runs; its wording fixes are applied here. Every number below is in those folders (`decision.txt`, `decision.json`, `result-*.json`, which also hold every
artifact's class for every eval item). Eval splits only; nothing was tuned on them.

## Verdict by the fixed rule: **rankings differ**
| measure | flips | which |
|---|---|---|
| accuracy | 3, over 2 family pairs (2 survive the budget check, one per pair, both digits vs text) | linear vs mlp64 and mlp16 vs mlp64: the larger MLP wins on digits, loses on text (both survive at 4×). linear vs mlp64 on adult vs text: does **not** survive (at 4× linear and mlp64 tie on adult) |
| instructions per item | 4 | lgn cheaper than linear and than mlp16 on digits only; dearer on text and adult |
| native ns per item | 2 | lgn faster than mlp32 on digits, slower on text (same host) and on adult (another host; each order is measured within one host) |

## The table (main run; mean of 3 seeds; bold marks the top value, not a rule-backed win: close values are ties)
| | optdigits acc | text acc | adult acc | Ir/item: digits / text / adult |
|---|---|---|---|---|
| linear | 0.961 | **0.334** | 0.853 | 4,729 / 6,975 / **397** |
| mlp16 | 0.973 | 0.333 | **0.858** | 8,332 / 9,710 / 1,975 |
| mlp32 | 0.982 | 0.331 | 0.857 | 15,766 / 18,692 / 3,721 |
| mlp64 | **0.983** | 0.317 | 0.858 | 30,620 / 36,652 / 7,209 |
| lgn | 0.969 | 0.315 | 0.817 | **3,939** / 20,060 / 2,526 |
| majority class | 0.093 | 0.149 | 0.764 | — |

## What it says, and what it doesn't
- **One-task results don't carry over here.** The order of families changed between digits and text on accuracy,
  and between digits and the other two tasks on cost, under one fixed recipe per family.
- **The logic-gate network's cost lead is a digits fact.** It is 0.47× mlp16's instructions on digits, 2.1× on
  text and 1.3× on adult. Likely cause (hypothesis, not tested): its size rule (50 output gates per class, so 3,250
  on text's 65 classes) and the bit-sliced count per class, against table models whose cost grows with the bytes
  per item, not the classes. It was also last or near last on accuracy on text and adult, and underfit on adult
  (training accuracy 0.81–0.84).
- **The surviving accuracy flips look like capacity under a fixed L2** (hypothesis, not tested: no run varied L2
  or width per task), not kinds of structure: mlp64 fits text's training items far better (0.70; 0.82 at 4×) and
  does worse on eval (0.317; 0.295 at 4×), so more training made it worse. Whether per-task tuning removes the flip is
  untested; the plan tuned nothing, on purpose.
- **Not shown:** that any family is best at anything in general; anything about PFNs, trees or symbolic systems
  (not run); any time comparison between hosts (each task ran on its own host).
- **Limits** are listed in the plan: three tasks, one encoding per kind, three seeds, no jail, a generic logic-gate
  program that is not lgn_d's compiled circuit.

## What follows (from the plan)
Step 2: a general task contract (bytes in, bytes out, a checker) and a mixed suite, before any cross-task claim.
Two lessons for it, from this run: a per-task tuning budget (on a split inside training) belongs in every task card,
or capacity effects pass for structural ones; and a size rule that grows with the number of classes needs its own test.
