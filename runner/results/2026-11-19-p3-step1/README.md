# P3 step 1 result: outside-strength families on optdigits, text and adult (2026-11-19)

Plan and forecasts fixed before any new code existed: [PLAN.md](PLAN.md) (8408ef2). Families: 8f46c37. CI: runner run
44486290814 (ubuntu-24.04, bubblewrap jail), raw output in [ci/](ci/); the rules applied by [decide.py](decide.py) in
[DECISION.md](DECISION.md). Dev split only; every number below holds for these three tasks, this host type and these
training recipes.

## Answers to the fixed questions
1. **Floor margin: holds on all three tasks** (best family over majority by 85.3, 38.2 and 20.5 paired SE).
2. **An outside-strength family changes the picture on two of three tasks.** Gradient-boosted trees beat the best of
   linear and mlp16 on text (0.392 vs 0.338, +10.4 paired SE) and adult (0.866 vs 0.859, +2.4 SE), not on optdigits
   (0.979 vs 0.977, +0.5 SE). The accuracy costs instructions: on text the trees use 131,154 instructions per item
   against linear's 7,360 (18 times), on adult 7,570 against mlp16's 2,325 (3.3 times).
3. **Q54 for lookup tables: weakened.** WiSARD (lookup tables only) was less accurate than the better of linear and
   mlp16 on every task (-3.1, -25.5 and -7.0 paired SE) and also used more instructions per item (34,076 vs 8,686 on
   optdigits; 119,869 vs 7,360 on text; 20,923 vs 2,325 on adult). Together with the logic-gate network's result on text and
   tabular data in transfer step 1 (a different family, made of logic gates), both families of this kind tested so far weaken the hypothesis that structures
   matching the hardware's own primitives are cheapest at matched accuracy. Families not yet tested: constraint
   propagation, Tsetlin machines, cellular automata, event-driven.
4. **Reproduction: all three new families gave identical weights twice on every task.** The same 9 weight files
   were also trained on a second machine (this chat's workspace, a different CPU): all 9 byte-identical to CI's
   (sha256 in ci/reproduce-*.json), so these families reproduce across hosts too.

## Forecasts (Claude's, fixed in PLAN.md) against outcomes
| question | forecast | outcome |
|---|---|---|
| floor margin on all 3 | 0.97 | yes |
| new family beats old, adult | 0.65 | yes |
| new family beats old, optdigits | 0.25 | no |
| new family beats old, text | 0.35 | yes |
| Q54 supported / mixed / weakened | 0.2 / 0.3 / 0.5 | weakened |

Brier score over the four yes/no forecasts (the Q54 three-way forecast is reported in the table only): 0.152 (lower is better; 0.25 is what always saying 0.5 scores). One run is
far too few to judge calibration; this row is the start of a record (queue P23, P11).

## Limitations, stated
- **Grid edges.** kNN chose k = 15, the largest value in its grid, on text and adult; gradient-boosted trees on
  optdigits chose 293 rounds of the 300-round cap during tuning (text: 260; adult: 59, both inside the cap; from the
  tuning logs of a local run whose 9 weight files are byte-identical to CI's, so the same settings were chosen; CI
  keeps no training logs). kNN's k comes from the same local logs. Both families might do better with wider grids. The
  grids were fixed in the plan and were not changed after local dev scores were seen.
- **Unequal tuning.** The new families chose settings on the tuning split; linear and mlp16 use fixed settings.
- **Categorical columns** are split by trees on their byte codes (an arbitrary order), and compared by kNN as equal or
  not; WiSARD uses one-hot bits for the 8 or 16 most frequent values only.
- **Cost** is instructions and native time per item on one CI host type; kNN's cost grows with the training set.

## What this changes
- The suite's strongest baseline on text and tabular data is now gradient-boosted trees, at a much higher cost per
  item: an accuracy-cost frontier, not one winner. Later claims on these tasks are measured against it.
- Q54 needs a new kind of evidence or should be narrowed; it is not supported by any family tested so far.
