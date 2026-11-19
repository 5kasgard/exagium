# P5b: are P5's best proposals better on data the selection never saw? (plan fixed before any run, 2026-11-20)

**Why.** 100+ proposals were scored on one dev split and the best reported. On digits, 32 valid proposals with a median dev
SE of 0.0063: if all were equally good, the best would read about 2.07 SE (0.013) high from luck alone, more than the gap
between the top proposal (0.9864) and gbdt (0.9791). The claim "above every family" is not established (queue P5b).

**Candidates, fixed now** (top 5 on digits dev by accuracy, proposals branch at 3a8dc25):
prop-44492370928-2, prop-44587812936-1, prop-44610648477-1, prop-44549931886-1, prop-44601956368-1;
baselines label-gbdt, label-mlp16, label-linear, majority.

**Data.** The confirmation split is 20% of digits train (seed 20261120, stratified by nothing, a plain shuffle): 573 items.
Every candidate retrains on the other 80% (2,294 items). Neither dev nor test is used. No proposal or proposer ever saw
these 573 items as evaluation data: proposers never see data at all, and P5 scored only on dev.
Caveat stated now: the confirmation items were in train for the original P5 scoring, so a proposal tuned to train quirks is
not excluded; the split removes selection on dev, which is the bias measured above.

**Measures.** Accuracy with paired SE against each baseline (runner/run.py evaluate, 3 rounds), Ir/item and ns/item.

**Rules, fixed now.**
1. A proposal's "above every family" stands if its paired difference over each of gbdt, mlp16 and linear exceeds 2 SE on
   the confirmation split.
2. A proposal is an advance on the accuracy-cost front if no baseline is at least as accurate (within 1 paired SE) and
   cheaper in Ir/item.
3. Winner's curse measured: the mean drop of the five from dev to confirmation accuracy, minus the baselines' mean drop
   (both splits differ in size and items, so baselines are the control).
Forecast (Claude, before running): rule 1 holds for 0 of 5 (p = 0.7); rule 2 for at most 1 (p = 0.6); the five drop more
than the baselines (p = 0.75).
