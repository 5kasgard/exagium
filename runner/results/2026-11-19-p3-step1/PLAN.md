# P3 step 1: outside-strength families on the three label tasks (plan, fixed before any run)

Written 2026-11-19 before the new candidates were written or run. Commit time is the registration time.
Dev split only; the test split stays closed (no ledger plan).

## Why
An outside review (2026-11-19, reviews/44470377631, finding 2) found the suite's baselines too weak to say anything:
only majority, linear and a 16-unit MLP. Transfer step 1 found family rankings change across tasks. The queue's Q54
hypothesis (cheapest artifacts are those whose structure matches the hardware's own primitives, e.g. lookup tables)
was weakened by the logic-gate network on text and tabular data, but lookup-table families were never tested.

## What runs
Tasks: optdigits, text (next character, 65 classes), adult (tabular), as built by `tasks/build.py`.
Existing candidates, unchanged: majority, label-linear, label-mlp16.
New candidates (numpy training, C inference, deterministic, single thread, the same contract):
- **label-gbdt**: gradient-boosted trees, one multi-output tree per round (softmax gradients, second order), splits
  on byte values. Outside-strength baseline for tabular data (published comparisons find tuned tree ensembles strong
  on tabular data; to be checked against research card R5 when it lands).
- **label-knn**: k nearest neighbours, per-column distance by the card's types (ordinal: scaled absolute
  difference; categorical: mismatch).
- **label-wisard**: a WiSARD n-tuple classifier (weightless, lookup tables only) with thermometer or one-hot bits and
  bleaching. The Q54 test family.

Tuning, the same for every new family: the tuning split is the last 20% of `train.in` in file order. Each new family
chooses at most 6 settings on it (fixed grids in its `train.py`), then retrains on all of `train` with the chosen
one. Existing candidates keep their fixed settings (a stated limitation: the new families get a tuning budget the old
ones do not).

## Measures (runner/run.py, as for P2)
Per candidate and task: per-item scores (kept), dev accuracy and SE, simulated instructions per item (cachegrind),
native ns per item. Paired comparisons use per-item outcomes: SE of the difference from paired items.

## Questions and rules (fixed now)
1. **Floor margin** (review finding 2): a task stays in the suite only if its best family beats majority by at
   least 3 paired SE. Forecast: holds on all 3 (p = 0.97).
2. **Does an outside-strength family change the picture?** For each task, does the best new family beat the best of
   linear and mlp16 by more than 2 paired SE? Forecasts: adult (gbdt) p = 0.65; optdigits (knn) p = 0.25; text p = 0.35.
3. **Q54 for lookup tables.** On each task, wisard is "matched" if its accuracy is at least that of the more accurate
   of linear and mlp16 minus 2 paired SE, and "leads on cost" if, when matched, its instructions per item are lower
   than that family's. Supported if matched and leading on both non-digit tasks (text and adult); weakened if only on
   optdigits or on none; mixed otherwise. Forecast: supported p = 0.2, mixed p = 0.3, weakened p = 0.5.
4. **Reproduction**: every new candidate's training must give identical weights twice (`run.py reproduce`). A
   candidate that does not reproduce is reported and excluded from 1-3.

Forecasts are Claude's, stated so calibration can be scored later (queue P23, P11).

## Not claimed by this step
Anything about the test split; any task beyond these three; tuned versions of linear and mlp16.
