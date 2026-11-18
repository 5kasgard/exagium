# Transfer plan, step 1: do rankings transfer across kinds of task?

Fixed on 2026-11-18, before any full run. The commit that adds this file is the registration; the run that
answers it must come from a later commit that leaves this file, `decide.py` and the hyperparameters in
`families.py` unchanged (a check in `transfer.yml` compares them with the registered commit).

## Why
Every result so far holds for one task: optdigits (8×8 digit images). If the order of families changes from one
kind of task to another, a one-task Forge misleads its own spec cards, and the next steps (a general task contract,
a mixed suite) are needed before any "best part" claim. If no order changes, the cheaper path stays open.

## Hypothesis
Rankings of the same families differ across kinds of task.

## Tasks (byte records, `tasks.py`; downloads pinned by sha256)
| task | kind | items: train / eval | bytes per item | classes |
|---|---|---|---|---|
| optdigits | vision: 8×8 pixel counts 0..16 | 2,867 / 956 (loop.py's dev split) | 64 ordinal | 10 |
| text | language: next character from the previous 16 bytes, tinyshakespeare | 30,000 / 5,000 (eval from the last 10% of the text) | 16 categorical | 65 |
| adult | tabular: UCI Adult income; 6 numeric (255 training-quantile bins), 8 categorical | 32,561 / 16,281 (official split) | 14 mixed | 2 |

Eval is never used for a choice: no hyperparameter is tuned, on any split. The text metric is next-character
**accuracy**, not bits per character: the logic-gate network and the C programs output a class, not a
distribution, and one metric must serve every family.

## Families (`families.py`, one shared encoding rule per feature kind)
- **linear**: logistic regression. **mlp16, mlp32, mlp64**: one ReLU layer. Ordinal bytes z-scored, categorical
  bytes one-hot. Adam lr 3e-3, L2 1e-4. Exported as per-feature lookup tables (`infer_table.c`).
- **lgn**: logic-gate network with learned Top-K wiring (as lgn_d): K = 8, widths 2000, 1000, 50 × classes,
  tau 4, Adam lr 0.01. Ordinal bytes as thermometer bits at the training quantiles 0.2/0.4/0.6/0.8; categorical
  bytes as one bit per value (64 most frequent). Exported as gates, run bit-sliced (`infer_lgn.c`).
- **Same budget for all:** epochs = min(100, ⌈600,000 / n_train⌉), minibatch 64; 3 seeds (1, 2, 3) each.
- Mandatory baselines: linear is in every comparison; the majority class is reported.
- **Not in this step:** a PFN (TabPFN needs PyTorch and downloaded weights, and its context limit is below
  adult's 32,561 rows; whether it fits CI's CPU limits is untested). Trees and boosting: step 2.

## Measures (`meter.py`, one CI host per task)
- **Per-item outcomes**: every artifact's class for every eval item, kept. The C output must equal the numpy run
  of the exported form byte for byte.
- **Accuracy** on eval, averaged over seeds.
- **Instructions per item**: cachegrind I refs at N = n_eval minus N = 0, over n_eval. Deterministic.
- **Native ns per item**: wall time at 20,000 items (eval repeated) minus N = 0, over 20,000; 7 rounds, random
  order each round, one pinned CPU.

## The rule (`decide.py`), fixed now
For each pair of families A, B and each task, an order holds when:
- **accuracy:** D = mean over items of (A's correct, averaged over seeds) − (B's); SE² = per-item variance of the
  differences / n + seed variance of A's accuracy / 3 + the same for B. A above B when D > 2 SE.
- **instructions:** A cheaper when Ir_A × 1.10 < Ir_B (a deterministic meter has no SE; 10% is a fixed tolerance).
- **native time:** A cheaper when the mean of per-round differences is below −2 SE.

**A flip:** an order holds one way on one task and the other way on another. **Verdict:** "rankings differ" if
there is any flip on any of the three measures; else "no flip found". Any problem in the meter (a C/numpy mismatch,
a failed run) marks the verdict invalid.

**What follows:** "rankings differ" → step 2 (a general task contract and a mixed suite) before any cross-task
claim. "No flip found" → step 2 is still needed for breadth, but a single-task result gains weight. Either
way, accuracy flips and cost flips are reported apart, because cost flips can follow from input shape alone.

## Limits, stated in advance
- Three tasks, one host each, one budget rule: a flip may come from the budget (logic-gate networks often need
  more epochs than MLPs) rather than from structure. Training accuracy is reported so that underfitting shows.
- The encodings are one choice each; a family with its own preferred encoding could rank differently.
- Seed variance from 3 seeds is a rough estimate.
- No jail: every program here was written in this plan; none was proposed by an agent. Step 4's proposals
  must go through loop's jail.
- The generic logic-gate program is not lgn_d's compiled circuit (no constant folding or pruning), so its costs
  are not lgn_d's numbers.
