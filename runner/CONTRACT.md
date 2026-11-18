# The task contract (queue P2): bytes in, bytes out, a checker

**Status: hypothesis, being tested.** Written 2026-11-18 after transfer step 1 showed one-task results don't carry
over (`../transfer/results/2026-11-18/`).

## Hypothesis
One contract expresses every kind of task the Forge means to hold, with **no task-specific code in the runner**:
labels (vision, language, tabular) and tasks scored by a verifier with no answer key (SAT first; proofs and
emitted programs later).

**It fails for a kind of task if** adding that kind needs a change to `runner/run.py` beyond the contract, or if the
same artifact scores differently through the runner than through a direct run (the equivalence check below).

## A task (`tasks/<name>/card.json`, data built by `tasks/build.py`, every download pinned by sha256)
- **Records.** Every input and output is a file of records: a 4-byte little-endian length, then that many bytes.
  Fixed-size tasks just use one length. Inputs and outputs may vary in size per item (a SAT instance, a proof).
- **Splits.** `train`, `dev`, `test`; each has `<split>.in` (inputs) and `<split>.key` (what only the checker sees:
  labels, or empty records when the checker needs no key). Training sees `train.in`, `train.key` and the card.
- **Checker.** `"label"` (the built-in: an output must equal its key byte for byte) or a task's own `check.py`:
  `python3 -I check.py IN KEY OUT` prints one JSON line per item `{"score": s}` with 0 ≤ s ≤ 1. The checker is task
  code, trusted, run outside the jail; it treats the candidate's output as untrusted bytes.
- **Floor.** A named baseline whose dev score a candidate must beat to count (majority class; all-false assignment).
- **Baselines.** Named candidates that every evaluation of the task must include.
- **Test gate.** The test split is scored only under a ledger plan that freezes the runner, the card, the built data
  and the candidates' code, after the ledger verifies (as `loop.py` does).

## A candidate (`runner/candidates/<name>/`): code only, no weights
- `train.py`: `python3 -s -P train.py card.json train.in train.key weights.bin` (stdlib + numpy, deterministic,
  one process, single thread). A candidate with nothing to learn still writes a weights file.
- `infer.c`: `./infer weights.bin in.rec N` writes exactly N output records to stdout. N = 0 is start-up only.
- Weights are made by training in the jail at run time. `reproduce` trains twice and requires identical bytes;
  the test gate requires it.

## What the runner measures (`runner/run.py`; reuses `loop.py`'s jail, timer and counter by import; `loop.py` is not edited)
Per candidate and task, on one host: every item's output and score (kept), the mean score with its SE, simulated
instructions (cachegrind via `counter/count.py`) at N items and at N = 0, and native time at N and at 0 over R
rounds in random order, pinned to one CPU.

## Open questions, to settle with data, not here
- Whether one checker call per split is fast enough for slow verifiers (Lean): per-item calls may be needed.
- Whether a fixed per-task output bound is enough, or proofs and programs need a resource budget per item.
- Tuning budgets: transfer step 1 showed capacity effects can pass for structural ones. A candidate may split
  `train.in` itself; whether the card should fix a tuning split for everyone is open.
