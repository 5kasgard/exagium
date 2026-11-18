# The task contract, first test (2026-11-18): it held on four kinds of task

What was tested is the hypothesis in [`../../CONTRACT.md`](../../CONTRACT.md): one contract (record files in and out,
a checker per task) runs labels and a verifier-scored task with no task-specific code in `run.py`, and the runner
path gives the same outputs as a direct run.

**Evidence:** Actions run 44253054973 → `ci/` (bubblewrap jail; optdigits on Xeon 6975P-C, the rest on EPYC 9V74).
Local runs in the fallback jail (confine) → `local/`. Each `dev-*.json` holds every item's output and score.

## Results
| task (dev items) | kind | checker | baseline / floor | linear | mlp16 | other |
|---|---|---|---|---|---|---|
| optdigits (956) | vision | label | majority 0.093 | 0.961 | 0.977 | |
| text (6,000) | language | label | majority 0.157 | 0.338 | 0.328 | |
| adult (6,513) | tabular | label | majority 0.754 | 0.852 | 0.859 | |
| sat (200) | verifier-scored | `check.py`, no key | all-false 0 solved | | | WalkSAT 200 of 200 solved |

Dev accuracy, or the share of instances solved, from `ci/dev-*.txt`. Instructions and native time per item are in
the same files.

## What held
- **No task code in the runner.** All four kinds ran through the same `run.py`; a test (`test_runner.py`) fails if
  `run.py` names any task.
- **A checker with no answer key works.** SAT's checker verifies assignments from the instance alone. WalkSAT solved
  every dev instance (about 0.13 s for all 200 on one CPU locally: 0.11 and 0.15 s in two rounds); all-false solved none, though it satisfies 87% of clauses
  (`ci/dev-sat.json`).
- **Training reproduces on one host.** All 11 candidate–task pairs retrained to identical bytes in the jail
  (`ci/reproduce-*.txt`).
- **Same outputs through the runner as directly.** A direct run means step 1's `transfer/families.py`, seed 1, on the
  same split, with no jail (`local/equivalence.txt`, outputs in `local/direct-*.pred`):
  - **on the same host,** the runner matched it on every item for optdigits and text, linear and mlp16 (0 of 956 and
    0 of 6,000 differ); adult was not run through the runner locally;
  - **across hosts,** it matched on every item for optdigits and adult, and for text linear. Text mlp16 differed on
    59 of 6,000 (below). Step 1's own CI run and this CI run also match on optdigits, 0 of 956, on different CPUs.

## What did not hold, or is still open
- **Weights do not reproduce across hosts.** Text mlp16 trained on CI and locally differ in 59 of 6,000 dev outputs,
  and even linear's weights differ in bytes (`ci/` against `local/dev-text.json`: weight sha256 285d74… against
  c847e7… for mlp16). One possible cause is numpy's BLAS kernels choosing different summation orders on different CPUs
  (hypothesis, not tested). The two hosts also differ in Python (3.12 against 3.13), kernel and jail mode (bwrap
  against confine), so any of these could matter. Optdigits weights were identical across the same two kinds of host. The test gate retrains "here", as `loop.py` does, so the gate stands. A claim that an
  artifact "is" a set of weights is host-bound for these trainers. lgn_d avoided this with deterministic custom ops.
- **The SAT task is easy:** planted instances at 4.2 clauses per variable, 50 variables. It shows the contract works.
  It is not a benchmark yet.
- **Not tested:** variable-length outputs larger than one record bound (proofs, programs), slow checkers (Lean), and
  per-item resource limits. These are the contract's open questions, for the mixed suite (P3).

## Reviews
- gpt-oss-120b (Cloudflare, free; router-runs `reviews/44253058538`) raised 4 critical and about 20 major points. Each
  was checked against the code, and none held:
  - the seccomp flag and timer parsing follow `loop.py`'s reviewed pattern; the parser takes the last timer marker,
    which the timer writes after the child exits;
  - the jail's `/tmp` is its own tmpfs, and the host root is not mounted;
  - WalkSAT writes `best` on its first step, and an instance with no clauses ends before any clause is read;
  - outputs are checked for being identical across rounds, so item-level SE is the right one.

  Its one fair point, per-item output bounds, is already an open question in the contract.
- A separate agent's check of these claims is recorded in the handoff.
