# runner — spec card

**What it is.** The general task contract ([`CONTRACT.md`](CONTRACT.md)) and the runner that applies it. Any task is
records in, records out and a checker; any candidate is a `train.py` and an `infer.c`, trained and run in
`loop.py`'s jail, metered with its timer and counter. It replaces the single-task `loop.py` for new work.
`loop.py` stays frozen and is reused by import.

**Status (2026-11-18): the contract held on its first four tasks** ([`results/2026-11-18/`](results/2026-11-18/README.md)):
- three label tasks (optdigits, next-character text, UCI Adult) and one verifier-scored task (planted 3-SAT, checked
  with no answer key) ran with no task-specific code in `run.py`;
- training reproduced byte for byte in the jail on each host;
- outputs matched a direct run on the same host.

Weights do **not** reproduce across hosts for these numpy trainers.

**Parts.**
- `run.py`: `evaluate` (dev; test only under a ledger plan) and `reproduce` (train twice, compare).
- `records.py`: the record format. `checkers/label.py`: the built-in label checker.
- `candidates/`: `majority` and `label-linear`, `label-mlp16` (any label task); `sat-zeros`, `sat-walksat`.
- `../tasks/`: `build.py` (all tasks, downloads pinned by sha256), one `card.json` per task, `sat/check.py`.
- `../.github/workflows/runner.yml`: builds each task, runs the tests, reproduces and scores on dev (manual; no secrets).

**Run locally** (Linux, gcc, valgrind, Python 3 with numpy; bubblewrap, or Landlock for the fallback jail):
```
python3 tasks/build.py /tmp/built --only sat
python3 runner/run.py evaluate --data /tmp/built --task sat --cand sat-zeros sat-walksat --out /tmp/sat.json
python3 -m unittest runner/test_runner.py
```

**Adding a task:** a `card.json` (input and output description, output bound, checker, baselines, floor, pinned data
digests), a builder in `tasks/build.py`, and a `check.py` if labels don't fit. If that needs a change to `run.py`, the
contract failed for that kind of task: record it.

**Limits.** Fixed output bound per task; one checker call per split; no per-item time limit; the SAT task is easy;
cross-host weight reproduction fails for float trainers.
