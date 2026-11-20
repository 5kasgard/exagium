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

Weights first did not reproduce across hosts: OpenBLAS chose its kernels per CPU. Since Q55 the jail pins one
kernel type on x86, and text weights now match byte for byte across two CPU families
([`results/2026-11-18-q10-q55/`](results/2026-11-18-q10-q55/README.md)).

**The jail (Q10):** `loop.py`'s jail, plus `jail.py` and `limits.c`:
- rlimits (8 GiB address space, 1 GiB per file, 256 open files, CPU seconds) and the seccomp filter on every candidate
  run, in both modes;
- in bubblewrap mode, the counter's valgrind run inside confine, `--disable-userns`, and a 2 GiB tmpfs;
- a probe that fails closed if any of these is missing.

**Parts.**
- `run.py`: `evaluate` (dev; test only under a ledger plan) and `reproduce` (train twice, compare).
- `jail.py`, `limits.c`: the hardened jail. `q55.py`: the cross-host weights test.
- `records.py`: the record format. `checkers/label.py`: the built-in label checker.
- `candidates/`: `majority`; `label-linear`, `label-mlp16`, `label-gbdt`, `label-knn`, `label-wisard`, `label-mlp2l` (any label task); `sat-zeros`, `sat-walksat`.
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
weights reproduce across x86 hosts only with the kernel pinned (ARM untested); in confine mode the counter's run has
the filter but not the rlimits.
