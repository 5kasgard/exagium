# loop — spec card

**What it is.** The fourth brick, and the first loop: small classifiers for one task, scored against two mandatory cheap baselines on the project's meter (tier 2: simulated instructions and native time). One agent proposes candidates and a pre-registered plan; the plan goes into the ledger before the test set is touched; a separate agent runs the test and judges it by the plan's own rule; the verdict goes into the ledger.

**Run.**
```
python3 loop.py freeze DIR...                      # digests for a plan's "inputs"
python3 loop.py reproduce DIR...                   # retrain in the jail; weights must be byte-identical
python3 loop.py evaluate DIR... --test dev         # smoke test (in-sample for anything trained on the whole file)
python3 loop.py evaluate DIR... --test test --plan SEQ [--rounds R] [--seed S]
python3 -m unittest -v test_loop                   # 17 tests, ~15 s
```
Exit codes: 0 done; 1 a check failed; 2 bad arguments (e.g. `--test test` without `--plan`; the docstring omits this); 3 failure. Plan 2 froze `loop.py`, `timer.c`, `counter/count.py`, `record/record.py`, the task and the candidates. `record.py` changed after entry 4 (a fix found by Z), so re-measure plan 2 at commit `113a3f6`, the last one with every frozen file unchanged. Needs Linux, gcc, valgrind 3.22+, bubblewrap with unprivileged user namespaces, Python 3 (stdlib; candidates' `train.py` may use `numpy==2.5.3`).

## The task
UCI optdigits (`task/SOURCE.md`): 8×8 images of 0..16 pixel counts, 3,823 training rows from 30 writers, 1,797 test rows from 13 other writers. The **dev** split is the last 25% of the training file. It is the holdout the baselines use to choose their L2 strength, so it's in-sample for anything refit on the whole file: a smoke test only.

## A candidate
A directory holding exactly `train.py`, `infer.c` and `weights.bin` (plus an optional README that its code never sees):
- `python3 -s -P train.py train.csv weights.bin` must be deterministic;
- `./infer weights.bin x.bin N` prints N digits and a newline; `x.bin` is N rows of 64 bytes. N = 0 is startup only;
- compiled with `-O2 -std=c11 -ffp-contract=off`, no `-march`, from a copy.

**Baselines** (`baselines/`): `linear`, logistic regression on pixels/16; `thermo`, thermometer encoding (16 bits per pixel) plus logistic regression, run as a 64×17×10 lookup table. Both pick L2 on dev, then refit on all rows.

## What a score is
For each candidate, on one machine, in one `evaluate`:
- **Accuracy** on the split, and **paired** differences against every other candidate, with SEs from per-image differences. Images are treated as independent; the test set has 13 writers, so true SEs are larger.
- **Simulated instructions per image:** `counter/count.py` (reference cache) counts *one* process classifying all N images, divided by N. Startup is included, so no work can escape the count by moving to N = 0. The N = 0 count is shown as "marginal", informational only.
- **Native ns per image:** the same one-process run, timed inside the jail by `timer.c` (pinned to one CPU). R rounds, candidates in a fresh random order each round, rows freshly shuffled each run. Predictions of every timed run and the counted run must agree, after undoing the shuffle.
- **Cross-checks:** time per simulated instruction more than 30× the lowest candidate's is flagged. The scoreboard JSON carries `valid`: false whenever any check raised a problem.

**The test set is gated.** `--test test` refuses unless the ledger verifies and its anchors check, entry SEQ is a plan whose `inputs` hold every digest `freeze` gives now (task files, `loop.py`, `timer.c`, `count.py`, `record.py`, every candidate file), and every candidate's training reproduces its weights byte for byte, in the jail, here.

## The jail
All candidate code (training, the counted run, the timed runs) runs in bubblewrap:
- a fresh tmpfs `/tmp`, read-only `/usr`, no network, its own PID and IPC namespaces;
- uid 65534 with every capability dropped, and a fresh session keyring;
- a working directory holding only that step's files. The repo, the test file and the frozen weights are not in it;
- files named like the dataset (`optdigits*`, `digits.csv*`) under `/usr` and the Python prefix are masked.

**One process, one thread.** The counter (`--single-process`) and the timer both read the jail's PID namespace `ns_last_pid` before and after a run, and refuse unless exactly one pid was handed out. That catches a child killed before it reports, an unreaped child and a thread alike. The jail can't write that file. The timer also requires the child to be the very next pid, so a pid wrap can't hide one.

## Review
Four rounds. Rounds 1 and 2 were by one independent adversarial agent; their attack scripts are in the handoff bundle (`review-loop/`).
1. **Round 1** found: work moved into the N = 0 run; `/tmp` caches across runs; memoising passes; reading labels through `__FILE__` and the repo; `reproduce` reporting identical falsely; training reading the test file; name collisions; unfrozen inputs; a local plan passing the gate; dev being in-sample. Fixed by the jail, the startup-included score, the trusted timer, unique names, and the gate.
2. **Round 2** found: a session-keyring cache; unmounting the dataset masks (root in the user namespace); ptracing or killing the counter; threads unpinning the timer; a fork child killed before it reports. Fixed by uid 65534 with no capabilities, a fresh keyring per jail, the counter non-dumpable inside the jail with its record on stdout only, CPU time against wall time, and the time/Ir band.
3. **Round 3** was cut off by a safety filter after reading the code. From its reading I demonstrated a light fork-kill: 149 Ir/img reported against an honest 3,949, and no flag, because native time is mostly startup. Fixed by the PID-namespace rule above; every fork, thread and kill attack is now refused.
4. **Round 4** was also cut off while writing attacks, then finished as a code-only review. It confirmed the PID rule, the output parsing, the label isolation and the maths. Its findings are fixed or now stated as limits: a `valid` flag, the next-pid rule, unfingerprinted `/usr/lib/debug` libraries, and the two limits below. Its suggested test of writing into the jail's stdout through `/proc/1/fd` fails closed (tested).

Attack testing in rounds 3 and 4 is thinner than in rounds 1 and 2.

## Trust model and limits
A verifier must cover these by reading the candidates' code:
- **The test labels are public.** scikit-learn's `load_digits()` is the test file. A `train.py` can carry the labels in as data and still reproduce byte for byte.
- **Counts are a claim for cooperative code.** Valgrind is always detectable from inside (its `LD_PRELOAD`, `/proc/self/maps`, its CPU), so a candidate could take a cheaper path when counted than when timed. Predictions are cross-checked between the runs; the work done isn't. Code that dispatches on CPU features (glibc's own `memcpy` included) is counted on its AVX2-or-lower path.
- **The time/Ir band** is relative to the lowest candidate and wide (30×). It catches gross hiding only.

Other limits:
- **The jail needs a kernel that lets an unprivileged user namespace mount `/proc`.** Z's kata container (kernel 5.10, with read-only `/proc` submounts) refuses it ("mount too revealing"), so `loop.py` fails closed there: 11 failures and 2 errors in `test_loop`, all "bwrap cannot make a jail". Z re-measured plan 2 without the jail, after reading the code, and matched every accuracy and prediction exactly (`verifications/z-20261106/`).
- **Tier 2 only.** Instructions and nanoseconds, not joules; native time is dominated by process startup at this task's size.
- **One task, one machine family per run.** Ir differs between x86-64 and arm64, so every Ir claim is per architecture.
- **SEs assume independent images.**
- **Size:** `loop.py` 486 lines, tests 224, `timer.c` 69, `loop.yml` 84.

## Plan 2: the first claim
**Outcome: inconclusive** (ledger entry 3, by `claude-loop1-verifier`; evidence in `results/plan2/`).

A proposer agent built three MLPs, 64-H-10 with ReLU, run with layer 1 as int16 lookup tables, and registered two claims (entry 2) before the test set was touched:
- **A:** mlp16 needs less than half the baselines' instructions per image, beats both on Ir per correct answer past 2 SE, and stays within 0.5 points of the better baseline's accuracy.
- **B:** mlp32 or mlp64 is more accurate than *both* baselines past 2 SE, and beats both on Ir per correct answer past 2 SE.

The verifier ran the test command here, and CI ran it on 3 x86-64 and 3 arm64 hosts. Every scoreboard is valid, every candidate retrained byte-identically, and predictions are identical on all 7 hosts.

| | Test acc | Ir/img x86-64 | Ir/img arm64 |
|---|---|---|---|
| linear | 0.9505 | 3,888 | 3,142 |
| thermo | 0.9599 | 3,630 | 2,886 |
| mlp16 | 0.9594 | 720 | 835 |
| mlp32 | 0.9655 | 1,025 | 1,234 |
| mlp64 | 0.9610 | 1,684 | 2,049 |

- **Part A passes on every host.** mlp16 uses 0.19–0.29× the baselines' instructions, and its accuracy (0.9594) is 0.06 points below thermo's, inside the 0.5-point floor.
- **Part B is inconclusive.** Both MLPs beat linear clearly (z 4.3 and 3.5), and both are cheaper per correct answer. But their lead over thermo is +0.56 points (z 1.62) for mlp32 and +0.11 (z 0.33) for mlp64, short of 2 SE. On the validation holdout it had been about +1.1 points: the edge shrank on the 13 new writers.
- **Code reading** (rule amendment 2): no test data carried in, and no environment-dependent path, in any of the five.

**So the plan as a whole wasn't earned; part A was.** At thermo's accuracy, mlp16 does the same job with about a fifth (x86-64) to a quarter (arm64) of the simulated instructions, *startup included*. Natively it is also 17–31% faster than both baselines on every host (for example 520 ± 5 ns/img against 706 and 711 on an EPYC 7763), although startup is most of every candidate's time at this size. Native time wasn't part of the rule. On this task, a bigger MLP's accuracy edge over thermometer-plus-linear is within noise. A re-run would need a new plan.

## Files
- `loop.py`: freeze, reproduce, evaluate; `timer.c`: the trusted timer; `test_loop.py`: the tests.
- `task/`: the dataset and its source. `baselines/`, `candidates/`: the five entries of plan 2.
- `verify/`: CI requests to score a plan on the test set, removed once the plan has a verdict (empty now). `results/plan2/`: the verifier's evidence.
- `../.github/workflows/loop.yml`: reproduce and score on 3 x86-64 and 3 arm64 runners, results to the `ci-runs` branch.
