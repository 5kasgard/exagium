# loop — spec card

**What it is.** The fourth brick, and the first loop: small classifiers for one task, scored against two mandatory cheap baselines on the project's meter (tier 2: simulated instructions and native time). One agent proposes candidates and a pre-registered plan; the plan goes into the ledger before the test set is touched; a separate agent runs the test and judges it by the plan's own rule; the verdict goes into the ledger.

**Run.**
```
python3 loop.py freeze DIR...                      # digests for a plan's "inputs"
python3 loop.py reproduce DIR...                   # retrain in the jail; weights must be byte-identical
python3 loop.py evaluate DIR... --test dev         # smoke test (in-sample for anything trained on the whole file)
python3 loop.py evaluate DIR... --test test --plan SEQ [--rounds R] [--seed S]
python3 -m unittest -v test_loop                   # 18 tests, ~15 s
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
- **Meter disagreements** (`cost_disagreements`, printed as `METERS DISAGREE`): every pair where one candidate uses fewer simulated instructions yet is slower natively past 2 SE (paired, interleaved rounds). These aren't problems with the run; they mark where an instruction-count cost claim alone would be wrong.

**The test set is gated.** `--test test` refuses unless the ledger verifies and its anchors check, entry SEQ is a plan whose `inputs` hold every digest `freeze` gives now (task files, `loop.py`, `timer.c`, `count.py`, `record.py`, every candidate file), and every candidate's training reproduces its weights byte for byte, in the jail, here.

## The jail
All candidate code (training, the counted run, the timed runs) runs in bubblewrap:
- a fresh tmpfs `/tmp`, read-only `/usr`, no network, its own PID and IPC namespaces;
- uid 65534 with every capability dropped, and a fresh session keyring;
- a working directory holding only that step's files. The repo, the test file and the frozen weights are not in it;
- files named like the dataset (`optdigits*`, `digits.csv*`) under `/usr` and the Python prefix are masked.

**One process, one thread.** The counter (`--single-process`) and the timer both read the jail's PID namespace `ns_last_pid` before and after a run, and refuse unless exactly one pid was handed out. That catches a child killed before it reports, an unreaped child and a thread alike. The jail can't write that file. The timer also requires the child to be the very next pid, so a pid wrap can't hide one.

## The fallback jail (when bubblewrap can't run)
Stock Docker, Kubernetes and some VMs refuse unprivileged user namespaces, so bubblewrap can't build its jail. `loop.py` then falls back to **confine** (`confine.c`, `seccomp_filter.h`), which needs no privilege at all. The mode is chosen at start-up by a self-test (bubblewrap first), recorded in every output as `"jail"`, and can be forced with `EXAGIUM_JAIL=bwrap|confine`.
- **Files (Landlock ABI 4+, Linux 6.7+; `confine` refuses older ABIs, which leave truncate or TCP outside Landlock):** reads are allowed only on a list carved around the dataset copies. A directory on the way to a hidden file can be listed but not read through; a symlink is granted only if its real path doesn't lead to a hidden file (site-packages is a symlink on Debian and Ubuntu). Writes go only to the step's own directories (under the counter, also its sandbox and count files, which it cross-checks). No TCP.
- **Everything else (seccomp):** no fork, clone, sockets, keyring, ptrace, signals to other processes, namespaces, mounts, io_uring, BPF or perf. No SysV IPC or POSIX message queues, and no chmod, chown, utimes or xattr writes, because those outlive a run and Landlock doesn't cover them. Rlimits, CPU affinity, priority and scheduling only on the process itself: the harness shares its uid and pid namespace, so a nice value set on `loop.py` would slow every later timed run. An outside reviewer found the IPC hole independently, our own review the metadata hole, and a separate fact-check agent the rlimit and priority hole (its test fails on the old filter).
- **The meters, almost unchanged.** The timer runs under Landlock only, so Landlock's setup (about 630 rules) isn't timed; the seccomp filter is installed in the timed child just before `exec`, a small cost bubblewrap runs don't pay (not yet measured). The counter wraps valgrind in confine with the same lists, plus `/proc/self` and `/proc/cpuinfo` (valgrind needs them). bubblewrap's one-pid check is replaced by seccomp refusing every way to make a process or thread.

**Results** (`test_loop`, 20 tests: the attacks from rounds 1, 2 and 4 replayed, plus network, keyring, fork, IPC, metadata, rlimit, priority, affinity and scheduler attempts; ptrace, namespaces, mounts, io_uring, BPF and perf are denied but not separately tested):
- **H1, it refuses what bubblewrap refuses:** holds. Every test passes in both modes, here and in stock `ubuntu:24.04` Docker on x86-64 and arm64 (the `docker` job in `loop.yml`).
- **H2, same programs, same counts:** holds on the same host: Ir/img is identical in both modes. Between Docker's image and the runner's, only process startup differs (−413 Ir per run on x86-64, −175 on arm64), so compare Ir across hosts with the same OS image.
- **H3, it works where bubblewrap can't:** holds in stock Docker. Not on Z's kata sandbox: its kernel (5.10) predates Landlock, so neither jail runs there and `loop.py` fails closed.

**Limits of the fallback** (bubblewrap has none of these):
- `stat()` still reveals that a hidden file *exists*; only reading it is refused. Dataset copies are masked by name, as under bubblewrap.
- Candidate code runs as the harness's own uid, not 65534 (as root in CI's Docker job; Landlock and seccomp bind root too). It can't write or signal outside its directories, but it can read whatever the allow-list grants.
- Reading a file can update its access time, a weak channel that `relatime` mounts mostly close.
- A hostile candidate that stalls (for example, one that retries fork) costs time until the step's timeout. That is a slow failure, never a better score.

## Review
Four rounds. Rounds 1 and 2 were by one independent adversarial agent; their attack scripts are in the handoff bundle (`review-loop/`).
1. **Round 1** found: work moved into the N = 0 run; `/tmp` caches across runs; memoising passes; reading labels through `__FILE__` and the repo; `reproduce` reporting identical falsely; training reading the test file; name collisions; unfrozen inputs; a local plan passing the gate; dev being in-sample. Fixed by the jail, the startup-included score, the trusted timer, unique names, and the gate.
2. **Round 2** found: a session-keyring cache; unmounting the dataset masks (root in the user namespace); ptracing or killing the counter; threads unpinning the timer; a fork child killed before it reports. Fixed by uid 65534 with no capabilities, a fresh keyring per jail, the counter non-dumpable inside the jail with its record on stdout only, CPU time against wall time, and the time/Ir band.
3. **Round 3** was cut off by a safety filter after reading the code. From its reading I demonstrated a light fork-kill: 149 Ir/img reported against an honest 3,949, and no flag, because native time is mostly startup. Fixed by the PID-namespace rule above; every fork, thread and kill attack is now refused.
4. **Round 4** was also cut off while writing attacks, then finished as a code-only review. It confirmed the PID rule, the output parsing, the label isolation and the maths. Its findings are fixed or now stated as limits: a `valid` flag, the next-pid rule, unfingerprinted `/usr/lib/debug` libraries, and the two limits below. Its suggested test of writing into the jail's stdout through `/proc/1/fd` fails closed (tested).

Attack testing in rounds 3 and 4 is thinner than in rounds 1 and 2.

## Two meters, and why both count
Simulated instructions are a deterministic work count; they are not a cost. Plan 2's own data shows the gap. Native time per simulated instruction, startup removed, from the committed scoreboards:

| | EPYC 7763 | arm64 |
|---|---|---|
| linear | 0.089 ns | 0.077 ns |
| thermo | 0.094 ns | 0.075 ns |
| mlp16 | 0.224 ns | 0.145 ns |
| mlp32 | 0.228 ns | 0.143 ns |
| mlp64 | 0.214 ns | 0.145 ns |

An MLP instruction took 1.9–2.6× as long as a baseline one. mlp64 used 0.43× linear's instructions on x86-64 (0.65× on arm64), yet it was slower than linear or thermo, past 2 SE, on all 7 hosts that scored plan 2.

**What explains it** (tested 2026-11-11 on the same data, after the fact, so hypotheses only):
- **Cache misses: not enough.** Adding simulated L1 misses to instructions cuts the error of a per-host fitted time model (5 candidates, so easily overfitted) from 30–46% to 22–32%. mlp16 has almost no misses (2.4 per image) yet the same high time per instruction.
- **Branch mispredictions: not for mlp16.** Cachegrind's branch simulator gives mlp16 3.7 mispredictions per image against linear's 4.1. mlp64 has 14.6, which may explain part of its gap.
- **Instructions per nanosecond: the leading hypothesis.** On the EPYC, linear retires about 11 simulated instructions per ns, the MLPs about 4.4–4.7 (the clock speed isn't recorded, so this isn't stated per cycle). A plausible cause is chains of dependent table lookups. Neither instruction counts nor cachegrind can see this; hardware counters (`perf`) can, and no host so far exposes them. Testing it is part of calibration on real hardware.

**The rule for every plan from now on.** A cost claim of candidate X over baseline B must pass both meters:
1. **Simulated instructions,** per architecture: Ir_X / (acc_X − 2·SE(acc_X)) < Ir_B / acc_B. This is plan 2's form; it lets accuracy noise count only against the candidate.
2. **Native time,** on every host that scores the plan, each judged on its own: X's paired `ns_per_image_diff` against B is below −2 × its SE. There's no averaging across hosts, and a host where the difference is under 2 SE fails the claim.

If one meter passes and the other doesn't, the claim is inconclusive; `cost_disagreements` lists the pairs where they point opposite ways. Plan 2 predates this rule and was judged by its own.

**A phantom meter.** This workspace's VM offers a `power/energy-psys` counter, which opens without error but reads exactly 0 J idle and under load. A meter that is present but dead must be reported as unavailable, never as zero joules.

## Trust model and limits
A verifier must cover these by reading the candidates' code:
- **The test labels are public.** scikit-learn's `load_digits()` is the test file. A `train.py` can carry the labels in as data and still reproduce byte for byte.
- **Counts are a claim for cooperative code.** Valgrind is always detectable from inside (its `LD_PRELOAD`, `/proc/self/maps`, its CPU), so a candidate could take a cheaper path when counted than when timed. Predictions are cross-checked between the runs; the work done isn't. Code that dispatches on CPU features (glibc's own `memcpy` included) is counted on its AVX2-or-lower path.
- **The time/Ir band** is relative to the lowest candidate and wide (30×). It catches gross hiding only.

Other limits:
- **The bubblewrap jail needs a kernel that lets an unprivileged user namespace mount `/proc`.** Z's kata container (kernel 5.10) refuses it. The fallback above needs Landlock ABI 4 (Linux 6.7+) instead; with neither, `loop.py` fails closed. Z re-measured plan 2 without the jail, after reading the code, and matched every accuracy and prediction exactly (`verifications/z-20261106/`).
- **Tier 2 only.** Instructions and nanoseconds, not joules; native time is dominated by process startup at this task's size.
- **One task, one machine family per run.** Ir differs between x86-64 and arm64, so every Ir claim is per architecture.
- **SEs assume independent images.**
- **Size:** `loop.py` 666 lines, tests 331, `timer.c` 76, `confine.c` 99, `seccomp_filter.h` 113, `loop.yml` 133.

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

**So the plan as a whole wasn't earned; part A was.** At thermo's accuracy, mlp16 does the same job with about a fifth (x86-64) to a quarter (arm64) of the simulated instructions, *startup included*. Natively it is also 17–31% faster than both baselines on every host (for example 520 ± 5 ns/img against 706 and 711 on an EPYC 7763), although startup is most of every candidate's time at this size; with startup removed it is 1.8–2.6× faster on the six CI hosts (2.3× on the EPYC cited), not the 3.5–5.4× its instruction count suggests. mlp64 is the warning: scored at 0.43× linear's instructions, it was slower than at least one baseline on all 7 hosts (see "Two meters" above). Native time wasn't part of the rule. On this task, a bigger MLP's accuracy edge over thermometer-plus-linear is within noise. A re-run would need a new plan.

## Files
- `loop.py`: freeze, reproduce, evaluate; `timer.c`: the trusted timer; `confine.c` and `seccomp_filter.h`: the fallback jail; `test_loop.py`: the tests.
- `task/`: the dataset and its source. `baselines/`, `candidates/`: the five entries of plan 2.
- `verify/`: CI requests to score a plan on the test set, removed once the plan has a verdict (empty now). `results/plan2/`: the verifier's evidence.
- `../.github/workflows/loop.yml`: reproduce and score on 3 x86-64 and 3 arm64 runners, plus the fallback jail in stock Docker on both; results to the `ci-runs` branch.
