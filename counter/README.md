# counter — spec card

**What it is.** The second brick: it runs a command under `valgrind --tool=cachegrind` several times and records the instruction and simulated cache-miss counts. This is the deterministic half of meter tier 2 (timing is the other half). Every count is labelled **simulated**. Cachegrind replays the program on a model CPU and a model cache; it reads no hardware.

**Run.**
```
python3 count.py -- ./candidate args           # 3 reps, reference cache, record in ../runs/
python3 count.py --reps 5 --cache host -- python3 net.py
python3 count.py --check ../runs/<record>.json  # re-count here and compare
python3 -m unittest -v test_count               # 12 tests, ~13 s
```
Exit codes: 0 counted and every rep identical (`--check`: same counts); 1 reps differ (`--check`: counts differ, or the record was edited); 2 `--check` can't compare like for like (another program or input, valgrind version, architecture or cache model); 3 the command or valgrind failed.

Needs Linux, Python 3 (stdlib only) and valgrind 3.22 or later. CI installs it with `apt install valgrind`.

## What it records
One JSON file per run in `runs/`, opened with exclusive create, so a record is never overwritten. Each record holds:
- the command, working directory and the exact environment the command saw;
- sha256 of the program, of every shared library `ldd` resolves for it, and of any `--hash` input file;
- the valgrind version, the host (CPU model, kernel, libc, `/etc/ld.so.cache` digest), and the cache model requested and the one valgrind actually simulated;
- every rep's counts (`Ir`, `Dr`, `Dw`, L1 and LL misses for instructions, reads and writes), the process count, wall time under valgrind and the sha256 of stdout;
- the median, min, max and relative spread per event, and a verdict: `exact` (every rep identical) and `stdout_identical`;
- caveats, and a sha256 digest of the canonical record.

**How it keeps runs comparable:**
- **Clean environment.** The command sees only `PATH`, `HOME` and a fixed set: `LC_ALL=C`, `TZ=UTC`, `PYTHONHASHSEED=0`, and one thread for OpenMP, OpenBLAS and MKL. Use `--env` to add more.
- **No caller settings.** `--command-line-only=yes`, so `~/.valgrindrc`, `./.valgrindrc` and `VALGRIND_OPTS` can't change the cache model (tested).
- **Children are counted.** `--trace-children=yes`, so a candidate can't hide work in a subprocess; counts are summed over every process.
- **Explicit cache.** All three caches are always passed. Passing only `--LL` makes valgrind take I1 and D1 from the host's CPUID.

## Reference cache (an explicit choice)
| Cache | Size | Ways | Line |
|---|---|---|---|
| I1 | 32 KiB | 8 | 64 B |
| D1 | 48 KiB | 12 | 64 B |
| LL | 8 MiB | 16 | 64 B |

This is a convention for comparing candidates across hosts. It doesn't model any one host. Why these sizes:
- **D1 48 KiB.** Both measured machines hold 44 KiB at L1 latency. Machine C's VM *reports* 32 KiB, and the probe measures 1.55 ns at 44 KiB.
- **LL 8 MiB.** It sits between the effective one-core last level the probe measured on machine C (about 6–8 MiB) and on machine A (8–16 MiB).
- **Shape.** Valgrind needs a power-of-two number of sets: 64 for D1, 8192 for LL.

`--cache host` uses CPUID instead. That gives a different model on every host, so its counts are only comparable on the same host.

## Hypothesis 1: are the counts reproducible?
**Within one host: yes, for deterministic programs.**
- **C** (a small MLP, `workloads/mlp.c`): every rep identical, in every run here and in CI.
- **Pure Python:** without a fixed hash seed, `Ir` varied by 0.35% between reps. With `PYTHONHASHSEED=0` (the counter's default), every rep was identical.
- **NumPy (OpenBLAS):** reps still varied by 2–6 × 10⁻⁶ with one thread and a fixed hash seed. `cg_diff` places the difference in interpreter code (dict resizing, memcpy). Its source isn't proven; the process reads `getrandom` four times.
- **ASLR** changed nothing here (`setarch -R` gave identical counts).

**Across hosts: yes for the same software, whatever the CPU.**
- The same C binary (sha256 `a330b376…`, built by gcc 13.3 here and on the runners) gave `Ir` = 36,018,338 on all 16 x86-64 CI jobs. Those jobs ran on six CPU models: AMD EPYC 7763, 9V74 and 9V45, and three Intel Xeons.
- On 16 arm64 jobs (Neoverse V2) its arm64 build gave 31,067,197 every time.
- This workspace (machine C, another kernel and image) gave 36,018,339: one instruction off (3 × 10⁻⁸), and one D1 miss off.
- **Why the CPU doesn't matter:** valgrind presents its own synthetic CPU to the program. Here it was an Intel signature `0x306c3` with AVX2 and no AVX-512, while the host has AVX-512. So code that dispatches on CPU features takes the same path on every AVX2 host.

**What does move the counts is software and setting:**
- **Runner image.** Python counts split exactly by GitHub runner image, never by CPU. On x86, image 20261026.354 gave 104,492,328 and 20261101.360 gave 104,492,251 (7 × 10⁻⁷), on whatever CPU the job landed. Arm shows the same split. The `python3` binary and `ld.so.cache` were identical, so some other file in the image is read at startup (inferred, not traced).
- **Working directory.** It changed Python's `Ir` by up to 1.3 × 10⁻⁴ (`sys.path` holds it).
- **Environment strings** changed `D1mr` by 3 in 1,204, because the stack layout moved.
- **Startup-only differencing** (count `n` minus count `0`) shrank the image difference from 77 to 9 instructions, but didn't remove it.

**Where the clean environment can be defeated:**
- **`python3 -I`** ignores every `PYTHON*` variable, including `PYTHONHASHSEED`. Every CI job then reported reps that differ, and the verdict flagged it each time. This was my hypothesis for removing image dependence, and it was refuted.
- **Any program that reads randomness, the time or the host** varies, and the verdict says so; the test `noisy.c` confirms it.

## Hypothesis 2: how far are simulated misses from the measured latency curve?
Machine C, probe run seed 20261110 (`runs/probe_C_20261110.json`). It ran the same Sattolo chase at the probe's 32 sizes, under cachegrind, with steady-state misses per load. Script: `experiments/sweep.py`.
- **Two levels, sharp edges.** The model has L1 and LL only, with LRU, so for a cyclic chase every size gives exactly 0 or 1 miss per load. Real latency rises in steps with mixed points.
- **Host model** (CPUID: D1 32 KiB, LL 34 MiB): the simulated level matches the measured one at 26 of 32 sizes.
  - At 44 KiB the model misses D1, but the hardware is still at L1 latency (1.55 ns).
  - At 8–33.5 MiB the model hits LL, but the hardware is at DRAM-like latency (94–161 ns).
- **Reference model:** 31 of 32 match. The only miss is exactly 8 MiB, which fits the model's LL but measures 161 ns.
- **L2 is invisible.** The measured L2 (64 KiB–1 MiB, 4.3 ns) and the 25 ns region at 1.4–5.9 MiB are both just "D1 miss, LL hit".
- **Machine A**, by arithmetic only: its VM reports a 260 MiB L3, so a host model would call everything up to 260 MiB an LL hit. The probe measured DRAM from 16 MiB.

**So the counts are a deterministic work measure, relative to a stated model, not a latency predictor.** Turning counts into time or joules is calibration: tier 3, open item 5.

**Found along the way:** counting only a region (cachegrind client requests with `--instr-at-start=no`) starts with a cold model cache, so every line misses once. The sweep removed this by differencing two lap counts. The counter counts whole processes, so the issue doesn't arise there.

## What `--check` can and can't show
It refuses to compare (exit 2) unless the program and every hashed input match, and valgrind, the architecture and the simulated cache model are the same. It then re-runs the command in the recorded directory and environment and compares the counts.
- **Catches:** a record edited after it was written (digest mismatch), a summary or verdict that disagrees with its reps, and a self-consistent forged record with a new digest, because re-counting gives different numbers (tested).
- **Can't show:** that a record was made on the host it names, or when. It also can't distinguish a genuine difference in software (for example, another runner image) from a forgery: both exit 1. That needs the record and its outside anchor (open item 3).
- **On another image,** a Python record can fail by about 10⁻⁶ while a C record passes. Compare records only within one software fingerprint.

## Known limits
- **The synthetic CPU.** Counts are for the AVX2-or-lower path valgrind exposes. On AVX-512 hosts that isn't the path that runs natively.
- **Threads.** Valgrind runs one thread at a time, so threaded programs can depend on scheduling. Thread counts are pinned to 1 by default.
- **Speed.** About 20–100× slower than native.
- **Platforms.** Tested on Linux x86-64 and arm64 with valgrind 3.22 only. No macOS.
- **No branch simulation.** `--branch-sim` is off. It's an inbox item.

## Files
- `count.py`: the counter.
- `test_count.py`: the tests.
- `workloads/`: `mlp.c` and `mlp.py`, the test candidates.
- `experiments/`: `chase_count.c`, `sweep.py` and the three sweep results.
- `../runs/`: one record per run.
- `../.github/workflows/counter.yml`: cross-host CI. Its records go to the `ci-runs` branch.
