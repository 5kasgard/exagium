# counter — spec card

**What it is.** The second brick: it runs a command under `valgrind --tool=cachegrind` several times and records the instruction and simulated cache-miss counts. This is the deterministic half of meter tier 2 (timing is the other half). Every count is labelled **simulated**. Cachegrind replays the program on a model CPU and a model cache; it reads no hardware.

**Run.**
```
python3 count.py -- /abs/or/PATH/program args              # 3 reps, reference cache, record in ../runs/
python3 count.py --input net.py --input w.bin -- python3 net.py
python3 count.py --input cand -- ./cand 1000               # ./name must be an --input
python3 count.py --check ../runs/<record>.json [--input-dir DIR]
python3 -m unittest -v test_count                          # 23 tests, ~34 s
```
**Exit codes:**
- **0:** counted, every rep identical in counts and output, and no code found that can switch counting off. With `--check`: same counts, processes and output, still clean.
- **1:** reps differ in counts or output, or such code was found. With `--check`: counts or output differ, or the record was edited or is malformed.
- **2:** `--check` can't compare like for like: another program, input, interpreter, shared library, loader cache, valgrind build, architecture or cache model; or the recorded run itself wasn't a clean exact count.
- **3:** the command or valgrind failed, a process called `exec` (below), or anything unexpected happened. A failure is never reported as a verdict.

Needs Linux, Python 3 (stdlib only) and valgrind 3.22 or later. CI installs it with `apt install valgrind`.

## How a run is isolated
Each rep starts from a **fresh sandbox at a fixed path**, `/tmp/exagium-count`:
- an empty working directory holding only the declared `--input` files;
- an empty `HOME` and an empty `TMPDIR`.

The path is fixed because path strings themselves change counts (below). A lock file lets one counter at a time use it.

**The environment is fixed byte for byte.** The command sees only:
- `PATH=/usr/local/bin:/usr/bin:/bin`, plus `HOME` and `TMPDIR` in the sandbox;
- `LC_ALL=C`, `TZ=UTC`, `PYTHONHASHSEED=0`;
- one thread for OpenMP, OpenBLAS and MKL;
- anything added with `--env`. `PATH`, `HOME`, `TMPDIR` and anything starting `VALGRIND_`, `LD_`, `GCONV_PATH` or `GLIBC_TUNABLES` are refused, because they could swap the cachegrind tool or load unfingerprinted code (review 2).

**Valgrind:**
- `--command-line-only=yes`, so `.valgrindrc` files and `VALGRIND_OPTS` are ignored (tested).
- `--trace-children=yes`, with counts summed over every process.
- **One log on a pipe.** Every process logs to one pipe, on a file descriptor valgrind reserves. Unlike a file, a pipe can't be edited afterwards. Each process image writes a `Command:` header when it starts, and every process, forked children included, writes its `I refs` total when it ends. The counter requires exactly one count file per reported total, with the same `Ir`. A deleted count file, or one whose `Ir` was edited, is therefore a failure, and so is a process that outlives the command.
- **`exec` is refused, by the command or by any child.** Valgrind drops the counts of a process image when it calls `exec`, so work before an `exec` would vanish. Review 2 measured a ~1,500× undercount, and ~860× from a fork child that did the work and then exec'd. Only the command itself may start a process image, so any second `Command:` header is a failure.
  - **Allowed:** children made by `fork` alone, such as Python `os.fork` or multiprocessing with fork.
  - **Refused:** shells, `subprocess` spawns, `posix_spawn` and `#!/usr/bin/env` scripts. Run programs directly, e.g. `python3 script.py`.
- All three caches are always passed. Given only `--LL`, valgrind would take I1 and D1 from the host's CPUID.

**What's executed is fingerprinted.** sha256 is recorded for:
- the program, and its `#!` interpreter if it's a script;
- every shared library `ldd` resolves for them;
- every input;
- the host's `/etc/ld.so.cache`;
- the valgrind launcher, the cachegrind tool binary and `count.py` itself.

These files are also scanned for the **valgrind client-request preamble** (x86-64 and arm64). That code can switch counting off (`CACHEGRIND_STOP_INSTRUMENTATION`) or detect valgrind. If it's found, the run exits 1. A test shows the stop request really does hide about 3 million instructions, and that the scan flags it.

## What it records
One JSON file per run in `runs/`, opened with exclusive create, so a record is never overwritten. Each record holds:
- the command, environment, inputs and fingerprints;
- the valgrind version and options, and the host (CPU model, kernel, libc, loader-cache digest);
- the cache model requested and the one actually simulated (a mismatch is a failure);
- every rep's counts (`Ir`, `Dr`, `Dw`, and L1 and LL misses for instructions, reads and writes), its process count, wall time under valgrind, and the sha256 of its stdout;
- the median, min, max and relative spread of each count;
- the verdict: `exact`, `stdout_identical` and `client_request_code`;
- caveats, and a sha256 digest of the canonical record. The digest is an integrity checksum, not a seal, since anyone can recompute it.

## Reference cache (an explicit choice)
| Cache | Size | Ways | Line |
|---|---|---|---|
| I1 | 32 KiB | 8 | 64 B |
| D1 | 48 KiB | 12 | 64 B |
| LL | 8 MiB | 16 | 64 B |

This is a convention for comparing candidates across hosts. It doesn't model any one host. Why these sizes:
- **D1 48 KiB.** Machine C's VM reports 32 KiB, but the probe measures 1.55 ns at 44 KiB (`runs/probe_C_20261110.json`). Machine A's level 1 also covered 4–44 KiB (probe spec card).
- **LL 8 MiB.** On machine C, the curve sits at 25 ns up to 5.9 MiB and is DRAM-like at 8 MiB. Machine A's spec card gives its one-core last level as 8–16 MiB.
- **Shape.** Valgrind needs a power-of-two number of sets: 64 for D1, 8192 for LL.

`--cache host` uses CPUID instead. That gives a different model on every host, so its counts are only comparable on the same host.

## Hypothesis 1: are the counts reproducible?
**Within one host: yes, for deterministic programs.**
- **C** (a small MLP, `workloads/mlp.c`): every rep identical, in all 88 CI records and every run here.
- **Pure Python:** without a fixed hash seed, `Ir` varied by 0.3% between reps. With `PYTHONHASHSEED=0` (the default), every rep was identical.
- **NumPy (OpenBLAS):** reps still varied by 2–8 × 10⁻⁶ with one thread and a fixed hash seed, with or without ASLR. `cg_diff` places the difference in interpreter code (dict resizing, memcpy). The cause isn't traced.
- **ASLR** changed nothing for C or pure Python (`setarch -R` gave identical counts).
- These single-host numbers are reproducible with `experiments/h1_local/run.sh`; its output is `h1_local.txt` (made before the sandbox, so it toggles each factor by hand).

**Across hosts: the CPU never mattered.** Valgrind presents its own synthetic CPU to the program. Here it was an Intel signature `0x306c3` with AVX2 and no AVX-512, while the host has AVX-512. So code that dispatches on CPU features takes the same path on every AVX2 host.
- **Totals, same binary and image:** identical across six x86-64 CPU models. AMD: EPYC 7763, 9V74 and 9V45. Intel: Xeon 8573C, 8370C and 6973P-C. A seventh, Xeon 6975P-C, ran only once, on its own image (review 2).
- **arm64:** Neoverse N2, CPU part `0xd49`. Every arm64 job gave the same total.

**What moves the totals is software and setting, all at startup:**
- **The runner image:** for example, 36,018,926 vs 36,018,955 `Ir` for `./mlp 1000` on two images.
- **Path and environment strings:** the working directory changed Python's `Ir` by 7 × 10⁻⁵ to 1.3 × 10⁻⁴, and `HOME` changed C's `D1mr` by 2–3. That's why the sandbox path and environment are now fixed.
- **This workspace** (machine C, another kernel) gives 36,018,927: one instruction off the CI value.

**The program's own work is what reproduces.** Subtract a startup-only run (`n` = 0) from the full count. With the sandbox (CI run 41988500105 plus this workspace):

| Workload | Hosts | Count minus startup |
|---|---|---|
| C, x86-64 | 4 CI jobs, 3 images and 3 CPU models, plus machine C | 35,765,911 on every one |
| C, arm64 | 4 jobs, 2 images | 30,898,727 on every one |
| Python, x86-64 | 2 Azure images | 40,676,904 on both |
| Python, x86-64 | AWS image (another loader cache) | 40,701,411 (6 × 10⁻⁴ off) |
| Python, arm64 | 2 images with the same fingerprint | 41,774,558 vs 41,774,514 (1 × 10⁻⁶) |

**So:**
- **C:** counts are reproducible across hosts once startup is subtracted.
- **Python:** the interpreter reads image files that the fingerprint doesn't cover (inferred, not traced). Its counts are comparable only within one image.
- **`--check`** returns 2 (can't compare) when the loader cache differs, as with the AWS image. It can't detect a difference that leaves the loader cache unchanged, as with the arm64 pair.

**Where the clean environment is defeated:**
- **`python3 -I`** ignores every `PYTHON*` variable, including the hash seed. Every CI job then reported reps that differ (16 of 16), and the verdict flagged each one. My hypothesis that isolated mode would remove image dependence was refuted.
- **Any program that reads randomness, the time or the host** varies, and the verdict says so; the test `noisy.c` confirms it.

## Hypothesis 2: how far are simulated misses from the measured latency curve?
Machine C, probe run seed 20261110. The same Sattolo chase ran at the probe's 32 sizes, under cachegrind, with steady-state misses per load. Script: `experiments/sweep.py`.
- **Two levels, sharp edges.** The model has L1 and LL only, with LRU, so for a cyclic chase every size gives exactly 0 or 1 miss per load. Real latency rises in steps with mixed points.
- **Host model** (CPUID: D1 32 KiB, LL 34 MiB): the simulated level matches the measured one at 26 of 32 sizes.
  - At 44 KiB the model misses D1, but the hardware is still at L1 latency (1.55 ns).
  - At 8–33.5 MiB the model hits LL, but the hardware is at DRAM-like latency (94–161 ns).
- **Reference model:** 31 of 32 match. The only miss is exactly 8 MiB, which fits the model's LL but measures 161 ns.
- **L2 is invisible.** The measured L2 (64 KiB–1 MiB, 4.3 ns) and the 25 ns region at 1.4–5.9 MiB are both just "D1 miss, LL hit".
- **Machine A**, by arithmetic only, from its spec card: its VM reports a 260 MiB L3, so a host model would call everything up to 260 MiB an LL hit. The probe measured DRAM there from 16 MiB.

**So the counts are a deterministic work measure, relative to a stated model, not a latency predictor.** Turning counts into time or joules is calibration: tier 3, open item 5.

**Found along the way:** counting only a region (client requests with `--instr-at-start=no`) starts with a cold model cache, so every line misses once. The sweep removed this by differencing two lap counts.

## What `--check` can and can't show
**Before re-counting,** it requires that:
- the record's digest and internal consistency hold (malformed → 1, never a crash);
- every input, the program, interpreter and shared libraries, the loader cache, the valgrind version and the architecture match (otherwise 2).

It then re-counts in the fixed sandbox and compares counts, process count and stdout.
- **Catches** (tested): an edited record, and a self-consistent forged record with a recomputed digest, because re-counting gives different numbers.
- **Can't show** that a record was made on the host it names, or when. That needs the record and its outside anchor (open item 3).
- **Can't tell an image difference from a forgery** when the loader cache is the same (the arm64 Python case). Both exit 1.

## Review
**Review 1** (an independent Claude agent) found real breaks, all now fixed and tested:
1. **A cached answer.** A precomputed answer in the working directory let a candidate skip its work: about 87× undercount, `exact`, and `--check` exit 0. Fixed by the fresh sandbox.
2. **A false check failure.** `--check` used the checker's `HOME` and `PATH`, so a genuine record failed. Fixed by the fixed environment.
3. **Crashes read as verdicts.** A malformed record crashed with exit 1. Fixed: crashes now exit 3, and a malformed record is exit 1 with a message.
4. **Unchecked fingerprints.** Library hashes, loader cache, process count and output weren't compared. Fixed.
5. **Quiet undercounts.** Lost child count files could undercount silently. Fixed by matching each process's log against its count file.
6. **Loose parsing.** Missing events read as 0, and a cache-model mismatch only warned. Fixed.

It also corrected the arm64 CPU named in this card: part `0xd49` is Neoverse N2, not V2.

**Review 2** (a fresh independent agent, against the fixed code) confirmed fixes 1, 2 and 4, and found five new ways to get exit 0 with a wrong number. Fixed and tested:
1. **Work before `exec` is lost** (~1,500×, and `--check` agreed). Now detected and refused.
2. **`--env VALGRIND_LIB` swapped in a fake cachegrind** (Ir 1,000). `--env LD_PRELOAD` loaded unscanned code that switched counting off. Both are now refused, and the tool binaries are fingerprinted.
3. **`--check` ignored the client-request flag,** so a forged clean record passed. Now `--check` vouches only for clean, exact records and requires the re-count to be clean too.
4. **Output that differs between reps still exited 0.** Now it exits 1.
5. **Malformed fields with a recomputed digest crashed (exit 3).** Fields are now type-checked, and the re-check is guarded, so they exit 1. Shebangs must name an absolute interpreter.

It also corrected the CPU-model count above.

**Re-test of those fixes** (same agent): 3, 4, 5a, 6 and 7 confirmed fixed. Two attacks still got exit 0, because the log was then a file the program could edit:
- a program that execs itself and deletes its first header (Ir 152,783, `--check` exit 0);
- a child that deletes its count file and its lines in the log (Ir 155,778 vs ~250 M).

Both are now caught, because the log is a pipe (tests `test_a_child_deleting_its_counts_is_caught` and `test_work_before_exec_is_refused`). `#!/usr/bin/env` scripts are now refused with a clear message instead of an `exec` failure.

**Second re-test** (same agent): both caught. It found that a fork child could do the work and then exec, without detection: ~860× under, exit 0. That's fixed: any exec anywhere is now refused (`test_children_that_exec_are_refused`). It also showed a candidate can *append* lines to the log pipe through `/proc/self/fd`. That didn't produce an undercount, since true lines can't be removed and a duplicate total is a failure, but it's listed below.

## Trust model
Cachegrind runs inside the candidate's own process, as the same user. **Hostile native code can therefore always falsify its counts.** Known ways that still work:
- edit the cache-miss fields of a count file (only `Ir` is cross-checked against the pipe);
- have a forked child do the work and then have its parent kill it with `SIGKILL`, so it never reports (review 2: Ir 132,941, exit 0);
- append forged lines to the log pipe through `/proc/self/fd` (no undercount found this way yet);
- write into valgrind's memory;
- build code at run time that the client-request scan never sees. Review 2 decoded a stop-counting binary from a base64 input and ran it: Ir 1.07–1.31 M instead of ~250 M.

So the counts are trustworthy for code that isn't trying to cheat. Against code that might be, they are a claim, not proof. The cross-checks are:
- the timing half of tier 2;
- output checked against expected answers;
- independent re-measurement by someone else (item 4);
- running untrusted candidates in a fresh container, as another user, with the counter outside it. That is the next isolation step, and an inbox item.

A clean scan (`client_request_code: []`) means only that no such code was found in the files scanned. It is not evidence that counting stayed on.

## Known limits
- **Not a container.** A program can still use absolute paths outside the sandbox (a cache left in `/var/tmp`, say), other processes, or the network. Run untrusted candidates in a fresh container; CI does, since each job is a fresh VM. The verifier must also check the output against the expected answers.
- **Code loaded later isn't scanned:** `dlopen`, JIT, and code decoded at run time (see the trust model).
- **The synthetic CPU.** Counts are for the AVX2-or-lower path. On AVX-512 hosts that isn't the path that runs natively.
- **Threads.** Valgrind runs one thread at a time, and thread counts are pinned to 1.
- **Speed.** About 20–100× slower than native.
- **Platforms.** Tested on Linux x86-64 and arm64 with valgrind 3.22 only. No branch simulation.

## Files
- `count.py`: the counter.
- `test_count.py`: the tests.
- `workloads/`: `mlp.c` and `mlp.py`.
- `experiments/`: `chase_count.c`, `sweep.py` and the sweep results (H2); `h1_local/` (single-host H1 checks).
- `../runs/`: records from machine C.
- `../.github/workflows/counter.yml`: cross-host CI, 8 jobs per push, with records on the `ci-runs` branch.
