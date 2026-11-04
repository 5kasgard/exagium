# forge-probe — spec card

**What it is.** The Forge's first brick: a tool that measures a machine's invariants and writes `profile.json`. It records memory latency at each working-set size, the memory levels fitted to that curve, read bandwidth, compute ceilings, timer quality, host interference, and which meter tiers the machine offers. Every number is labelled **measured** (timed here), **reported** (the OS or CPU's claim) or **derived** (computed or fitted from measured numbers).

**Run.**
```
python3 probe.py                      # full: ~100 s on 2 vCPUs, 7 reps, sweep to 4x reported LLC (cap: 1/4 of free RAM)
python3 probe.py --quick              # ~10 s smoke run, 3 reps, coarser sweep
python3 probe.py --reanalyse p.json   # recompute everything from raw reps and check consistency; exit 1 on any problem
python3 -m unittest -v test_probe     # 24 tests, ~40 s
```
Needs a C compiler and Python 3 (stdlib only). Linux. Valgrind is optional and detected as a tier.

## What it measures
| Measurement | How | Bound by |
|---|---|---|
| Latency per working set | Dependent pointer chase over one random cycle of cache lines (Sattolo), √2 size steps from 4 KiB, new permutation every rep | load-to-use latency of whichever level holds the set |
| Read bandwidth | 8 independent 512-bit accumulators; L1-sized and DRAM-sized buffers; 1 thread and all threads | loads (L1) / memory system (DRAM) |
| Compute ceilings | 16 independent FMA chains: scalar f64, vector f64, vector f32; 1 and all threads | FMA issue rate |
| Timer | smallest observable step between reads, read cost, and the kernel's `clock_getres` claim | — |
| Host interference | CPU steal per rep from `/proc/stat` | — |
| Meter tiers | Each one is tried, not assumed: RAPL zone read; CPU PMU in sysfs and then `perf stat`; NVML library and then `nvidia-smi`; a cachegrind run | — |

**Noise.** Each rep runs every measurement once in a freshly shuffled order (the seed is recorded), each in a fresh process. Every figure carries n, median, bootstrap SE of the median, mean, SD, SE, CV, min and max. With a single rep, every spread figure is `null`, never a made-up value.

**Memory levels (derived).** These are fitted to the curve, and no cache sizes are assumed:
- **Fit.** A piecewise-constant fit of log latency, with each point weighted by its own noise.
- **Thresholds.** The owner's 2-SE rule, adjusted for small samples (Student-t, df = reps−1) and for breakpoints chosen from the data (Bonferroni over n−1 candidates).
- **What counts as a level.** It must span at least 2× in working-set size and be flat within noise.
- **Ramps.** If a single sloped line explains two neighbouring levels as well as a step does, they are reported as one *transition*.
- **Capacity brackets.** A bracket starts at the last size where no rep had yet reached the next level's speed.

Every doubt resolves toward "transition", never toward "level".

Simulated error rates (`test_probe.py`, 200 seeds, through the tool's own reps → bootstrap → fit path):

| Case | Phantom level | False boundary | Exact recovery |
|---|---|---|---|
| 4 levels, 7 reps, lognormal 3% noise | 0.5% | 0.5% | 91% |
| 4 levels, 7 reps, heavy-tailed (10% outliers at 1.5–3×) | 0% | 0% | 92% |
| 4 levels, 3 reps (`--quick`) | 0% | 0% | 92% |
| Flat curve, 12 / 34 points | 0% / 0% | 0.5% / 0% | 97% / 95% |

A gentle ramp is reported as two or more levels in 1.5–5.5% of runs, worst at 1.5% per step under 5% noise. That's a known limit. Whenever the level count was right, the values and brackets were right too (0 wrong).

**What `--reanalyse` can and can't show.** It recomputes every statistic from raw reps and checks the following:
- the rep counts match the settings;
- each rate equals its own recorded work ÷ time;
- each chase seed matches the run seed;
- on the same host, it re-detects the meters and compares.

It can't show that the raw data itself is genuine. That needs the anchored record (open item 3).

## First result (full run, this cloud VM, 2026-11-05, `profile.json`)
Intel Xeon (KVM guest, 2 vCPUs, AVX-512). CPU steal stayed at or below 1.3% per rep. Meter tiers: **timing and cachegrind only**. RAPL, a CPU PMU and NVML are all absent, so nothing here is energy.

| | Measured / derived | Reported by the VM |
|---|---|---|
| Level 1 (L1) | 1.81 ns, flat 4–44 KiB, capacity 44–64 KiB | 48 KiB |
| Level 2 (L2) | 5.71 ns, flat 64 KiB–1.4 MiB, capacity 1.4–2 MiB | 2 MiB |
| Level 3 (effective L3, one core) | 33.5 ns, flat 2–8 MiB, capacity **5.7–11.3 MiB** | **260 MiB** |
| Level 4 (DRAM) | 121 ns, flat 11.3 MiB–1 GiB | — |
| DRAM read, 1 / 2 threads | 15.7 / 30.1 GB/s | — |
| L1 read, 1 thread | 281 GB/s (CV 11%) | — |
| f64 scalar / f64 vector / f32 vector, 1 thread | 11.0 / 81.5 / 166 GFLOP/s | — |
| Ridge point (f64, all cores) | 4.9 flop/byte (derived) | — |
| Far ÷ near load latency | 67× (derived) | — |
| Timer | 29 ns smallest step (cost of a read); kernel claims 1 ns | — |

**What this already changes:**
- **The old probe measured its own loops, not the hardware.**
  - Its "3.3 GFLOP/s scalar" was limited by its loop. The real scalar FMA ceiling is 11 GFLOP/s.
  - Its "6.5 GB/s" came from one serial add chain. The real DRAM read rate is 15.7 GB/s.
  - The independent review confirmed both by rebuilding the old probe.
  - Anything tuned to the old numbers would have been tuned to the probe.
- **The VM's 260 MiB L3 isn't what one core gets.**
  - The 260 MiB is the host's figure. One core here holds L3 latency only up to about 6–11 MiB.
  - Optimising to the reported size would be optimising to a fiction.
- **The noise is real and doesn't come from steal.** All-thread figures vary by 7–20% CV even though steal stays at or below 1.3%. Comparisons must be interleaved and gated on SE, as decided.

## How it was checked
Two independent agent reviews ran: one before the fixes and one after the first round. Neither could break the kernels. Codegen, FLOP formula and ceilings were all confirmed against an imul-measured ~2.75 GHz clock. Both found real problems in the level fit and in `--reanalyse`, and all of them are fixed above.
- **First review:** phantom levels, staircases on slopes, fabricated averages on short or one-rep curves.
- **Second review:** its own noise generators showed phantom rates of 10–34%, because the noise estimate from few reps was too optimistic. It also found a `seconds=0` bypass and unchecked meter claims.

The commit history shows each step.

## Limits
- **Tier 2 only here.** These are times, not joules. The energy score needs item 5 (real-hardware calibration) or a machine where RAPL or NVML shows as available.
- **One run, one machine.** Noise figures cover this run only.
- **Latency at large sizes includes TLB effects.** The review saw huge pages applied during a run, and the THP mode is recorded. The host's own page size is unknown.
- **All-thread figures on 2 vCPUs are noisier** than single-thread ones.
- **Only x86 Linux has been tested.** The NVML path is untested (no GPU here). Where `-march=native` isn't supported the build falls back to plain `-O2`, which can lower the vector ceiling; the flags used are recorded.
- **Not measured yet:** write bandwidth, L2/L3 bandwidth, integer and low-precision throughput, synchronisation cost. Each is a candidate hypothesis, not a gap to fill by default.

## Files
`kernels.c` (measurements) · `probe.py` (driver, tiers, stats, levels, profile) · `test_probe.py` · `profile.json` (the run above, with raw reps) · `probe_v0_reference.c` (the old probe, kept for comparison)
