# forge-probe — spec card

**What it is.** The Forge's first brick: a tool that measures a machine's invariants and writes `profile.json`. It records memory latency at each working-set size, the memory levels fitted to that curve, read bandwidth, compute ceilings, timer quality, host interference, and which meter tiers the machine offers. Every number is labelled **measured** (timed here), **reported** (the OS or CPU's claim) or **derived** (computed or fitted from measured numbers).

**Run.**
```
python3 probe.py                      # full: ~100 s on 2 vCPUs, 7 reps, sweep to 4x reported LLC (cap: 1/4 of free RAM)
python3 probe.py --quick              # ~10 s smoke run, 3 reps, coarser sweep
python3 probe.py --reanalyse p.json   # recompute from raw reps; exit 0 consistent, 1 inconsistent,
                                      #   2 consistent but meter claims unverifiable (different host)
python3 -m unittest -v test_probe     # 28 tests, ~40 s
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
| Meter tiers | Each one is tried, not assumed: RAPL zone read; CPU PMU in sysfs and then `perf stat`; NVML library and then `nvidia-smi`; a cachegrind run; tier-3 calibration file | — |

**Noise.** Each rep runs every measurement once in a freshly shuffled order (the seed is recorded), each in a fresh process. Every figure carries n, median, bootstrap SE of the median, mean, SD, SE, CV, min and max. With a single rep, every spread figure is `null`, and no levels are fitted.

**Memory levels (derived).** No cache sizes are assumed.
- **Fit.** A piecewise-linear fit of log latency on log size, with each point weighted by its own noise, widened to Student-t tails because it comes from few reps. Real plateaus drift a little, so a plateau is one sloped segment.
- **What counts as a level:**
  - It spans at least 2× in working-set size.
  - It rises significantly less than half as steeply as the edge to each neighbour. The edge slope is each segment's line evaluated at its own edge point, divided by the gap between those points. "Significantly" uses the owner's 2-SE rule, Bonferroni-corrected because the breakpoints come from the data.
  - A lone segment must show no detectable slope.
- **Ramps.** If one line explains two neighbouring levels as well as a step does, they are reported as a *transition*.
- **Capacity brackets.** A bracket starts at the last size where no rep had yet reached the next level, and ends at the first size where no rep was still at this level.
- **Explicit choices.** The 2× span and the ½ slope ratio are recorded in the profile.

Every doubt resolves toward "transition", never toward "level".

**Simulated error rates.** 200 seeds, run through the tool's own reps → bootstrap → fit path. The third reviewer's noise generators were used: lognormal 3%, and heavy-tailed (10% outliers at 1.5–3×). The table shows my runs; the reviewer's independent runs of the previous fitter were within a few points.

| Case | Phantom level | False boundary | Wrong value/bracket | Exact recovery |
|---|---|---|---|---|
| 4 levels, 7 reps, lognormal | 0% | 0% | 0 | 97% |
| 4 levels, 7 reps, heavy-tailed | 0% | 0% | 0 | 99% |
| 4 levels, 3 reps, heavy-tailed | 0% | 0% | 0 | 95.5% |
| 4 levels, 3 reps, `--quick` sweep | 0% | 0% | 0 | 98% |
| Flat curve, 12 / 34 points | 0% / 1% | 0% / 1% | 0 | 96% / 92.5% |
| Ramps (1–3%/step, 2–5% noise, 3–7 reps) | 1–2.5% any level | 1–1.5% | — | — |

- **Misses** are conservative: a real level is reported as a transition.
- **A designed-in false positive.** The reviewer built a curve with no flat region, whose slope alternates between 2% and 8% per step. On it, the previous fitter reported a "level" in 26% of runs, because a stretch rising at under half its neighbours' slope is, by this definition, a plateau. If that ever matters, the ½ ratio is the knob.

**What `--reanalyse` can and can't show.** It recomputes every statistic from raw reps and checks the following:
- the rep counts match the settings;
- each rate equals its own recorded work ÷ time, and `seconds` must be positive;
- each chase seed matches the run seed;
- on the same host, it re-detects every meter tier, including the calibration flag.

On another host it exits 2, because meter claims can't be re-checked there. It can never show that the raw data is genuine: that needs the anchored record (open item 3). Timer readings can't be checked against anything either.

## First result (full run, this cloud VM, 2026-11-05, `profile.json`, seed 20261106)
Intel Xeon (KVM guest, 2 vCPUs, AVX-512). CPU steal stayed at or below 0.9% per rep. Meter tiers: **timing and cachegrind only**. RAPL, a CPU PMU and NVML are all absent, so nothing here is energy.

| | Measured / derived | Reported by the VM |
|---|---|---|
| Level 1 (L1) | 1.84 ns over 4–44 KiB, capacity 24–64 KiB | 48 KiB |
| Level 2 (L2) | 5.86 ns over 64 KiB–1.4 MiB, capacity 1–2 MiB | 2 MiB |
| Level 3 (effective L3, one core) | 33.6 ns over 2–8 MiB, capacity **5.7–16 MiB** | **260 MiB** |
| Level 4 (DRAM) | 119 ns over 11.3 MiB–1 GiB | — |
| DRAM read, 1 / 2 threads | 15.7 / 30.2 GB/s | — |
| L1 read, 1 thread | 305 GB/s | — |
| f64 scalar / f64 vector / f32 vector, 1 thread | 10.9 / 79.9 / 155 GFLOP/s | — |
| Ridge point (f64, all cores) | 5.2 flop/byte (derived) | — |
| Far ÷ near load latency | 65× (derived) | — |
| Timer | 29 ns smallest step (cost of a read); kernel claims 1 ns | — |

**Three full runs agree.** Refitting the raw reps of all three kept runs with the final fitter (runs in commits `a69c20a` and `f2c1c1c` with seed 20261105, and this one with seed 20261106) gives:

| Run | L1 | L2 | L3 | DRAM |
|---|---|---|---|---|
| `a69c20a` | 1.8 ns | 5.7 ns | 33.5 ns, capacity 5.7–16 MiB | 121 ns |
| `f2c1c1c` | 1.8 ns | 5.7 ns | 32.9 ns, capacity 5.7–16 MiB | 118 ns |
| this run | 1.8 ns | 5.9 ns | 33.6 ns, capacity 5.7–16 MiB | 119 ns |

In the `a69c20a` run, one mixed point at 8–11 MiB is a transition. An earlier version of this card claimed L3 was unstable between runs. That was wrong: the claim rested on a run whose data I had overwritten, and the fitter at the time was what missed L3. The fourth review caught it.

**What this already changes:**
- **The old probe measured its own loops, not the hardware.**
  - Its "3.3 GFLOP/s scalar" came from a store-and-reload loop. The real scalar FMA ceiling is 11 GFLOP/s.
  - Its "6.5 GB/s" came from one serial add chain. The real DRAM read rate is about 16 GB/s.
  - The first independent review reported confirming both by rebuilding the old probe.
- **The VM's 260 MiB L3 isn't what one core gets.** One core holds L3 latency only up to about 6–16 MiB. Optimising to the reported size would be optimising to a fiction.
- **The noise is real and doesn't come from steal.** All-thread figures reach 7–20% CV, while steal stays at or below 1.3% in every run. Comparisons must be interleaved and gated on SE, as decided.

## How it was checked
Four independent agent reviews ran, and each found real problems in the level fit or `--reanalyse` that are now fixed. None could break the kernels. The first review reported checking the codegen, FLOP formula and ceilings against an imul-measured ~2.75 GHz clock; that check exists only in its report, not in this repo.
- **Review 1:** phantom levels, staircases on slopes, fabricated averages on short or one-rep curves, an unlabelled fitted output.
- **Review 2:** with its own noise generators, phantom rates were 10–34%. The noise estimate from few reps was too optimistic, because my simulation had fed in the true noise. It also found a `seconds=0` bypass and unchecked meter claims.
- **Review 3:** ramps still produced levels 55% of the time, upper brackets landed too early, and a changed host ID let a faked meter claim pass with exit 0.
- **A real run then exposed a design error the simulations had hidden.** Real plateaus drift, and precise data made every level fail an absolute-flatness test. That led to the piecewise-linear fit.
- **Review 4:** that fitter still missed L3 on real data, and my README claim about it was false. The edge slope was being diluted by DRAM's huge span, so it is now measured at the boundary.

The commit history shows each step.

## Limits
- **Tier 2 only here.** These are times, not joules. The energy score needs item 5 (real-hardware calibration) or a machine where RAPL or NVML shows as available.
- **Three full runs, one machine.** Two runs shared a seed, so they used the same chase permutations. Noise figures cover one run each.
- **Latency at large sizes includes TLB effects.** The first review saw huge pages applied during a run, and the THP mode is recorded.
- **Only x86 Linux has been tested.** The NVML path is untested (no GPU here). Where `-march=native` isn't supported the build falls back to plain `-O2`; the flags used are recorded.
- **Complexity debt.** The tool is ~1,300 lines in total, mostly the level-fit statistics and their tests, against ~30 lines for the old probe. The fit earned its place by catching false levels, but it's the part to keep an eye on.
- **Not measured yet:** write bandwidth, L2/L3 bandwidth, integer and low-precision throughput, synchronisation cost. Each is a candidate hypothesis, not a gap to fill by default.

## Files
`kernels.c` (measurements) · `probe.py` (driver, tiers, stats, levels, profile) · `test_probe.py` · `profile.json` (the run above, with raw reps) · `probe_v0_reference.c` (the old probe, kept for comparison)
