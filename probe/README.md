# forge-probe — spec card

**What it is.** The first brick: a tool that measures a machine's invariants and writes `profile.json`. It records memory latency at each working-set size, the memory levels fitted to that curve, read bandwidth, compute ceilings, timer quality, host interference, and which meter tiers the machine offers. Every number is labelled **measured** (timed here), **reported** (the OS or CPU's claim) or **derived** (computed or fitted from measured numbers).

**Run.**
```
python3 probe.py                      # full: ~100 s on 2 vCPUs, 7 reps, sweep to 4x reported LLC (cap: 1/4 of free RAM)
python3 probe.py --quick              # ~10 s smoke run, 3 reps, coarser sweep
python3 probe.py --reanalyse p.json   # recompute from raw reps; exit 0 consistent, 1 inconsistent,
                                      #   2 consistent but meter claims unverifiable (different host)
python3 -m unittest -v test_probe     # 34 tests, ~45 s
```
Needs a C compiler and Python 3 (stdlib only). Linux. Valgrind is optional and detected as a tier.

## What it measures
| Measurement | How | Bound by |
|---|---|---|
| Latency per working set | Dependent pointer chase over one random cycle of cache lines (Sattolo), √2 size steps from 4 KiB, new permutation every rep | load-to-use latency of whichever level holds the set |
| Read bandwidth | 8 independent 64-byte-vector accumulators; L1-sized and DRAM-sized buffers; 1 thread and all threads | loads (L1) / memory system (DRAM) |
| Compute ceilings | 16 independent FMA chains: scalar f64, vector f64, vector f32; 1 and all threads. The build's vector ISA is recorded (`build.vector_isa`) | FMA issue rate on that ISA |
| Timer | smallest observable step between reads, read cost, and the kernel's `clock_getres` claim | — |
| Host interference | CPU steal per rep from `/proc/stat` | — |
| Meter tiers | Each one is tried, not assumed: RAPL zone read; CPU PMU in sysfs and then `perf stat`; NVML library and then `nvidia-smi`; a cachegrind run; tier-3 calibration file | — |

**Noise.** Each rep runs every measurement once in a freshly shuffled order (the seed is recorded), each in a fresh process. Every figure carries n, median, bootstrap SE of the median, mean, SD, SE, CV, min and max. With a single rep, every spread figure is `null`, and no levels are fitted.

**Memory levels (derived).** No cache sizes are assumed.
1. **Physics first.** Latency can't fall as the working set grows. A point significantly slower than the next kept size is interference: it is excluded and listed in `memory_levels_excluded_bytes`. If more than 20% of the curve breaks this, no levels are fitted, and the profile says why.
2. **Fit.** A piecewise-linear fit of log latency on log size, with each point weighted by its own noise, widened to Student-t tails because it comes from few reps. Real plateaus drift a little, so a plateau is one sloped segment.
3. **What counts as a level:**
   - It spans at least 2× in size.
   - Its fitted rise is significantly less than half the jump at each edge. "Significantly" uses the owner's 2-SE rule, Bonferroni-corrected because the breakpoints come from the data.
   - The plain spread of its medians is also under half of each jump. The weighting can't hide that check.
   - A lone segment must show no detectable rise.
   - Neighbouring segments that would pass as one level when merged are merged first.
4. **Capacity brackets.** A bracket runs from the last size where no rep had yet reached the next level to the first size where no rep was still at this level. **If no next level was found, the upper end is `null` (unknown), never "end of sweep".**
5. **Sweep reach.** `sweep_reach` warns when the sweep stopped below 2× the reported last-level cache (it is RAM-limited), because DRAM may never have been reached.
6. **Explicit choices,** all recorded: the 2× span, the ½ rise ratio, and the 20% exclusion cap.

Every doubt resolves toward "transition", never toward "level", and each level reports its own `rise_across_span`.

**Simulated error rates.** 200 seeds through the tool's own reps → bootstrap → fit path, on two independent noise generators: mine, and the independent verifier's (Student-t noise, interference spikes, drift).

| Case | Phantom level | False boundary | Wrong value/bracket | Exact recovery |
|---|---|---|---|---|
| 4 levels, 7 reps (lognormal / Student-t / drift / rampy, both generators) | 0% | 0% | 0 | 99–100% |
| 4 levels, 7 reps, heavy-tailed (10% outliers at 1.5–3×) | 0% | 0% | 0 | 85% |
| 4 levels, 3 reps, `--quick` sweep | 0% | 0% | 0 | 99–100% |
| 4 levels, 3 reps, heavy-tailed | 0% | 0% | 0 | 56.5% (misses only) |
| Interference spikes: 10% of sizes, every rep slowed 3× | 0% | 0.5% | 12% | 70.5% |
| Flat curve, 12 / 34 points | 0% | 0% | 0 | 99% |
| Ramps (1–5%/step, 2–5% noise, 3–7 reps); alternating-slope curve; bump inside a ramp | 0% any level | 0% | — | — |
| Drifting precise plateaus, 7 reps / 3 reps quick | — | — | — | 100% / 89% (of 100) |

**Known limit.** All the wrong brackets in the spike case come from one thing: a spike on the *last* point of a level. That point then looks exactly like the start of the next level (every rep is shifted equally), so the bracket ends one grid step early. No method can tell it apart from a real boundary using this data.

**What `--reanalyse` can and can't show.** It recomputes every statistic from raw reps and checks the following:
- the rep counts match the settings;
- each rate equals its own recorded work ÷ time;
- each timed window is at least the minimum it was meant to run for;
- each chase seed matches the run seed;
- on the same host, it re-detects every meter tier.

On another host it exits 2.

> **It can't detect forgery.** Anyone can fabricate a raw section that is self-consistent and holds any numbers at all. The independent verifier forged 1.5× compute, 2× bandwidth and 0.5× latency, and all of them passed. Only an outside-anchored record (open item 3) can stop that. CPU steal and timer readings can't be checked against anything either.

## Results on two machines
**Machine A** (where it was built): Intel Xeon, KVM guest, 2 vCPUs, AVX-512. Full run 2026-11-05, seed 20261107, `profile.json`. CPU steal stayed at or below 1.1% per rep. Meter tiers: timing and cachegrind only.

| | Measured / derived | Reported by the VM |
|---|---|---|
| Level 1 (L1) | 1.81 ns over 4–44 KiB, capacity 32–64 KiB | 48 KiB |
| Level 2 (L2) | 5.77 ns over 64 KiB–1.4 MiB, capacity 1.4–4 MiB | 2 MiB |
| Level 3 (effective L3, one core) | 33.6 ns over 4–8 MiB, capacity **5.7–11.3 MiB** | **260 MiB** |
| Level 4 (DRAM) | 118 ns over 11.3 MiB–1 GiB | — |
| DRAM read, 1 / 2 threads | 16.3 / 30.6 GB/s | — |
| L1 read, 1 thread | 310 GB/s | — |
| f64 scalar / f64 vector / f32 vector, 1 thread | 10.9 / 81.9 / 157 GFLOP/s | — |
| Ridge point (f64, all cores) | 5.2 flop/byte (derived) | — |

Four full runs on this machine (seeds 20261105 ×2, 20261106 and 20261107; the first three are in this repo's history) all give L1 ≈ 1.8 ns, L2 ≈ 5.7–5.9 ns, L3 ≈ 33 ns and DRAM ≈ 118–121 ns.

**Machine B** (independent verifier): Intel Xeon model 173 at 3.2 GHz, 2 vCPUs, in Docker on KVM, 4 GiB RAM, reporting a 504 MiB L3. These are its three full runs, refitted with the current code:
- **Compute:** vec f64 102.5–103.7 GFLOP/s and vec f32 203.9–206.1 GFLOP/s against a theoretical 102.4 / 204.8. That's 100% of peak, and the verifier's own objdump confirmed the codegen.
- **L1 and L2:** L1 1.40 ns and L2 4.48 ns, found in all three runs.
- **Beyond L2:** a transition. Latency climbs noisily (10–30% per point) from about 35 to 210 ns without a plateau, up to the RAM-limited end of the sweep at 0.9 GiB, which is under 2× the reported L3. The `sweep_reach` warning fires.
- The verifier's own pointer chase and bandwidth tests matched L1 and L2 within 1%, and L1 bandwidth (306 GB/s) exactly.

**What this already changes:**
- **The old probe measured its own loops, not the hardware.** Its "3.3 GFLOP/s scalar" and "6.5 GB/s" were loop limits. The real figures here are 11 GFLOP/s and about 16 GB/s.
- **A VM's reported L3 isn't what one core gets.** Machine A reports 260 MiB, and one core sees about 6–11 MiB. Machine B shows no L3 plateau at all.
- **Shared hosts are noisy beyond L2.** Comparisons must be interleaved and gated on SE, as decided, and level fits there will often, honestly, stop at L2.

## How it was checked
Five independent reviews ran: four Claude agents, then GLM-5.2 on a different machine. Each found real problems that are now fixed. None could break the kernels.

1. **Review 1:** phantom levels, staircases on slopes, fabricated averages on short or one-rep curves.
2. **Review 2:** the noise estimate from few reps was too optimistic, because my simulation had fed in the true noise. It also found a `seconds=0` bypass and unchecked meter claims.
3. **Review 3:** ramps produced levels, upper brackets landed too early, and a changed host ID let a faked meter claim pass with exit 0.
4. **A real run, then review 4:** real plateaus drift, which broke an absolute-flatness test. Then the edge slope was being diluted by DRAM's long span, and a false claim about L3 in this card was corrected.
5. **Review 5 (GLM-5.2, machine B):**
   - The old line-vs-step check cascaded on a noisy shared host and erased L1 and L2.
   - Brackets silently stretched to the end of the sweep.
   - Interference spikes produced 7% false boundaries.
   - The sweep couldn't reach DRAM.
   - The scale of forgery wasn't stated.
   - Short timed windows and the vector ISA weren't checked or recorded.

   Its curve is kept as a regression fixture (`fixtures/shared_vm_curve.json`).

## Limits
- **Tier 2 only on both machines.** These are times, not joules. The energy score needs real-hardware calibration or a machine where RAPL or NVML shows as available.
- **Only x86 Linux has been tested.** The NVML and ARM paths are untested; the vector ISA is recorded so an ARM ceiling isn't misread.
- **Large sizes include TLB effects.** The THP mode is recorded.
- **Complexity debt.** About 1,450 lines in total, mostly the level-fit statistics and their tests. The fit has earned its place by catching false levels on two machines, but it's the part to watch.
- **Not measured yet:** write bandwidth, L2/L3 bandwidth, integer and low-precision throughput, synchronisation cost.

## Files
`kernels.c` (measurements) · `probe.py` (driver, tiers, stats, levels, profile) · `test_probe.py` · `fixtures/shared_vm_curve.json` (machine B regression curve) · `profile.json` (machine A run above, with raw reps) · `probe_v0_reference.c` (the old probe, kept for comparison)
