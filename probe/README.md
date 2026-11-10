# forge-probe — spec card

**What it is.** The first brick: a tool that measures a machine's invariants and writes `profile.json`. It records memory latency at each working-set size, the memory levels fitted to that curve, read bandwidth, compute ceilings, timer quality, host interference, and which meter tiers the machine offers. Every number is labelled **measured** (timed here), **reported** (the OS or CPU's claim) or **derived** (computed or fitted from measured numbers).

**Run.**
```
python3 probe.py                      # full: ~100 s on 2 vCPUs, 7 reps, sweep to 4x reported LLC (cap: 1/4 of free RAM)
python3 probe.py --quick              # ~10 s smoke run, 3 reps, coarser sweep
python3 probe.py --reanalyse p.json   # recompute from raw reps; exit 0 consistent, 1 inconsistent,
                                      #   2 consistent but meter claims unverifiable (different host)
python3 -m unittest -v test_probe     # 40 tests, ~65 s
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
1. **Physics first.** Latency can't fall as the working set grows. A point significantly slower than the next kept size is interference: it is excluded and listed (`memory_levels_excluded_bytes`). If more than 20% of the curve breaks this, no levels are fitted, and the profile says why.
2. **Fit.** A piecewise-linear fit of log latency on log size, with each point weighted by its own noise, widened to Student-t tails because it comes from few reps. Real plateaus drift a little, so a plateau is one sloped segment.
3. **What counts as a level.** A segment spanning at least 2× in size, where at each edge:
   - the edge steps *up* toward larger sizes;
   - its fitted rise is significantly less than half the jump. "Significantly" uses the owner's 2-SE rule, Bonferroni-corrected because the breakpoints come from the data;
   - the plain spread of its interior medians is under half the jump. Weighting can't hide this check, and the end points are excluded because that's where reps mix at a boundary;
   - its slope is under half the edge's slope, so a long ramp is never taken for an edge.

   A lone segment must show no detectable rise.
4. **Merging.** Two neighbours are merged only when the step between them is under half the outer jumps, each measured from that piece's own line, and the result still passes as a level.
5. **Capacity brackets.** A bracket runs from the last size where no rep had yet reached the next level to the first size where no rep was still at this level. If no next level was found, the upper end is `null` (unknown), never "end of sweep". **A bracket covers what these reps showed. It isn't a confidence interval for the true capacity.**
6. **Sweep reach.** `sweep_reach` warns when the sweep stopped below 2× the reported last-level cache (it is RAM-limited), because DRAM may never have been reached.
7. **Explicit choices** are stored as structured fields in `memory_levels_choices`: 2× span, ½ ratio, 20% exclusion cap.

Every doubt resolves toward "transition", never toward "level", and each level reports its own `rise_across_span`.

**Simulated error rates.** Run through the tool's own reps → bootstrap → fit path on three independent noise generators: mine, GLM-5.2's (Student-t noise, interference spikes, drift), and a fresh Claude reviewer's (soft random-replacement transitions, off-grid capacities, per-level noise, correlated noise, per-rep capacity jitter). 200 seeds each.

| Case | Phantom level | False boundary | Wrong value/bracket | Exact recovery |
|---|---|---|---|---|
| 4 levels, 7 reps: lognormal, Student-t, drift, rampy, off-grid hard steps | 0% | 0% | 0 | 100% |
| 4 levels, 7 reps, heavy-tailed (10% outliers at 1.5–3×) | 0% | 0% | 0 | 85.5% |
| 4 levels, 7 reps, very different noise per level / one precise level beside noisy ones | 0% | 0% | 0 | 99.5% |
| 4 levels, 3 reps (`--quick` sweep) | 0% | 0% | 0 | 99–100% |
| 4 levels, 3 reps, heavy-tailed or Student-t(2) | 0% | 0% | 0 | 63–72.5% (misses only) |
| Drifting precise plateaus, 7 reps / 3 reps quick | 0% | 0% | 0 | 100% / 100% (of 100) |
| Flat curve, 12 / 34 points | 0% | 0% | 0 | 93.5–97.5% |
| Ramps (1–5%/step, 2–5% noise; a curved precise ramp; alternating slopes; a bump in a ramp) | 0% any level | 0% | — | — |
| Noise correlated across neighbouring sizes | 0% | 0% | 0 | 21–40%; refuses 45% (the safe failure) |
| **Interference spikes:** 10% of sizes, every rep slowed 3× | 0% | 0.5% | **12.5%** | 73% |
| **Capacity varies per rep** (15–30%), 3 reps | 0% | **11–18.5%** | — | 70.5% (7 reps: 0.5% wrong) |
| **Soft transitions** (hit rate = capacity ÷ size, as in random-replacement caches), 7 reps | 0% | **15.5%** | value 19% | 5% |

**Known limits** (the bold rows above):
- **A spike on the last point of a level** looks exactly like the start of the next level, so the bracket ends one grid step early. No method can tell them apart using this data.
- **With 3 reps, a bracket only covers what 3 reps showed.** When capacity varies from rep to rep (page placement in physically indexed caches), it misses the true capacity by up to 1.38×. Use 7 reps for brackets.
- **Under soft transitions there is no clean plateau after a capacity.** Brackets miss by up to 1.45×, and a level's value is a mixture of two levels. Fitting a soft-transition model is the next improvement to test.

**What `--reanalyse` can and can't show.** It recomputes every statistic from raw reps and checks the following:
- the rep counts match the settings;
- each rate equals its own recorded work ÷ time;
- each timed window is at least its minimum;
- each chase seed matches the run seed;
- on the same host, it re-detects every meter tier.

On another host it exits 2.

**Digests, for the record to anchor.** Each profile carries `digests.profile_sha256` (the whole profile) and `digests.measurement_sha256` (raw reps, settings, build, environment, meter tiers, time). Both are sha256 over canonical JSON: sorted keys, no whitespace, UTF-8, and no field joining. A run prints the profile digest at the end, and `--reanalyse` recomputes it. Once a digest is anchored somewhere outside (open item 3), any later edit shows up as a different digest. A digest proves only that the profile hasn't changed since anchoring, never that it's genuine; independent re-measurement (open item 4) is what tests genuineness. The scheme uses Python's float formatting, which isn't yet RFC 8785 (JCS), so another language must match it to reproduce a digest.

> **It can't detect forgery.** Anyone can fabricate a raw section that is self-consistent and holds any numbers at all. An independent verifier forged 1.5× compute, 2× bandwidth and 0.5× latency, and all of them passed. Only an outside-anchored record (open item 3) can stop that. CPU steal, timer readings and the recorded seed (rewriting it and every chase seed together stays consistent) can't be checked against anything either.

## Results on three machines
**Machine A** (where it was built): Intel Xeon @ 2.10 GHz, KVM guest, 2 vCPUs, AVX-512. Full run 2026-11-05, seed 20261109; its profile is `probe/profile.json` at commit `a014e15`. CPU steal stayed at or below 1.0% per rep. Meter tiers: timing and cachegrind only.

| | Measured / derived | Reported by the VM |
|---|---|---|
| Level 1 (L1) | 1.79 ns over 4–44 KiB, capacity 44–64 KiB | 48 KiB |
| Level 2 (L2) | 5.79 ns over 64 KiB–2 MiB, capacity 1.4–2.8 MiB | 2 MiB |
| Level 3 (effective L3, one core) | 33.5 ns over 2.8–11.3 MiB, capacity **8–16 MiB** | **260 MiB** |
| Level 4 (DRAM) | 117 ns over 16 MiB–1 GiB | — |
| DRAM read, 1 / 2 threads | 16.5 / 31.0 GB/s | — |
| L1 read, 1 thread | 305 GB/s | — |
| f64 scalar / f64 vector / f32 vector, 1 thread | 11.0 / 81.8 / 155 GFLOP/s | — |
| Ridge point (f64, all cores) | 5.2 flop/byte (derived) | — |

**Eight full runs on machine A.** Seven are kept in this repo's history: commits `a671157`, `a333eaf`, `a69c20a`, `f2c1c1c`, `6760b5c`, `ce1f0de`, plus this one. Refitted with the current code, all seven give L1 1.8 ns, L2 5.7–5.9 ns, L3 32.9–33.9 ns and DRAM 115–121 ns. One further run (seed 20261108) found L1, L2 and DRAM but called 2–8 MiB a transition, because mixed points at both ends hid L3. That's a conservative miss. Its raw data was overwritten before commit, so it can't be re-checked.

**Machine B** (independent verifier GLM-5.2): Intel Xeon model 173 at 3.2 GHz, 2 vCPUs, in Docker on KVM, 4 GiB RAM, reporting a 504 MiB L3. These are its three full runs, refitted with the current code:
- **Compute:** vec f64 102.5–103.7 GFLOP/s and vec f32 203.9–206.1 GFLOP/s against a theoretical 102.4 / 204.8. That's 100% of peak, and the verifier's own objdump confirmed the codegen.
- **L1 and L2:** L1 1.40 ns and L2 4.48 ns (drifting to about 5.8 ns by 1 MiB), found in all three runs.
- **Beyond L2:** a transition, from about 9 ns at 1.4 MiB through 35 ns at 2 MiB to about 200 ns at 724 MiB, the largest size measured, with no plateau and 10–30% noise per point. The RAM-limited sweep ended below 2× the reported L3, and `sweep_reach` warns about it.
- The verifier's own pointer chase and bandwidth tests matched L1 and L2 within 1%, and L1 bandwidth (306 GB/s) exactly.

**Machine C** (this workspace after it restarted on a different host): Intel Xeon @ 2.80 GHz (AVX-512 with VNNI), KVM guest, 2 vCPUs, reporting L1d 32 KiB and L2 1 MiB per core, and a shared 33 MiB L3. Full run 2026-11-05, seed 20261110, `profile.json`, with digest `cda1c740…3df7`. CPU steal stayed at or below 1.1%. Meter tiers: timing and cachegrind only.
- **Levels:** L1 1.57 ns over 4–44 KiB (capacity 32–64 KiB) and L2 4.41 ns over 64–724 KiB (capacity above 724 KiB, upper end unknown). Beyond that, a transition.
- **Beyond 1 MiB:** latency is about 20–29 ns at 1.4–2.9 MiB. Above 4 MiB every size is **two-speed**: the fastest rep is always about 95 ns, while medians swing between about 97 and 165 ns, with slow reps up to 210 ns. That looks like two memory paths (remote NUMA memory or host contention, for example). There's no plateau to fit, and the fit says so. The sweep reached 7.8× the reported L3, so this isn't a range problem.
- **Bandwidth and compute:** DRAM read 10.0 / 17.8 GB/s (1 / 2 threads), L1 read 174 GB/s. Compute: scalar f64 12.6, vec f64 79.4 and vec f32 143 GFLOP/s (1 thread). Those figures are consistent with two FMA units and the usual AVX-512 clock drop. No peak claim is made, because the clock under load wasn't measured.

**What this already changes:**
- **The old probe measured its own loops, not the hardware.** Its "3.3 GFLOP/s scalar" and "6.5 GB/s" were loop limits. The real figures here are 11 GFLOP/s and about 16 GB/s.
- **A VM's reported L3 isn't what one core gets.** Machine A reports 260 MiB, and one core sees about 8–16 MiB. Machines B and C show no L3 plateau at all.
- **Shared hosts are noisy beyond L2.** Comparisons must be interleaved and gated on SE, as decided, and level fits there will often, honestly, stop at L2: on two of the three machines they did. Machine C's two-speed memory also means the same workload can see about 1.7× different DRAM latency from rep to rep. The meter has to compare candidates interleaved, never one run against another.
- **The cloud workspace can move hosts between sessions** (A → C here). A profile belongs to the host it was measured on, and comparisons are only valid within one host.

## How it was checked
Seven independent reviews ran: five Claude agents, and GLM-5.2 twice on a different machine. Each of the first six found real problems, which are now fixed; the seventh confirmed the fixes. None could break the kernels.

1. **Review 1:** phantom levels, staircases on slopes, fabricated averages on short or one-rep curves.
2. **Review 2:** the noise estimate from few reps was too optimistic, because my simulation had fed in the true noise. It also found a `seconds=0` bypass and unchecked meter claims.
3. **Review 3:** ramps produced levels, upper brackets landed too early, and a changed host ID let a faked meter claim pass.
4. **A real run, then review 4:** drifting real plateaus broke an absolute-flatness test, the edge slope was diluted by DRAM's long span, and a false L3 claim in this card was corrected.
5. **GLM-5.2 on machine B:**
   - a cascading check erased L1 and L2 on a noisy shared host;
   - brackets stretched to the end of the sweep;
   - interference spikes made false boundaries;
   - the sweep couldn't reach DRAM;
   - the scale of forgery wasn't stated;
   - no timed-window check, and no ISA record.

   Its curve is a regression fixture (`fixtures/shared_vm_curve.json`).
6. **Review 6:** a curved precise ramp gave a phantom level 51.5% of the time, because the edge test ignored direction and a long ramp could act as an "edge". It also found weighting traps when merging, crash paths, and overstated figures in this card. Fixing those also exposed two false merges on real data (L2 with noisy L3 on machine B, and L3 with DRAM on machine A), both fixed and checked against every kept real run (the machine B case is a regression test; the machine A case isn't yet).

7. **GLM-5.2 re-run on machine B (commit `c6a5999`):**
   - every claim confirmed, and its earlier break cases are now fixed or documented;
   - all three seeds find L1 and L2;
   - five new adversarial cases gave no phantom levels or false boundaries;
   - what still passes is careful self-consistent forgery and rewritten seed or steal values, which only an outside anchor can catch.

The commit history shows each step.

## Limits
- **Tier 2 only on every machine so far** (A, B, C and Z). These are times, not joules. The energy score needs real-hardware calibration or a machine where RAPL or NVML shows as available.
- **Only x86 Linux has been tested.** The NVML and ARM paths are untested; the vector ISA is recorded so an ARM ceiling isn't misread.
- **Large sizes include TLB effects.** The THP mode is recorded.
- **Complexity debt.** About 1,600 lines in total, mostly the level-fit statistics and their tests. Seven reviews say the fit has earned its place, but it's the part to watch. The known limits above are where the next round should start.
- **Not measured yet:** write bandwidth, L2/L3 bandwidth, integer and low-precision throughput, synchronisation cost.

## Files
`kernels.c` (measurements) · `probe.py` (driver, tiers, stats, levels, profile) · `test_probe.py` · `fixtures/shared_vm_curve.json` (machine B regression curve) · `profile.json` (machine C run above, with raw reps and digests; machine A's run is at commit `a014e15`) · `probe_v0_reference.c` (the old probe, kept for comparison)
