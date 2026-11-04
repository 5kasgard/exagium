# forge-probe — spec card

**What it is.** The Forge's first brick: a tool that measures a machine's invariants and writes `profile.json`. It records memory latency at each working-set size, the memory plateaus found in that curve, read bandwidth, compute ceilings, timer quality, and which meter tiers the machine offers. Every number is labelled **measured** (timed here), **reported** (the OS or CPU's claim) or **derived** (arithmetic on measured numbers).

**Run.**
```
python3 probe.py                      # full: ~100 s on 2 vCPUs, 7 reps, sweep to 4x reported LLC (cap: 1/4 of free RAM)
python3 probe.py --quick              # ~10 s smoke run
python3 probe.py --reanalyse p.json   # recompute every statistic from raw reps; exit 1 on mismatch
python3 -m unittest -v test_probe     # 13 tests, ~2 s
```
Needs a C compiler and Python 3 (stdlib only). Linux. Valgrind optional (detected as a tier).

## What it measures
| Measurement | How | Bound by |
|---|---|---|
| Latency per working set | Dependent pointer chase over one random cycle of cache lines (Sattolo), √2 size steps from 4 KiB | load-to-use latency of whichever level holds the set |
| Memory plateaus | Piecewise-constant fit of log latency, each point weighted by its own repeat noise, segment count by BIC; single-point segments reported as *ramps*, never as levels | data only: no cache sizes assumed |
| Read bandwidth | 8 independent 512-bit accumulators; L1-sized and DRAM-sized buffers; 1 thread and all threads | loads (L1) / memory system (DRAM) |
| Compute ceilings | 16 independent FMA chains: scalar f64, vector f64, vector f32; 1 and all threads | FMA issue rate |
| Timer | clock resolution and read overhead | — |
| Meter tiers | Each one is tried, not assumed: RAPL zone read, `perf stat` counts, `nvidia-smi power.draw`, a cachegrind run | — |

**Noise.** Each rep runs every measurement once in a freshly shuffled order (seed recorded), each in a fresh process. Reported per figure: n, median, mean, SD, SE, CV, min, max. A single rep gives `null` noise, never a made-up value. Raw per-rep records are stored, so every figure can be recomputed (`--reanalyse`).

## First result (one full run, this cloud VM, 2026-11-05)
Intel Xeon (KVM guest, 2 vCPUs, AVX-512). Meter tiers: **timing and cachegrind only**. No RAPL, perf counters or NVML, so nothing here is energy.

| | Measured | Reported by the VM |
|---|---|---|
| L1 | 1.8 ns, ends 44–64 KiB | 48 KiB |
| L2 | 5.8 ns, ends 1.4–2 MiB | 2 MiB |
| L3 (effective, one core) | 33.7 ns, ends 11–16 MiB | **260 MiB** |
| DRAM | 105–118 ns | — |
| DRAM read, 1 / 2 threads | 17.1 / 31.3 GB/s | — |
| L1 read, 1 thread | 305 GB/s | — |
| f64 scalar / vector / f32 vector, 1 thread | 11.2 / 83.6 / 166 GFLOP/s | — |
| Ridge point (f64, all cores) | 5.0 flop/byte (derived) | — |

**What this already changes:**
- The old `probe.c` figures were limits of its own loops, not of the hardware. Its "3.3 GFLOP/s scalar" ran far below the measured 11 GFLOP/s scalar FMA ceiling, and its "6.5 GB/s" bandwidth used a single scalar accumulator against the measured 17 GB/s. Anything built on those numbers would have been tuned to the probe, not the machine.
- The VM's reported 260 MiB L3 is the host's. One core here sees roughly 11–16 MiB at L3 latency. Optimising to the reported size would be optimising to a fiction.

## Limits
- Tier 2 only here: times, not joules. The energy score needs item 5 (real-hardware calibration) or a machine where RAPL/NVML shows as available.
- One run, one machine. Noise figures cover this run only.
- Latency at large sizes includes TLB effects. Huge pages are requested (`madvise`) and the THP mode is recorded, but whether they were granted isn't verified.
- All-thread figures on a 2-vCPU VM share cores with the OS and the driver, so they are noisier (CV 5–15%).
- x86 Linux tested; `-march=native` falls back to plain `-O2` when unsupported, which can lower the vector ceiling, and the build flags are recorded.
- Not measured yet: write bandwidth, L2/L3 bandwidth, integer/low-precision throughput, synchronisation cost. Each is a candidate hypothesis, not a gap to fill by default.

## Files
`kernels.c` (measurements) · `probe.py` (driver, tiers, stats, plateaus, profile) · `test_probe.py` · `profile.json` (the run above, with raw reps) · `probe_v0_reference.c` (old probe, kept for comparison)
