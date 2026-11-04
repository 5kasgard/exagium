# Independent verification brief: exagium `probe/`

You are an independent verifier. Someone else built this tool. Your job is to check its claims and try to make it report something false. Don't trust the README; report only what you can show with commands, numbers and line references. A result of "I couldn't break X" is useful, and so is "X is wrong, here is how".

**The most valuable thing you can do is run it on hardware the author never had.** It was built and checked on one 2-vCPU Intel cloud VM with no energy meters. Examples:
- bare metal;
- an AMD or ARM CPU;
- a machine with readable RAPL (`/sys/class/powercap/intel-rapl*`) or working `perf` counters;
- an NVIDIA GPU host.

## Setup
```
git clone https://github.com/5kasgard/exagium && cd exagium/probe
python3 -m unittest -v test_probe          # expect 36 passing, ~50 s
python3 probe.py --quick --out quick.json  # ~10 s
python3 probe.py --out full.json           # ~100 s on 2 cores; longer with more cores or RAM
python3 probe.py --reanalyse full.json; echo $?
```
Needs Linux, a C compiler and Python 3 (stdlib only). Valgrind is optional.

**Rules:**
- Run only one measurement process at a time, and nothing heavy alongside it, because it corrupts timing.
- Record your exact hardware (`lscpu`, `nproc`, RAM, VM or bare metal, kernel) in your report.

## Claims to test
Give each a verdict: CONFIRMED / REFUTED / UNCLEAR, with evidence.

1. **Compute ceilings are real FMA throughput.**
   - Check this with `objdump -d` on a build of `kernels.c` (`-O2 -march=native`): `scalar_f64` should be scalar FMAs with no auto-vectorisation, and `vec_f64`/`vec_f32` should be full-width vector FMAs with enough independent chains to be throughput-bound.
   - Compare the measured GFLOP/s against your CPU's theoretical peak: cores × clock × FMA units × lanes × 2. Find out your clock under load yourself.
   - Check the FLOP formula in the code.
2. **Bandwidth.**
   - The L1 figure should be load-bound, not limited by the add chain. Try changing the number of accumulators.
   - The DRAM figure should be a genuine DRAM read rate. Compare with an independent tool if you can (e.g. `mbw`, STREAM, Intel MLC, likwid).
3. **Latency curve.** The pointer chase is one random cycle over cache lines, can't be optimised away, and reports ns per dependent load. Compare with an independent tool if you can (e.g. lmbench `lat_mem_rd`, Intel MLC).
4. **Memory levels (derived).**
   - The tool claims it assumes no cache sizes, excludes points that break monotonicity as interference, and reports a level only where the rise across it is small next to the jump at each edge. When no next level is found, the bracket's upper end must be `null`, never "end of sweep".
   - Compare its levels and capacity brackets with your CPU's documented caches and with an independent latency tool.
   - Claimed simulation error rates are in the table in `probe/README.md`. Mostly they are 0% phantom levels, 0% false boundaries and 0 wrong brackets, apart from a stated limit with interference spikes on a level's last point.
   - A previous verifier's adversarial cases and noisy real curve are now regression tests, so try something new.
   - Reproduce those rates with **your own** synthetic generator, not the one in `test_probe.py`. Then try to make the fit lie with a new adversarial curve, calling `probe.describe_levels(sizes, medians, ses, df=reps-1, rep_max=..., rep_min=...)` directly. Report any curve that yields a phantom level, a false boundary, or a wrong bracket in more than 5% of noisy draws.
5. **Meter tiers.**
   - Each tier is meant to be detected by trying it, not by checking whether a binary exists.
   - On your machine, check every tier's `available` value and evidence against reality.
   - If you can, change the conditions and confirm the tier flips: install or remove `perf`, change `/proc/sys/kernel/perf_event_paranoid`, use a GPU host, make RAPL readable.
6. **`--reanalyse`.**
   - It claims to recompute every statistic from raw reps, and to check rep counts, rate = work ÷ time, chase seeds and, on the same host, the meter tiers.
   - It exits 0 when consistent, 1 when not, and 2 when consistent but run on another host.
   - It explicitly does **not** claim to prove the raw data is genuine.
   - Find a tamper it misses that would let someone overstate a result, and say whether it matters.
7. **Labels.** Every number in the profile should be marked measured / reported / derived, or be plainly raw or settings. List anything that could be mistaken for a measurement.
8. **Reproducibility.** Run the full probe 3 times with different `--seed` values. Do the levels, brackets and ceilings agree within the reported noise? Is the run-to-run spread larger than the within-run SE suggests?
9. **Portability.** What breaks on your platform (ARM, AMD, other compilers, containers, macOS)? Does anything fail silently or quietly report a wrong number instead of failing loudly?

## Report
Keep it under about 800 words, in this order:
1. Your hardware.
2. The verdict table.
3. Evidence per claim.
4. Every way you found that the tool could report something false, ranked by how badly, each with a concrete reproduction.

Attach your `full.json` profiles if you can. If you have a GitHub account, an issue on the repository with the report is ideal.
