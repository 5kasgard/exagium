# reports/glm-20261115.md — why does mlp16 retire so few instructions per nanosecond?

Independent run of the `pilot/lgn-20261115` kit, full track (shell + valgrind), all numbers below are my own
measurements on my host. `bin/clock` calibration: **3.56 GHz** (4 runs: 3.557, 3.564, 3.571, 3.570 —
the second was a 400M-iteration run; min 3.516, max 3.575), pinned `taskset -c 1`, nominal 2.30 GHz; re-checked at
delivery time: 3.561, 3.565). All cycle figures below use 3.56 GHz (clock stable to ±0.5% across the session; see H7).

## 1. Host

- CPU: Intel Xeon, family 6 model 173, 2 vCPUs, 1 thread/core (KVM guest, "full" virtualization; `lscpu` saved as
  `reference/glm-host-lscpu.txt`). L1d 48 KiB per core, L2 2 MiB, L3 504 MiB. x86-64, little-endian.
- Kernel: `5.10.134-013.15.kangaroo.al8.x86_64`. Compiler: gcc 14.2.0 (Debian) — the kit's reference used
  gcc 13.3, so my instruction counts differ where gcc 14.2 codegen differs (biggest for `linear`).
- Hardware counters: **none**. `perf_event_open(PERF_COUNT_HW_CPU_CYCLES)` returns ENOENT (PMU not
  virtualized; `perf_event_paranoid=2`). `perf` binary not present. So rule 4 falls back to `bin/clock` GHz.
- valgrind 3.24.0 (user-local build under `$HOME/local`, wrapper sets `VALGRIND_LIB`); every Ir/img below is
  the cachegrind `I refs` at N=95,600 minus N=0, exactly as `bench.py` computes it.
- VM: yes (KVM). That matters for H7: noise exists but is absorbed by the interleaved bench (below).

## 2. Baseline and derived quantities (my numbers)

`results/glm-base-1.json`, 21 rounds, N=95,600, pinned (tag `glm-base-1`); cycles = ns × 3.56:

| program | ns/img median | IQR | Ir/img | cyc/img | IPC |
|---|---|---|---|---|---|
| linear | 233.00 | 231.1–235.4 | 4187.6 | 829 | 5.05 |
| thermo | 202.61 | 202.0–203.4 | 3545.6 | 721 | 4.92 |
| **mlp16** | **67.42** | 66.4–68.2 | 652.4 | 240 | **2.72** |
| mlp32 | 101.22 | 99.8–104.4 | 952.3 | 360 | 2.64 |
| mlp64 | 205.64 | 203.5–210.3 | 1605.9 | 732 | 2.19 |
| mlp16_nb4_bl1 | 70.04 | 69.2–71.0 | 794.5 | 249 | 3.19 |

The brief's puzzle reproduces on my host: mlp16 does 15.6% of linear's instructions for 29% of its time;
each mlp16 instruction costs 1.86× a linear one (reference host: 2.4×; gcc-14.2 linear carries +9.8% Ir vs
the reference's gcc-13.3 linear, which narrows the ratio). All programs above print the same predictions as
their reference groups: mlp32 ≡ thermo, mlp16_nb4_bl1 ≡ mlp16 (`identical_output_groups` in the JSON).

Layer split (same run, `results/glm-h3.json`; differences of medians, IQRs disjoint or touching by <1 ns):

| component | ns/img | cyc/img | share |
|---|---|---|---|
| layer 1 + validation + fread + output (h3_l1only) | 47.96 | 171 | 71.5% |
| layer 2 only (h3_noarg − h3_l1only) | 13.38 | 48 | 20.0% |
| argmax only (mlp16 − h3_noarg) | 5.72 | 20 | 8.5% |

nz = 12.33 nonzero hidden units/img (my cachegrind per-line data: line `s0 += b*W2[j][0]` executed 12.33×/img,
Dr 12.33/img). Static per-image counts in the hot path (my objdump `reference/glm-mlp16.objdump.txt`):
64 × 1 B pixel loads + 128 × 16 B fused `paddw` loads in layer 1, ~120 ALU for addresses, 2 chains of 64
`paddw` (1 c each, floor 64 c); layer 2: 3 × 16 B stack loads per nonzero unit (W2 hoisted out of the loop by
gcc 14.2 into xmm regs + stack slots); argmax: 9 data-dependent `comiss` branches; validation 32 Ir/img;
total ≈ 235 load-ops/img. At 2–3 loads/cycle that alone is 78–118 c of the measured 240 c.

## 3. Hypothesis table

All comparisons are same-run against the original in that run's spec (rule 2). Verdicts: **supported** /
**ruled out** / **inconclusive**, each with the deciding numbers.

| H | variant (run, tag) | one thing changed | ns/img median [IQR] vs same-run mlp16 | Ir/img | verdict |
|---|---|---|---|---|---|
| H1 | h1_const (`glm-h1h2`) | layer-1 row address drops the `pp[q]*32` term: no byte loads, no data-dependent addresses, no address ALU; same 128 table loads + 128 paddw + same 2 acc chains | 55.23 [54.5–56.4] vs 67.46 [66.3–70.7] → **−18.1%**, disjoint | 454.0 | **partial**: removing byte loads + address math + (side effect) layer-1-driven mispredict data saves 43 c, but Ir/ns *drops* to 8.2 — the remaining loop is still mlp16-class slow. Load count alone does not explain the time. Confound declared: this also changes the h distribution, so mispredicts fall 3.47→0.035/img (model). |
| H1 | h1_half (`glm-h1h2`) | accumulation loop k: 2→1: 64 table load-ops + 64 paddw instead of 128 + 128; byte loads, addresses, layer 2, argmax unchanged | 53.56 [52.3–54.3] vs 67.46 → **−20.6%**, disjoint | 513.7 | **partial**: −49 c for −64 load-ops −64 add-ops (≈32 c at 2 loads/c + ≈13 c at 5 ALU/c, both ports near limits). Consistent with mixed load/add throughput binding layer 1; the 64 c accumulator chain persists in both, so the saving is not from the chain. |
| H1 | h1_addless (`glm-h1addless-ceil8`) | `acc[k] += row[k]` → `= row[k]` (drop the adds) | 29.39 [28.9–30.1] vs 67.08 | 164.1 | **failed ablation, reported as such**: Ir/img 164 proves gcc deleted 7 of 8 blocks (only the last block's loads are live). The number is unusable as designed; rule 1 exists for exactly this. Not counted as evidence. |
| H2 | h2_pre (`glm-h1h2`) | two phases: compute all 64 row pointers first (all byte loads + address math kept), then do the 128 loads+adds | 98.79 [97.3–99.3] vs 67.46 → **+46.4% slower**, disjoint | 1047.4 | **ruled out**: breaking the load→lea→load chain gives no benefit; the slowdown matches the declared added traffic (395 Ir: 64+64 pointer ops; 101 br/img vs 37). The address chain is not what the 275-ish cycles are waiting on. |
| H3 | h3_l1only / h3_noarg (`glm-h3`) | see layer split above | 47.96 [46.6–49.4] / 61.34 [60.3–62.4] vs 67.06 [65.7–69.0] | 410.0 / 576.0 | **supported** (it is the measurement): layer 2 = 48 c ≈ 12.33 units × ~4 c serial `addps` chain latency (estimate; the s-chains are 12.3 deep, 3 parallel chains — chain *depth*, so NB-interleaving cannot shorten it, consistent with the kit's NB finding). Argmax = 20 c total *including* its branches. |
| H4 | h4_noh (`glm-h45`) | remove `short h[16]` + its memcpy; layer 2 reads constant-index vector lanes | 66.86 [66.4–68.1] vs 67.31 [66.6–68.9] → no difference (overlapping IQRs) | 652.4 | **ruled out**, twice: (a) statically, the premise is absent — both builds read h via `pextrw` from registers, `memcpy(h,...)` compiles to 0 Ir, there is not a single 2-byte scalar load (`movzwl`) in either objdump; (b) empirically, h4_noh ≡ mlp16 predictions and speed. |
| H5 | h5_u1 (`glm-h45`) | layer-2 `#pragma GCC unroll 64` → `1` | 73.31 [72.4–74.9] vs 67.31 [66.6–68.9] → **+8.9% slower**, disjoint | 708.1 | **ruled out**: code size of `main` 1255 B vs 2852 B, and the *smaller* code is slower. I1 misses are 1,419 per run = 0.015/img (cachegrind). The unrolled code is not hurting; the front end is not the cause. |
| H6 | linear itself + ceil probes | — | linear 230.4–234.0 over 8 runs; Ir/ns 17.89–18.18 | 4187.6 | **inconclusive as a ceiling claim, informative as a contrast**: the host does reach ~18 Ir/ns (5.05 IPC) on linear/thermo-shaped loops, reproducibly; nothing I built got above 3.19 IPC (mlp16_nb4_bl1). My attempts to write a *faster* trivial loop were defeated by gcc, and are reported as evidence of that, not of a ceiling: ceil16 (16 int chains, 3 accumulators spilled to stack, `add %r,0xNN(%rsp)` RMW chains) = 6.68 Ir/ns; ceil8 (8 chains) = 7.71 Ir/ns but gcc vectorized it into a pack/psrlw soup (objdump), not the intended scalar chains. |
| H7 | base-1 vs base-2 + 5 clock runs | bench repeated ~35 min apart in-session; clock before/after each | mlp16 67.42 [66.4–68.2] → 65.84 [64.9–66.4]; linear 233.0 → 230.4; clock 3.557→3.571 GHz | — | **ruled out as an explanation**: whole-run levels drift together ±2% (base-2 uniformly faster); medians stable across 8 runs (mlp16 65.8–68.3, linear 230.4–234.0). One run (`glm-h9`) showed transient noise (linear IQR up to 354 ns) — its medians still landed within 1% of the other runs, and its conclusions were replicated (`glm-h9b`). A 2% shared drift cannot explain a 1.86× per-instruction cost gap. Caveat: "different times" means within one ~1 h session; no diurnal sampling was possible. |
| H8 | not run | — | — | — | **not testable here**: my L1d is 48 KiB per core (table fits). No 32 KB-L1d host is available in this environment. `lscpu` recorded. The kit's cross-host question stays open — nothing in my data speaks to it (my host, like the reference, fits the table in L1d). |
| H9 | h9_l2bl (`glm-h9`, `glm-h9b`, 2 runs) | remove layer-2 zero-skip (`if (!h[j]) continue`), all 16 units processed; 0·w = ±0, s+(±0)=s | 68.08 [66.6–77.0] and 68.82 [67.9–70.6] vs 68.28 [66.0–77.0] and 67.36 [66.4–69.3] → no improvement (IQRs overlap) | 680.5 | **ruled out as a major cause**: removing 56% of modelled mispredicts (1.96/img of 3.47) changes nothing measurable (Δ = +1.5 ns, if anything slightly slower: the s-chains go 12.3→16 deep, +15 c, offset by the removed branches). Predictions identical to mlp16 (output group). |
| H9 | h9_bl (`glm-h9`, `glm-h9b`, 2 runs) | h9_l2bl + branchless argmax (volatile compare-select → cmov chain) | 81.37 [79.7–100.3] and 81.81 [81.0–83.0] vs 68.28 [66.0–77.0] and 67.36 [66.4–69.3] → **+19%/+21% slower**, replicated, disjoint | 692.0 | **ruled out**: making all 25 data-dependent branches branchless (model mispredicts 3.47→0.035/img, verified by cachegrind) is *slower* — the serial cmov+volatile-load chain costs more than the branches it replaces. The argmax's *total* cost is only 5.7 ns = 20 c (H3 split), which bounds its mispredict cost below that. |

Cachegrind model vs reality (my host): the model says 3.47 misp/img (9.3% of 37.25 br/img); at the brief's
15–20 c each that would be 52–69 c of the 240 c. My ablations bound the real branch-related cost at ≤20 c
total: h9_l2bl Δ≈0, whole-argmax (branches included) = 20 c, h9_bl (zero mispredicts) +19/+21% *slower*.
The model's counts do not translate to cycles on this host. No PMU, so real mispredict counts are unmeasurable;
that is the honest limit of this H9 verdict.

## 4. What I think the cause is, and with what confidence

On my host mlp16 spends its 240 c/img as: layer 1 ≈ 171 c (71%), layer 2 ≈ 48 c (20%), argmax ≈ 20 c (8.5%),
the rest is validation/fread/output. With everything below backed by the table above:

1. **Layer 2 is chain-latency bound, by construction.** 12.33 nonzero units/img append to three s-chains;
   each append is a serial `addps` (≈4 c latency, estimate) → ≈49 c, measured 48 c. This is a chain-*depth*
   property: interleaving (NB) or unrolling adds parallel chains but cannot shorten the depth — which is why
   the kit's NB sweep found nothing, and why h9_l2bl (16-deep chain) is not faster than mlp16 (12.3-deep).
   Confidence: high (measured split + unit-count + the two independent NB/branchless results).
2. **Layer 1 is throughput-bound at roughly half its arithmetic-floor rate, and the residual is unattributed.**
   What is measured: (a) its cost scales with the load-op/add count (h1_half: −64 of each → −49 c); (b)
   killing the byte loads + address arithmetic + (side effect) the mispredict data saves only 43 c and leaves
   the loop at 8.2 Ir/ns (h1_const); (c) the load→address chain is not the constraint (h2_pre); (d) the
   64-c accumulator chain persists in every variant and nothing got layer-1-like code above ~9.6 Ir/ns.
   The port arithmetic (192 load-ops at 2/c = 96 c, 128 paddw at ≥4/c ≈ 26 c, ~120 ALU at 5/c ≈ 24 c) gives a
   floor of ~100–110 c; measured 171 c. The ~60 c residual is what I cannot attribute with the tools available
   (no PMU). Labelled hypothesis, with the test that would decide it: effective load-issue throughput of the
   mixed 1 B / 16 B stream under the 8-iteration block loop is ~1.5 ops/c — decidable with
   `perf stat -e cycles,instructions,mem_inst_retired.all_loads,L1-dcache-loads` on bare metal.
   Confidence: high for "throughput, not latency, not branches"; low for any finer split inside layer 1.
3. **Nothing else is carrying the time.** Ruled out by direct ablation with disjoint IQRs or identical output:
   address-chain serialization (H2), h store-to-load forwarding (H4), front-end/code size (H5), branch
   mispredicts as a major cost (H9), VM noise/frequency (H7, ±2% shared). Cache: D1 misses 2.1/img
   (mine: table 0.06/img, input stream the rest) — at L1-miss latency that is at most a minor, mostly-hidden
   contributor; H8 untestable here.

So the brief's question "what fills those cycles": on my host, ~71% is layer 1 running ~1.7× above its port
floor under a load-op stream gcc fuses into `paddw` memory operands, ~20% is layer 2's serial `addps` chain
depth (12.3 × ~4 c — unfixable by interleaving), ~8.5% is a well-predicted argmax, and the rest is I/O
machinery. There is no single dominant stall; the honest headline is the ruled-out list. Overall confidence:
high on the decomposition and every verdict in the table; medium on the inside of layer 1; explicitly
unattributed-without-PMU for its last ~60 c.

## 5. Things in the brief I found wrong (with the lines to point at)

1. **H4's premise is false, in the kit's own reference build.** "h[] is written as 16-byte vectors with
   memcpy, then read back as 2-byte scalars" — `reference/mlp16.objdump.txt` (gcc 13.3) contains 16
   `pextrw` register-lane reads, zero `movzwl` (2-byte loads) anywhere, and `reference/mlp16.cachegrind-lines.txt`
   shows the `memcpy(h + 8*k, &a, ...)` lines at 0 Ir: gcc's SRA removed h entirely. Same in my gcc-14.2 build
   (`reference/glm-mlp16.objdump.txt`, `reference/glm-mlp16.cachegrind-lines.txt`). H4 was untestable as
   stated; the compiled code already "extracts lanes from registers".
2. **H3's "W2 rows reloaded from the stack" is half-right in both builds.** gcc hoists all W2 rows out of the
   image loop (my objdump: `movaps` to `0x0(%rsp)`…`0x2c0(%rsp)` once before the loop, plus 6 rows live in xmm
   registers). Per nonzero unit layer 2 does 3 stack loads (`movaps 0xNN(%rsp)`), so "reloaded from the
   stack" holds per unit; nothing reloads from `.data` inside the loop.
3. **H9's branch attribution is off, and its cycle estimate is refuted by my ablations.** Per-line cachegrind
   (my run): the layer-2 zero-skip carries 187,144 modelled mispredicts (56.4%, 1.96/img), the argmax
   132,190 (39.8%, 1.38/img), and the p-loop — "layer 1's loop" in the brief — only 9,419 (0.10/img): the
   loop branch is innocent. The brief's "50–60 of the ~275 cycles" is not supported here: removing 56% of
   modelled mispredicts changes nothing (h9_l2bl), the argmax costs 20 c total, and fully branchless is 19–21%
   slower (h9_bl). The cachegrind predictor model should not be used to budget cycles.
4. **The brief's table is gcc-13.3-specific.** On gcc 14.2 (my host) linear is +9.8% Ir/img (4187.6 vs 3812)
   while mlp16 is +1.2% (652.4 vs 645): part of "linear retires 21.5 Ir/ns" is gcc-13.3 codegen for linear.
   The qualitative puzzle survives (my ratios: 18.0 vs 9.7 Ir/ns; IPC 5.05 vs 2.72), but the 2.4× per-
   instruction factor is compiler-dependent (1.86× here).
5. **Confirmed, not wrong:** "the branchless layer 2 ... is exact" — holds on my host (identical output
   group; 0·w = ±0, s+(±0) = s), and "isn't faster" also holds (it is 2–4% slower here, IQRs disjoint in both
   base runs); the NB-interleave conclusion is consistent with everything I measured (it is the same
   chain-depth argument as layer 2 above).

## 6. Reproducibility

- Bench JSONs: `results/glm-base-1.json`, `glm-base-2.json`, `glm-h1h2.json`, `glm-h3.json`, `glm-h45.json`,
  `glm-h9.json`, `glm-h9b.json`, `glm-h1addless-ceil8.json` (raw per-round times included in each).
- Variant sources: `src/h1_const.c`, `h1_half.c`, `h1_addless.c`, `h2_pre.c`, `h3_l1only.c`, `h3_noarg.c`,
  `h4_noh.c`, `h5_u1.c`, `h9_l2bl.c`, `h9_bl.c`, `ceil.c`; build lines in `build.sh` above the marker;
  spec JSONs `programs_h1h2.json`, `programs_h3.json`, `programs_h45.json`, `programs_h9.json`, `programs_f.json`.
- My static artifacts (gcc 14.2): `reference/glm-mlp16.objdump.txt`, `glm-linear.objdump.txt`,
  `glm-mlp16.cachegrind-summary.txt`, `glm-mlp16.cachegrind-lines.txt` (built from a `-g` twin of the exact
  `-O2` binary), plus `src/perfprobe.c` for the PMU check.
- Every "must be identical" variant landed in mlp16's output group in the results JSON: h2_pre, h4_noh,
  h5_u1, h9_l2bl, h9_bl, mlp16_nb4_bl1.
