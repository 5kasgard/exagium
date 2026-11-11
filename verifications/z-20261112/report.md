# Independent verification of exagium — Z (GLM-5.3), 2026-11-12

Against `origin/main` at `410a6c9`. All four bricks tested.

## 1. Hardware

Intel Xeon model 173 (Granite Rapids, AVX-512), 2 vCPU, KVM guest in Docker, 4 GB RAM, kernel 5.10.134. gcc 14.2, Python 3.12.16, numpy 2.5.3, valgrind 3.24.0 (from .deb), bubblewrap 0.12.0 (same), sigstore 4.5.1, opentimestamps-client 0.8.0 (pip). No RAPL/NVML/perf/PMU. Neither jail works: bwrap can't mount /proc; confine needs Landlock ABI 4+ (Linux 6.7+). Loop re-measured without jail, code read first.

## 2. Verdict table

| Claim | Verdict |
|---|---|
| 1 compute ceilings | CONFIRMED |
| 2 bandwidth | CONFIRMED |
| 3 latency | CONFIRMED |
| 4 memory levels + adversarial | CONFIRMED |
| 5 meter tiers | CONFIRMED (cachegrind True) |
| 6 --reanalyse | CONFIRMED (forgery gap stated) |
| 7 labels | CONFIRMED (digests, vector_isa, sweep_reach) |
| 8 reproducibility | CONFIRMED (3/3 find L1+L2) |
| 9 portability | probe/counter/record fine; loop jail impossible |
| 10 repeatability | CONFIRMED (1 failure: Python hash seed, 1.3e-6) |
| 11 cross-host | CONFIRMED (same binary exact; different gcc differs) |
| 12 isolation | CONFIRMED |
| 13 --check | CONFIRMED (old-schema exits 1 not 2 — minor) |
| 14 simulated vs measured | CONFIRMED |
| record 1 monitor | exit 2 (GitHub 403; correct) |
| record 2 fake entry | CONFIRMED (all attacks caught) |
| record 3 anchor without record.py | CONFIRMED (sigstore OK; JCS sha256 matches) |
| record 4 exit codes | CONFIRMED |
| loop 1 re-measure plan 2 | CONFIRMED (unjailed; accuracies match exactly) |
| loop 2 read code | CONFIRMED clean |
| loop 3 attack scoring | NOT ATTEMPTED (jail impossible) |

## 3. Evidence

**Probe (1–9).** 40/40 tests (59 s). objdump: scalar = 29 `vfmadd132sd` xmm; vec_f64 = 16 `vfmadd132pd %zmm`; vec_f32 = 16 `vfmadd132ps %zmm`. Measured vs peak (2 FMA512 × 3.2 GHz): f64 101.8–103.2 vs 102.4; f32 204.8–206.9 vs 204.8. 3 seeds: all find L1 (1.40 ns, cap 44–64 KiB) + L2 (4.49–4.65 ns, cap 1.4M–null); L3/DRAM refused with `sweep_reach` warning (ratio 1.44 < 2). L2 upper bracket `null` as claimed. Adversarial (200 seeds/case): co-tenant-spike A2 → 0.5% false boundary, 12.5% wrong (matches README's documented limit); 5 new cases E1–E5 → 0% phantom, 0% false boundary. Nuclear tamper (inflate rates + adjust work params + add digests) still passes exit 0 — documented in profile caveats. Meter tiers: cachegrind True; all others correctly False.

**Counter (10–14).** 22/23 tests (1 failure: `test_python_hash_seed_is_fixed`, 1.3e-6 spread on ~48.5M Ir — the handoff's "unexplained failure"). 2 skipped (valgrind headers; replicated by hand). My mlp sha `45185f26` differs from README's `a330b376` (different gcc); Ir(1000)=35,781,414, Ir(0)=248,251, work=35,533,163; reps `exact=True`. `--check`: genuine→0, forged→1 (digest mismatch), old-schema-v2→1 (not 2 — minor). `sweep.py probeL1`: D1 miss jumps 0→1.0 at 64 KiB (L1 cap), LL miss jumps at ~2 MiB (L2→DRAM) — matches probe curve.

**Record (1–4).** `verify`→0 (4 entries, 2 plans, 2 verdicts, 0 open). `check-anchors`→0. `monitor`→2 (GitHub 403). Attacks: hand-written→"bytes aren't in canonical form"; JCS-canonical wrong seq→"seq out of order"; JCS-canonical correct seq+prev→"plan already has a verdict". All caught. Anchor without record.py: `sigstore verify identity` with GitHub Actions identity→"OK"; JCS sha256 via Python `rfc8785.dumps`=`9933d6766864afcc…` matches head; `ots info` confirms stamp covers same hash. Bitcoin chain needs a node (unavailable).

**Loop (1–3).** Neither jail works (bwrap: /proc mount refused; confine: Landlock ABI -1). Code read: all 5 `train.py` import only `sys`+`numpy`; all 5 `infer.c` only `stdio`/`stdlib`/`string`; no `getenv`/`/proc`/`cpuid`/`rdtsc`/`fork`/`dlopen`; mlp16/32/64 differ only in `H`. Retrained all 5 unjailed: **byte-identical** to committed weights. Inference on 1797 test images: linear 0.950473 (1708), thermo 0.959933 (1725), mlp16 0.959377 (1724), mlp32 0.965498 (1735), mlp64 0.961046 (1727) — **exactly matching** the verifier's verdict. Part A PASS; Part B INCONCLUSIVE (z=1.62/0.33). Claim 3 not attempted.

## 4. Ways it could report something false

1. **(Severe, documented)** Probe raw forgery: inflate rates, adjust iters/passes/steps to keep timed windows valid, add digests → `--reanalyse` exits 0. Stated in profile caveats.
2. **(Minor)** `counter --check` on old-schema record exits 1, not 2. Schema mismatch is "can't compare" but code treats it as "edited."
3. **(Minor)** `test_python_hash_seed_is_fixed` fails intermittently (1.3e-6 spread). Not security-relevant.
4. **(Environment)** Loop jail can't run in Docker/kata (bwrap needs unprivileged /proc; confine needs Linux 6.7+).
5. **(Environment)** GitHub rate limits make `monitor` exit 2 (correct refusal).

## Attachments
`full_{11,22,33}.json`, `quick.json`, `lscpu.txt`, `valgrind_version.txt`.
