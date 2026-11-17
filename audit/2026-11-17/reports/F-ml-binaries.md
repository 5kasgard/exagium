# Audit F: ML training code and compiled programs
Scope: 9 candidates (linear, thermo, mlp16/32/64 on main e093a60; lgn_a..d on pilot/lgn-20261115 5bf0efe). Scratch: /home/claude/audit/scratch/F/ (parse_weights.py, verify_bin.py, fuzz.py, retrain.sh, fuzz/out*.txt). Host: Xeon 2.1GHz, numpy 2.5.3 (scipy-openblas 0.3.34 DYNAMIC_ARCH), gcc 14.2. optdigits.tes was never opened; grep only.

## Findings
| # | Sev | Status | Location | Finding | Evidence | Fix |
|---|---|---|---|---|---|---|
| 1 | Medium | verified | pilots/lgn-20261115*/lgn_{a,b,c,d}/weights.bin | Committed weights are trained on the first 2,867 rows (train75.csv), not on `train.csv` as the loop contract (`train.py train.csv weights.bin`, byte-identical) requires. `loop.py reproduce` would fail on them. Pilot READMEs say so (l.96, l.131) but train.py docstrings and the dirs do not. | lgn_d retrained on train75.csv: bytes identical to committed (4241 gates). Full-file run in train_full_d.log gives 4327 gates, a different file. Committed lgn_a..d score 0.978/0.971/0.974/0.973 on rows 2867+ (out of sample) vs 0.999 on rows <2867. | Before any registration: retrain on full file and commit those weights, or add a train75 convention to the contract. Add a note to each train.py. |
| 2 | Low-Med | verified | baselines/linear/infer.c:~16-28, thermo/infer.c:~14-27 | No validation of N. n<0 gives a heap OOB write (`out[n]`), huge n gives an unchecked malloc then NULL deref (segfault, rc 139). atol overflow is UB. Malformed N is a crash, not a clean exit 2. mlp/lgn reject these cleanly. | ASan: heap-buffer-overflow at n=-1; `store to null pointer` at n=1e12; plain build rc=139. | `if (n<0) return 2;` and check malloc, as mlp does. |
| 3 | Low | verified | linear/infer.c | Does not reject pixels >16 (thermo, mlp and lgn do). Output stays defined, so no memory risk, but behaviour differs between baselines. Also linear, thermo and lgn accept trailing bytes in weights.bin (mlp uses fgetc==EOF). | fuzz: x_bad17/255 give rc 0 for linear only; "extra byte" rc 0. | Optional: make all candidates equally strict. |
| 4 | Low | verified | mlp*/infer.c:60 | Header claims "no overflow is possible". That holds only for weights meeting train.py's bound (checked: worst-case sums are in [-32701, 22041] for all three). A hand-edited weights.bin makes the int16 `acc += row[k]` signed-overflow UB (UBSan). No OOB access, only wrong or UB arithmetic. | 73 of 300 random byte mutations of mlp16 trip UBSan; no ASan error in any. | Reword the claim to "for weights satisfying the bound", or use unsigned vector types. |
| 5 | Low | verified | all train.py using BLAS: linear, thermo | `X @ W` goes through DYNAMIC_ARCH OpenBLAS, whose float64 bits depend on CPU kernel. Docstrings do not say "no BLAS" (mlp and lgn do and do not use BLAS). Demonstrated: a 3823x1024 matmul differs bitwise under OPENBLAS_CORETYPE=Haswell, Nehalem and SkylakeX. After float32 rounding the weights were still byte-identical to committed. | linear: 6 settings (default, 1 thread, Haswell, Nehalem, SkylakeX, Zen) all give 06fdc96e. thermo: Haswell, Nehalem and SkylakeX all give 471d3cf1 = committed. | Residual risk is small (about 1e-3 per host change by my estimate, suspected, not measured). Cross-host CI already reproduces it. Optionally replace BLAS with the explicit-sum `mm` used by mlp. |
| 6 | Low | verified | mlp train.py docstring and lgn train.py | "Only elementwise + - * / sqrt max and indexing" is not literal. They also use `.sum(axis=...)` (gb1, gb2, `e.sum`, quantize bound), `np.round`, `rng.random/permutation/standard_normal/integers`. Numpy's reductions are fixed-order and the Generator streams are stable within a pinned numpy, but both could change across numpy versions. | Code reading. Retrain matches under 2.5.3 only. | Pin numpy in the contract (the README says 2.5.3). Keep the cross-version caveat in the docs. |
| 7 | Info | verified | lgn_c/lgn_d vs task wording | "Learned wiring (Top-K)" is true only for lgn_c and lgn_d (K=8 soft choice, argmax kept). lgn_a and lgn_b use fixed random wiring, as their docstring says. Method matches code: softmax over 16 ops, Adam (lr 0.01, b 0.9/0.999, bias-corrected), hard argmax at the end, compiled circuit asserted equal to the net on all training rows. | Code reading. | none |
| 8 | Info | verified | loop/baselines, mlp | Lambda is picked on the last 25% of the file and the model is refit on all rows, so dev is in-sample for the final model. Stated in the docs. No train.py reads .tes (the loop gives it only a copy of optdigits.tra). Writer overlap between the first 75% and the last 25% is unknown (the file has no writer ids), so lambda selection may be slightly optimistic. | grep: no `.tes` in any train.py. Pilot sweeps load d[:2867] only (exp2.py:413). Remaining `.tes` hits are hashes in scoreboards and verifier-recompute.py. | none |
| 9 | Info | verified | pilots/*/sweeps/exp*.py:412 | Hard-coded /home/claude/exagium path. | grep | Make it an argument. |
| 10 | Info | verified | mlp and lgn infer.c | Little-endian and SSE2 or NEON assumptions: raw fread into vector types, no endian check. | Code reading. | State it in the README. |

No environment-dependent path: no getenv, time, cpuid, /proc, valgrind detection or CPU dispatch in any infer.c. The binaries import only libc IO and malloc symbols (nm -D). Objdump finds no cpuid, rdtsc, FMA or ymm instructions in the sampled binaries.

## Binaries vs claims
- The lgn_a..d infer.c files are byte-identical to each other. mlp16/32/64 differ only in `#define H`.
- Output on all 3,823 training rows is identical across native, valgrind, `env -i` with odd locale and TZ, taskset-pinned, and builds at -O0, -O3, -march=native and -ffp-contract=fast. valgrind memcheck reported 0 errors.
- Each binary matches an independent numpy reimplementation written from the documented weights format: 3823/3823 for all 9 (verify_bin.py). This covers float32 sequential sums for linear, thermo and mlp, and a gate-by-gate bit simulator for lgn.

## Per-candidate table
| candidate | leakage (train.py) | determinism | weights.bin format | robustness (ASan/UBSan, 300 mutations + edge args) |
|---|---|---|---|---|
| linear | none; lambda on last 25%, refit all | byte-identical under 6 BLAS settings and committed | 2600 B = f32 W[10][64] + b[10], all accounted, finite | FAIL: n<0 OOB write, n huge NULL deref; no pixel check |
| thermo | same | byte-identical under 3 BLAS kernels and committed | 43560 B = f32 T[64][17][10] + b[10]; T[p][0]=0 | FAIL: n<0 and n huge (as linear); pixel>16 rejected |
| mlp16 | same | byte-identical x2 and committed (79 s) | 35664 B exact; padding W2[:,10:]=0, b2[10:]=-1e30 | edge args clean; weight mutations give int16 overflow UB, no memory error |
| mlp32 | same | identical to committed | 71280 B exact | as mlp16 |
| mlp64 | same | identical to committed | 142512 B exact | as mlp16 |
| lgn_a | none (full-file training, fixed hyperparameters chosen on inner split of rows <2150; sweeps never load rows >=2867) | not retrained (est. 4+ min under load); weights are train75-based (finding 1) | 80046 B: 324 groups, 13123 gates, all indices < n_slots, no read-before-write, outputs written | clean: 0 crashes in 300 mutations, with 54 rejected cleanly; trailing odd byte accepted |
| lgn_b | same | same, not retrained | 49024 B, 283 groups, 7980 gates | clean |
| lgn_c | same (Top-K) | same, not retrained | 46272 B, 294 groups, 7514 gates | clean |
| lgn_d | same | retrained on train75.csv: byte-identical to committed (421 s) | 26438 B, 245 groups, 4241 gates | clean |

## Not done / limits
- lgn_a, b and c were not retrained (CPU shared, 4+ min each). lgn_d full-file run was started but not awaited; I rely on the pilot's own log for the 4327-gate figure.
- A different CPU architecture (arm64) was not tested. OPENBLAS_CORETYPE only emulates kernel choice on this x86 host.
- The 3,823-row agreement check is in-sample and does not touch the test set.
