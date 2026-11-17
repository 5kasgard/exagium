# Audit D: research and results claims against raw data

Date 2026-11-17. Auditor: Claude Opus 5.5 (one agent). **Self-review, not independent:** no sub-agent tool was available in this session, so the adversarial-review skill's fresh second reviewer was not run. A same-family reviewer would be direction only anyway; the owner may want an outside-family pass on the mismatch list below.

Scope: every doc on `pilot/lgn-20261115` (a superset of `main`: main + 7 commits), raw data from that branch, `ci-runs` and `router-runs`. Other branches (`anchor-test`, `e2e-main`, `q2-*`) are older states with no research claims beyond the record/counter cards.

**How to reproduce:** `sh /home/claude/audit/scratch/D/run_all.sh` (output kept in `scratch/D/run_all.out`, exit 0, 0 stored-vs-recomputed disagreements in any scoreboard or bench JSON). Scripts:
- `sb.py`: recomputes every exagium-loop/2 scoreboard figure (accuracy, SEs, Ir/img, marginal Ir, ns mean ± SE from the 60 per-round values, paired ns diffs from rounds, D1 misses with startup excluded).
- `records_check.py`: counter-record sha256 and counts against each scoreboard.
- `explore_check.py`: medians, IQRs and paired per-round differences from `raw_ns` in explore results.
- `plan2_check.py`: the plan-2 decision rule (ledger entry 2), per host, from the 7 scoreboards; evidence hashes of entry 3.
- `ci_provenance.sh`: committed plan-2 CI scoreboards against `ci-runs:loop/42172779341/*/test-plan2.json`.
- `sweep_match.py`: counter/README model-vs-measured level matches.
- `bench12_check.py`: bench-1/2 tables from `router-runs` result.json.
- `status_check.py`: bench-3/4/5 STATUS recomputed with the repo's own `decide.py` `status()` (no files written).

## Counts by status (84 claims)
| status | n |
|---|---|
| match | 57 |
| rounding ok | 5 |
| mismatch | 5 |
| missing evidence | 9 |
| config mismatch | 2 |
| scope overclaim | 6 |

Every ledger verdict (entries 3 and 4) recomputes or checks out. No headline verdict changes. The problems are secondary numbers, one wrong reading of a cited paper, one stale status file, and "ruled out" wording stronger than the data.

## Mismatches and overclaims (each with evidence and a suggested correction)

**M1. mismatch** `loop/README.md:132` "Natively it is also 17–31% faster than both baselines on every host". Recomputed `1 − ns(mlp16)/ns(B)` over the 7 scoreboards: **14.5–33.1%** (local Xeon vs linear 14.5%; Xeon 8573C vs linear 33.1%); 16.0–33.1% on the 6 CI hosts. No formula I tried gives 17–31. "Faster on every host past 2 SE" does hold (smallest: local vs thermo, 2.4 SE). *Fix:* "15–33% faster (16–33% on the six CI hosts)". The outside audit (`meter-audit-2026-11-11`) flagged this sentence but only the startup-removed half was corrected.

**M2. config mismatch** `loop/README.md:132` "not the 3.5–5.4× its instruction count suggests" is set beside a *startup-removed* speed-up (1.8–2.6×), but 3.46–5.40 are the *startup-included* Ir ratios. The like-for-like (marginal) Ir ratios are **3.65–5.91×** (x86-64 5.51–5.91, arm64 3.65–3.98). *Fix:* "not the 3.7–5.9× its startup-removed instruction count suggests".

**M3. scope overclaim** `loop/README.md:74-82` "Two meters" table and "An MLP instruction took 1.9–2.6× as long". The EPYC column is `ubuntu-24.04-1` and the arm64 column is `ubuntu-24.04-arm-1`, exact matches, but neither is labelled, and 4 other CI hosts are left out. Over all six CI hosts the per-instruction ratio is **1.7–2.8×** (ubuntu-24.04-2: up to 0.224/0.080 = 2.8; arm-3: down to 0.138/0.080 = 1.73). *Fix:* name the two hosts, and give the six-host range.

**M4. scope overclaim (stale)** `loop/README.md:84-87` "Instructions per nanosecond: the leading hypothesis… A plausible cause is chains of dependent table lookups". On the same branch, pilot 2 (finding 1) and `explore/mlp16-speed/BRIEF.md` rule out dependency chains, and BRIEF's H10 shows page faults on the input buffer were ~a third of mlp16's steady-state time, invisible to the instruction meter. The Claude report says that finding "belongs in the loop spec card's limits"; it isn't there. Also the "branch mispredictions 3.7 vs 4.1" line adds linear's 2.0 *indirect* mispredicts per image (glibc calls) to its 2.1 conditional ones. The branch-sim raw output exists only inside `meter-audit-2026-11-11/prompt-as-built.txt`, not as a data file. *Fix:* update "What explains it" on the pilot branch, or link it to BRIEF's status, and commit the branch-sim output.

**M5. mismatch** `pilots/lgn-20261115-2/README.md:113` "LILogicNet's 7-vs-4 comparison is on CIFAR-10, and 7 thresholds beat 4 in every pair there, by 0.1–1.1 points". Also in `lgn_c/train.py:5-6` and `lgn_d/train.py:5-6`: "lost 0.1-1.1 points going from 7 thresholds to 4". In arXiv 2511.12340v2 Table 8 there are 14 pairs with both values. 7 beats 4 in **13 of 14**, by **0.02–1.12** points. In 1F-8,000, 4 thresholds won (44.18 vs 44.08), and 1L-64,000 differs by only 0.02. *Fix:* "7 beat 4 in 13 of 14 pairs, by 0.02–1.1 points; 4 won one (1F-8K) by 0.1". The pilot's conclusion ("4 thresholds cost nothing measurable here") is unaffected.

**M6. mismatch** `explore/mlp16-speed/BRIEF.md:90` H9: "mlp16 about 3.2 mispredicts/img … against linear's about 0.002 conditional". mlp16's 3.21 and 8.8% match `reference/mlp16.cachegrind-summary.txt`. Linear's own summary gives **198,931 conditional mispredicts / 95,600 = 2.08 per image** (191,270 of them in `infer.c:main`), plus 2.0 indirect. Only linear's inner `c` loop is near zero (16 in total). *Fix:* "linear about 2.1 conditional (only ~0 in its inner loop)". H9 is ruled out by ablation anyway, but the premise as written overstated the contrast by ~1000×.

**M7. scope overclaim** `BRIEF.md:4-10` "Ruled out by at least two independent runs: … code size (H5); VM noise (H7)". Only GLM ran H5 (`h5_u1`) and H7 (base-1 vs base-2). Gemini's report is chat-only static analysis, and the Claude report tests neither. *Fix:* split the list into "two runs" (dependency chains, H9, H4) and "one run" (H5, H7).

**M8. scope overclaim** `BRIEF.md:8` "address dependence (H2)" ruled out, and `reports/claude-2026-11-15.md:37` "Removing the dependence saves about 2 ns, inside the IQRs". That 2.1 ns is a difference of unpaired medians. Per round, from `layer1-split-xeon.json`:
- `mlp16_l1only − mlp16_l1_fixrow_indep` = **+3.8 ns** (IQR 0.8–6.2), positive in **18/21** rounds;
- `fixrow_dep − fixrow_indep` = **+8.6 ns** (IQR 5.0–10.3), **20/21** (confounded: +155 Ir/img).

GLM's `h2_pre` adds 395 Ir/img, so its slowdown is confounded too. The paired data show a small, consistent effect (about 5% of mlp16's time), not none. *Fix:* "no large effect; at most ~4 ns/img (~5%), unresolved because the variants change Ir". Also consider paired per-round differences in BRIEF rule 2, as the staging report already uses: the IQR-overlap rule can call a consistent effect noise.

**M9. scope overclaim (stale)** `README.md:43` "why mlp16 retires about 2.4× fewer instructions per ns than linear". This matches `reference-xeon-2.3.json` (21.54/8.87 = 2.43). But BRIEF's own status says that with I/O removed the gap is 1.7×, gcc 14.2 gives 1.86× (GLM), and the BRIEF status itself says "not 2.6×" (another session's baseline) beside its table's 2.4×. *Fix:* "about 2.4× on the reference host with gcc 13.3; about 1.7× once I/O is removed (BRIEF status)".

**M10. config mismatch (honest-views)** `BRIEF.md:17-21`, the H10 follow-up (32 ns, 46%, 2.0×), is from `staging-*-xeon.json`: **Xeon @ 2.10 GHz, kernel 6.18.44-fc-v553**, clock 2.69–2.83 GHz. The puzzle table above it is **2.30 GHz, kernel v470**, clock ~3.8 GHz. The staging report says "different VM"; the BRIEF status doesn't. *Fix:* add "(another VM: 2.10 GHz, ~2.8 GHz measured; compare only within a run)".

**M11. mismatch (stale status, honest-views)** `router-runs:bench/bench5/STATUS.md` (generated 2026-11-15 13:22) says "bench-3 arm A complete: False; verdicts final: False; P … A 9/11; Fisher p = 1.0". bench-3 finished on 2026-11-16 07:46. Recomputed from raw with the repo's `bench5/decide.py`: **A 11/15, p = 0.694, a_complete True, final True**, and A median calls 39 (not 38). The verdicts don't change (C fails, P holds), but the status page shows "Not final" for a result that is now final. *Fix:* regenerate bench-5 STATUS (or have the bench-3 run trigger it).

**M12. scope note (honest-views)** `README.md:39` status row "lgn_d uses 0.45× mlp16's instructions and 0.63× its time at steady state". This matches `timing/steady3.json` (291.4/644.6, 52.7/84.3). But it is one Python-harness session with medians and no SE (the pilot README says so; the root row doesn't). Both programs also carry the H10 input-buffer fault cost. With each program's I/O fixed (staging-full, another host), the ratio is 36.9/73.7 = **0.50**; with only mlp16 fixed it is 68.4/73.7 = 0.93. *Fix:* add "(medians, one session, I/O-sensitive; see staging report)".

**M13. mismatch (minor)** `reports/staging-2026-11-17.md:12,14` page faults "1,663" and "169–424". `results/staging-evidence.json` holds 1,662 and 168–425 (read: 168–425; the report says 167–424). The report says faults are medians of 5 runs, but the JSON has one value per program. *Fix:* commit the 5-run values, or quote the JSON's.

**Missing evidence (claims with no committed raw source)**
- `pilots/lgn-20261115/README.md:87`: lgn_b full-file 8,073 gates, 4 min 22 s. `reproduce.log` has only the reproduction (identical weights, 4 min 14.5 s).
- `pilots/lgn-20261115/README.md:69-73`: reviewer's gate breakdown (4,730 / 425 / 2,360 / 7,515 of 13,123). The arithmetic is consistent (57.3%), but no log is committed.
- `pilots/lgn-20261115/README.md:93`: `ctest.py` and its 0.862 dev printout are not committed. This is the one disclosed dev-exposure channel, so it is worth keeping.
- `pilots/lgn-20261115-2/README.md:49-50`: reviewer's page-fault counts (93 extra; 53/31/15) and fixed costs (141/41 µs).
- `pilots/lgn-20261115-2/README.md:67`: 11,967 gates after NOT folding. The Ir (730.7→690.5) and steady-state ns (90.0→86.3) match. Not stated: at task size, NOT folding was not faster (+33 ± 64 ns, `not_folding/scoreboard-dev.json`).
- `BRIEF.md:49`: clock 3.85/3.80 GHz (no committed `bin/clock` output for the 2.30 GHz host; the Claude report says 3.79–3.84).
- `reports/claude-2026-11-15.md:61-67`: compute-only Ir (3,667; 592) and null-program Ir (56): `io-split-xeon.json` and `layer1-split-xeon.json` have no `ir_per_img`. The ns figures match: 163.4 and 46.1 paired.
- `reports/claude-2026-11-15.md:71`: the fresh-clone repeat (31.2 ns, 1,594 vs 116 faults) has no committed result.
- `loop/README.md:85`: model-error drop 30–46% → 22–32%. No script or output is committed; the outside auditor recomputed it from numbers the builder supplied.

## Claim table
Paths are on `pilot/lgn-20261115` unless noted. "rec" = value recomputed by script (see `run_all.out`).

| id | location | claimed | raw source | recomputed | status |
|---|---|---|---|---|---|
| 1 | README.md:38 | lgn_b 0.85× mlp16 Ir, +72 ± 25 ns/img | pilots/lgn-20261115/scoreboard-dev.json | 0.849×, +72.3 ± 25.1 (paired, 60 rounds) | match |
| 2 | README.md:39 | lgn_d 0.45× Ir, 0.63× time, steady state | pilots/lgn-20261115-2/timing/steady3.json | 0.452, 0.625 (medians, no SE) | match (see M12) |
| 3 | README.md:39 | task size 0.66× Ir, −27.9 ± 14.9 ns | pilots/lgn-20261115-2/scoreboard-dev.json | 0.661, −27.9 ± 14.9 (1.88 SE) | match |
| 4 | README.md:39 | accuracy −0.73 ± 0.50 pt | same | −0.73 ± 0.50 | match |
| 5 | README.md:36 | plan 1 judged pass (entry 4) | ledger/entries/000004.json, verifications/z-20261106 | outcome pass; 9/9 evidence sha256 match | match |
| 6 | README.md:37 | plan 2 inconclusive (entry 3) | loop/results/plan2/*.json | A PASS, B INCONCLUSIVE on all 7 hosts | match |
| 7 | README.md:43 | mlp16 about 2.4× fewer Ir/ns than linear | explore results/reference-xeon-2.3.json | 2.43× (one host, gcc 13.3) | scope overclaim (M9) |
| 8 | pilots/lgn-20261115/README.md:27 | inner split thermo 0.972, mlp16 0.976 | sweep.md | 0.9721, 0.9763 | match |
| 9 | same:32-33 | inner acc lgn_a 0.979/0.982, lgn_b 0.974 | sweep.md | 0.9791/0.9819, 0.9735 | rounding ok |
| 10 | same:42-48 | dev table, 7 rows (acc, Ir, marginal Ir, ns ± SE, marginal ns) | scoreboard-dev.json + records/ | all equal; 14/14 record sha256 and counts match | match |
| 11 | same:56-59 | paired vs mlp16 (−0.21 ± 0.47 … +6 ± 19) | same | all equal, recomputed from rounds | match |
| 12 | same:62 | 0.18–0.38× thermo Ir; every LGN acc within 1 SE of thermo | same | 0.181–0.383; max abs(d)/SE = 0.74 | match |
| 13 | same:63-64 | vs thermo −36 ± 20, +36 ± 25, −65 ± 19, −103 ± 17 | same | −36.1 ± 20.3, +36.0 ± 25.3, −65.1 ± 18.9, −102.6 ± 16.7 | match |
| 14 | same:76 | D1 misses/img, startup excluded: 50, 27, 2.7 | same (counts) | 50.18, 27.22, 2.73 | match |
| 15 | same:78 | KV=2 roughly halves marginal ns (238→130), adds 60% Ir | same | 237.8→130.1; Ir +62% (lgn_a), +50% (lgn_b) | rounding ok |
| 16 | same:80 | Petersen 625 ns/MNIST image from compiled C, operands in registers | arXiv 2210.08277 Table 4, App. A | 625 ns = small model (97.69%), 1 thread at 2.5 GHz, compiled C, 64 images per int64; "registers" not stated | match (partly supported) |
| 17 | same:85 | two lgn_a seeds differ by 0.28 pt | sweep.md | 0.9819 − 0.9791 = 0.28 | match |
| 18 | same:87 | lgn_b full file 8,073 gates, 4 min 22 s; reproduce identical | reproduce.log | identical weights ✓; 4m14.5s is the repro; 8,073 / 4:22 absent | missing evidence |
| 19 | same:69-73 | 7,515 of 13,123 gates (57%) bookkeeping | none committed | 57.3% arithmetic only | missing evidence |
| 20 | same:93 | ctest.py printed 0.862 on rows incl. dev | none committed | — | missing evidence |
| 21 | lgn-20261115-2/README.md:6,27 | interleave 74–84 vs 76.5 ns, IQRs overlap | mlp_interleave/bench_nb.txt | 74.3–84.4 vs 76.5 | match |
| 22 | same:6 | interleaving costs 3–33% more Ir at steady state | explore results/reference-xeon-nb-sweep.json | NB≥2: +3.7% … +33.4% (not in pilot dir) | rounding ok |
| 23 | same:6 | NB=4 slower by 111 ± 30 ns at task size | mlp_interleave/sb1.json | +111.5 ± 29.7 | match |
| 24 | same:9-10 | lgn_c 0.9833 at 72% of 10,187 gates; lgn_d 0.9791 at 41% | sweeps/k5.log, k6.log, t3.log | 7,344/10,187 = 72.1%; 4,197 = 41.2% | match |
| 25 | same:34 | Ir/ns steady: 18.6, 16.1, 7.0, 7.5, 7.0 | timing/ipc.json | 18.55, 16.11, 6.96, 7.53, 7.04 | match |
| 26 | same:39-46 | run-size table (6 rows × 3) | timing/multiN3.json, steady3.json | all equal; lgn_d 9,560 is 60.5 (shown 61) | rounding ok |
| 27 | same:53 | mlp16 sessions 76, 84, 90, 93 ns | bench_nb, multiN3, multiN2, multiN | 76.5, 84.3, 90.4, 92.6 | match |
| 28 | same:49-50 | +93 faults; 53 faults/141 µs; 31, 15 faults/41 µs | none committed | — | missing evidence |
| 29 | same:60-63 | thresholds table: 11,847/0.9791, 10,415/0.9791, 10,187/0.9805 | sweeps/t0,t1,t3.log | equal | match |
| 30 | same:67 | NOT folding 13,123→11,967 gates; 731→691 Ir; 90→86 ns | not_folding/scoreboard-dev.json, timing/multiN2.json | 730.7→690.5; 90.0→86.3; gate count absent; task-size time +33 ± 64 not stated | missing evidence |
| 31 | same:71-78 | Top-K table, 6 rows | sweeps/k1–k6.log | equal | match |
| 32 | same:85-92 | dev table, 6 rows | scoreboard-dev.json + records/ | all equal; 12/12 records match | match |
| 33 | same:98-100 | paired: lgn_d −27.9 ± 14.9, −144.2 ± 19.7; lgn_c +74.0 ± 26.7 | same | equal | match |
| 34 | same:102 | inner lead 0.7/0.3 → dev trail 0.6–0.7, within ~1.5 SE | k5/k6 logs, scoreboard | +0.70/+0.28 → −0.63/−0.73 (1.24/1.46 SE) | match |
| 35 | same:106 | mlp16 moved ~51 ns between runs | sb1.json vs scoreboard-dev.json | 1298.2 − 1246.9 = 51.3 | match |
| 36 | same:112 | lgn_d full 4,327 gates, 4 min 18 s, reproduce identical | train_full_d.log, reproduce_d.log | equal | match |
| 37 | same:113; lgn_c,lgn_d train.py:5-6 | LILogicNet 7 thr beat 4 in every pair, by 0.1–1.1 | arXiv 2511.12340v2 Table 8 | 13/14 pairs, 0.02–1.12; 4 won 1F-8K by 0.10 | mismatch (M5) |
| 38 | same:114,118 | LILogicNet 98.45% MNIST with 8k gates; counts exclude popcount | arXiv 2511.12340v2 abstract, Table 3 | 98.45 ± 0.07, "1L-8K" (dense learnable, 10× augmentation); popcount separate | match |
| 39 | BRIEF.md:41-45 | puzzle table (5 programs) | results/reference-xeon-2.3.json (raw_ns) | equal | match |
| 40 | BRIEF.md:47 | 1/6 the Ir, 2.4× faster | same | 0.169; 2.44 | match |
| 41 | BRIEF.md:49-51 | clock 3.85/3.80 GHz; 670/275 cycles; IPC 5.7/2.3 | no clock output for this host | arithmetic 673/276, 5.67/2.34 | missing evidence |
| 42 | BRIEF.md:59-61 | NB sweep 73–86 vs 79; interleave adds 0–33% Ir | reference-xeon-nb-sweep.json | 73.2–85.8 vs 79.3; +0.06% … +33.4% | match |
| 43 | BRIEF.md:69 | L1d misses 2.1/img for mlp16 | reference/mlp16.cachegrind-summary.txt | 200,550/95,600 = 2.10 | match |
| 44 | BRIEF.md:90 | mlp16 3.2 mispredicts/img (8.8%) vs linear about 0.002 conditional | reference/*.cachegrind-summary.txt | mlp16 3.21, 8.8% ✓; linear 2.08 conditional/img | mismatch (M6) |
| 45 | BRIEF.md:4-10 | H5, H7 ruled out by ≥ 2 independent runs | reports/*.md, glm results | H5 and H7 tested by GLM only | scope overclaim (M7) |
| 46 | BRIEF.md:8 | H2 ruled out | layer1-split-xeon.json, glm-h1h2.json | paired +3.8 ns (18/21), +8.6 ns (20/21); GLM confounded | scope overclaim (M8) |
| 47 | BRIEF.md:13 | page faults 28–31 ns/img, about a third of mlp16 | io-split-xeon.json | paired 28.5 and 30.4; 28.5/84.5 = 0.34 | match |
| 48 | BRIEF.md:15 | compute-only 22.4 vs 12.8 Ir/ns, 1.7× not 2.6× | io-split-xeon.json | ns: 163.4, 46.1 ✓; compute Ir not in raw; "2.6×" vs own 2.4× | missing evidence |
| 49 | BRIEF.md:17-21 | staging: −32 ns, mmap ties within 3 ns, −46%, 2.0× | staging-full-xeon.json | 32.0–32.5; −1.1 (9/21); 46.1%; 1.998 — on a different host, unlabelled | config mismatch (M10) |
| 50 | reports/claude-2026-11-15.md:37 | H2: removing dependence saves about 2 ns, inside IQRs | layer1-split-xeon.json | unpaired 2.1 ✓; paired +3.8 (18/21) | scope overclaim (M8) |
| 51 | same:59 | mlp16 − chunked 28.5 ns (IQR 21.6–32.8); 32% faster | io-split-xeon.json | 28.48 (21.59–32.79); 32.2% | match |
| 52 | same:71 | fresh-clone repeat 31.2 ns, 1,594 vs 116 faults | none committed | — | missing evidence |
| 53 | same:101 | layer 1 = 71% (GLM), about 78% (Claude) | glm-h3.json, gemini-variants-xeon.json | 71.5%, 78.4% | match |
| 54 | reports/staging-2026-11-17.md:10-31 | staging table (16 programs) and 5 paired rows | staging-xeon.json, staging-evidence.json | all ns and paired values equal | match |
| 55 | same:12,14 | page faults 1,663; 169–424 | staging-evidence.json | 1,662; 168–425 | mismatch (M13) |
| 56 | same:38-50,61-62 | full classifiers, 4 paired diffs, 46%/36% wall, 2.0×/2.9× | staging-full-xeon.json | all equal (wall cut 35.6%) | match |
| 57 | reports/glm-20261115/REPORT.md §2-3 | base table, h1–h9 medians, Ir; 1.86× | glm results/*.json | all equal; 17.97/9.68 = 1.86 | match |
| 58 | loop/README.md:120-126 | plan-2 test accuracy and Ir/img per arch | loop/results/plan2/*.json | equal on all 7 hosts; predictions identical | match |
| 59 | same:128 | mlp16 0.19–0.29× baseline Ir; 0.06 pt below thermo | same | 0.185–0.289; 0.056 | match |
| 60 | same:129 | z 4.3/3.5; +0.56 (z 1.62), +0.11 (z 0.33); holdout ~+1.1 | same; entry 3 notes | 4.34/3.54/1.62/0.33; holdout +1.05/+1.15 | match |
| 61 | same:132 | 17–31% faster than both baselines on every host | same | 14.5–33.1% (7 hosts) | mismatch (M1) |
| 62 | same:132 | 520 ± 5 vs 706 and 711 on EPYC | ci/ubuntu-24.04-1.json | 520 ± 5, 706 ± 7, 711 ± 5 | match |
| 63 | same:132 | startup removed 1.8–2.6× on six CI hosts (2.3× EPYC) | ci/*.json | 1.80–2.63; 2.35/2.30 | match |
| 64 | same:132 | "not the 3.5–5.4× its instruction count suggests" | same | that is startup-included; marginal 3.65–5.91 | config mismatch (M2) |
| 65 | same:74-80 | two-meters table (10 cells) | ci/ubuntu-24.04-1.json, ci/ubuntu-24.04-arm-1.json | all 10 equal | match |
| 66 | same:82 | MLP instruction 1.9–2.6× a baseline one | ci/*.json | 1.9–2.6 on the 2 shown hosts; 1.7–2.8 on 6 CI hosts | scope overclaim (M3) |
| 67 | same:82 | mlp64 0.43× (0.65× arm) linear Ir, slower than linear or thermo past 2 SE on all 7 | all 7 scoreboards | 0.433/0.652; smallest worst-case t = 2.15 (local vs linear) | match |
| 68 | same:85 | mlp16 ~2.4 misses/img; model error 30–46% → 22–32% | ci/*.json counts | 2.37 ✓; model-error fit not committed | missing evidence |
| 69 | same:86-87 | mispredicts 3.7 vs 4.1, 14.6; dependent-lookup chains the leading hypothesis | meter-audit prompt raw block | numbers match the prompt (cond + indirect); hypothesis superseded on this branch | scope overclaim (M4) |
| 70 | same:87 | EPYC linear ~11 Ir/ns, MLPs 4.4–4.7 | ci/ubuntu-24.04-1.json | 11.2; 4.39–4.67 | match |
| 71 | ledger 000003 | verdict inconclusive; A PASS, B INCONCLUSIVE all hosts; counts 1708/1725/1724/1735/1727 | plan2 JSONs | equal; 15/15 evidence sha256 match | match |
| 72 | ledger 000003 notes | CI run 42172779341 at 3be61b2; 6 CI hosts valid | ci-runs:loop/42172779341 | 6/6 committed CI JSONs byte-identical to ci-runs; commit.txt = 3be61b2 | match |
| 73 | ledger 000004 | plan 1 pass; evidence as labelled | verifications/z-20261106/artifacts | 9/9 sha256 match; deleted-.ots weakness (verify and check-anchors stay at 0) judged "not a false 0" | match (judgement noted) |
| 74 | verifications/z-20261106/INDEX.md | every sha256 matches; plan 2 remeasure matches | artifacts/ | sha256 equal | match |
| 75 | verifications/z-20261112/INDEX.md | five test accuracies match | plan2 JSONs | equal | match |
| 76 | probe/README.md:101-104 | machine C: L1 1.57 ns 32–64 KiB, L2 4.41 ns to 724 KiB, DRAM 10.0/17.8 GB/s, L1 174 GB/s, 12.6/79.4 GFLOP/s, 7.8× sweep, digest cda1c740…3df7 | probe/profile.json | equal | match |
| 77 | counter/README.md:115-118 | host model 26 of 32, reference model 31 of 32 (miss at 8 MiB) | counter/experiments/sweep_*_7.json | 26/32, 31/32 (miss 8,388,608 B) | match |
| 78 | router/bench/RESULTS-1.md:14-26,40,44 | per-run table; median 37 actions; 27 attempts/day; 342–367 MB | router-runs bench1-42974778781 | equal | match |
| 79 | RESULTS-1.md:6 | control failed all 3 | same (Z hidden fail/error) | fail, error, error | match |
| 80 | router/bench2/RESULTS-2.md:15-44 | table; seal; H2 holds; median 47; 21/day; 0.28–2.24M prompt tokens | router-runs bench2-42983345686 | equal | match |
| 81 | RESULTS-2.md:9 | North Mini fetched two GitHub pages (sigstore-js) | T1-D result.json, events | `web_fetches` lists 6 URLs; events show 2 `browser_navigate` | rounding ok |
| 82 | router-runs bench3/STATUS | complete; A 11/15 … ; ranking none | raw results + bench3/decide.py | identical to committed STATUS.json | match |
| 83 | router-runs bench4/STATUS | 35 counted; switch candidates goose, pi, opencode; no switch | raw + bench4/decide.py | identical | match |
| 84 | router-runs bench5/STATUS | A 9/11, p 1.0, not final | raw + bench5/decide.py | now A 11/15, p 0.694, a_complete True, final True | mismatch (M11) |

(Status counts at the top are tallied from this table. M12 and M13 refer to rows 2 and 55.)

## Citation audit (D2)
Every arXiv ID in the repo, on all 8 branches (`git grep`): 2210.08277, 2511.12340, 2611.09024, 2607.22585, plus the UCI DOI. **2605.04109 does not occur anywhere in the repo** (all branches, all file types); it may be in a project doc outside this repo. All were opened via alphaXiv in this session.

| cited | repo text | exists / metadata | supports the sentence? | verdict |
|---|---|---|---|---|
| arXiv 2210.08277 | "Petersen et al. 2022" (pilot READMEs, all train.py) | Yes: "Deep Differentiable Logic Gate Networks", Petersen, Borgelt, Kuehne, Deussen, NeurIPS 2022. Metadata right. | Fixed random wiring, softmax over 16 ops, group sum / τ: **supports**. "625 ns per MNIST image from compiling each net to straight-line C, with operands in registers": 625 ns is Table 4 (small model 97.69%, 1 CPU thread at 2.5 GHz); App. A says nets compile to C binaries; §4.2 says 64 images per int64. "Operands in registers" is not stated: **weak on that clause**. | keep; fix wording: "(small model, 97.69%, one thread at 2.5 GHz, 64 images per int64 word)" and drop "operands in registers" or mark it as inference |
| arXiv 2511.12340 | "LILogicNet", Top-K K=8, thresholds, 98.45% with 8k gates | Yes: "LILogic Net: Compact Logic Gate Networks with Learnable Connectivity…", Fojcik, Zioma, Armaitis; v2 26 Apr 2026. Repo gives no authors or year (fine). | Top-K learned wiring: supports (paper sweeps K = 2–128; K=8 is the pilot's own choice). 98.45% / 8K gates, counts without popcount: supports, but the 8K MNIST model is "1L" (dense learnable, not Top-K) with 10× augmentation; worth saying when comparing with lgn_d. **7-vs-4 thresholds "every pair, 0.1–1.1": wrong** (M5). | fix (M5); keep otherwise |
| arXiv 2611.09024 | "Teacher distillation (Suh & Jha)" | Yes: "Knowledge Distillation from Neural Networks to Differentiable Logic Networks", Hee Yun Suh, Niraj K. Jha (Princeton), v1 6 Nov 2026. Metadata right. | Supports "teacher distillation for logic networks". Scope: 20 tabular datasets, DLNs with learned connections (Yue & Jha), MLP teacher; no image tasks. | keep; optionally add "(tabular DLNs)" |
| arXiv 2607.22585 | router/bench4/PREREG.md:8 (+ review prompt): "A published study ('The Scaffold Effect', ICML 2026 workshop)… up to 40× tokens per solved task between Goose, OpenCode and OpenHands SDK, pass rates within 0–8 points" | Yes: "The Scaffold Effect in Coding Agents…", Vats & Golev (Sentient Labs), v1 8 Jun 2026. **Metadata wrong:** the paper says "Preliminary work. Under review at the 5th DL4C Workshop @ ICML 2026"; it is a preprint, not "published". | 40× and 0–8 pp: **supports** (Goose cheapest, OpenCode ~40× more; Qwen 3.6 Plus and MiniMax M2.5 on 50 Terminal-Bench Pro tasks). Note for reading bench-4: the paper found OpenCode the *most* token-hungry; bench-4 so far finds it the *least* (0.18×), which suggests token accounting or version differences. Don't carry the paper's direction over. | fix metadata ("a preprint under review at the DL4C workshop, ICML 2026"); PREREG is registered, so add an erratum rather than editing it |
| doi:10.24432/C50P49 | loop/task/SOURCE.md: Alpaydin & Kaynak (1998), UCI | Resolves to archive.ics.uci.edu/dataset/80 (optdigits); the page names Alpaydin and Kaynak. | supports | keep |

## Benchmark-design check (weigh-the-evidence)
- **loop plan 2 (strongest evidence in the repo):** 7 hosts, 30 interleaved rounds, paired per-round SEs, predictions identical across hosts, a pre-registered rule, an outside verifier, and CI outputs byte-identical to the committed copies. Accuracy SEs treat images as independent although there are 13 writers (disclosed). Rankings are fair; the verdict isn't over-read.
- **Pilots 1 and 2:** same host, same `evaluate`, 60 interleaved rounds, paired. Weak points, all disclosed in the pilot READMEs but not in the root status row:
  - dev only, one host;
  - session drift (mlp16 moved 51 ns between runs, more than the 27.9 ns gap);
  - one training seed;
  - "best 2 of 6" selection on 717 rows.

  The steady-state ratios (0.45×/0.63×) come from a Python whole-process harness, medians with no SE, one session, and are sensitive to input staging (M12). Read lgn_d as "promising, unregistered", which is how the pilot README words it.
- **explore/mlp16-speed:** BRIEF rule 2 (unpaired IQR overlap) is conservative for claiming a cause but **anti-conservative for "ruled out"**: H2 shows a consistent 18/21-round effect that the overlap rule calls noise (M8). The staging report already uses paired per-round differences with a stated detection limit; apply the same rule to ruled-out claims. Results from three hosts (2.30 GHz v470, GLM's 5.10 kernel with gcc 14.2, 2.10 GHz v553) are each compared only within a run, which is correct, but the BRIEF status mixes two hosts without labels (M10).
- **Free-model benches:**
  - bench-1/2: n = 1 per cell; RESULTS-2 correctly says "directions rather than evidence".
  - bench-3: n = 15 per surviving arm over 3 tasks of one file. CIs are wide (A 45–92%, E 21–73%), and STATUS correctly makes no ranking claim.
  - bench-4: the same model across harnesses (fair on the model), but arms ran in different batches on different days on a free pool. n = 7 per arm and 3 tasks. Token ratios are bootstrapped over attempts that are not independent across tasks. "Switch candidates" are listed while incomplete; correctly "no switch" until complete.
  - bench-5: NIM arm N ran in one batch on 2026-11-14 (parallel 4) against bench-3 arm A, run on other days. That is not interleaved, so provider-day effects are confounded. Fisher "not worse" at 21 vs 15 has low power (the PREREG says so).
  - No model ranking is over-read in any RESULTS or STATUS file.

## Honest-views check (pictures vs record)
1. bench-5 STATUS is stale against the completed bench-3 (M11). It is the only generated status page that disagrees with a recomputation; bench-3 and bench-4 STATUS are byte-for-byte reproducible.
2. Root README "Open question" row: 2.4× without the H10/compiler caveats (M9). Root status row for pilot 2: steady-state ratios without "medians, one session, I/O-sensitive" (M12).
3. loop/README "Two meters": two unlabelled hosts shown as if representative (M3), and "What explains it" left stale after pilot 2 and explore (M4).
4. BRIEF status: two hosts' numbers in one list with no host label (M10); "two independent runs" stated for single-run results (M7).
5. Disclosed and fine: pilot scoreboards labelled `dev-in-sample` (wrong label, disclosed); scoreboard `dir` paths point at the original workspace (disclosed); the README states the bench plans are not in the ledger.

## What this audit could not do
- No independent or outside-family reviewer (no sub-agent tool). The owner may want one.
- Did not re-run any benchmark or training; all checks recompute from committed raw data.
- Did not recompute entry 3's per-host cost-rule constants one by one beyond the pass/fail outcome (the script applies the rule exactly).
- Did not audit VERIFY.md's procedural claims, or record/README's attack list, which is audit C's territory.
- The meter-audit model-error fit (30–46% → 22–32%) has no committed script, so it couldn't be recomputed.
