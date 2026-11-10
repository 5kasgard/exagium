### Claim-by-claim audit

| # | Claim | Verdict | Detail |
|---|-------|---------|--------|
| 1 | Native ns/instruction table (EPYC/arm64) | **SUPPORTED** | All 10 values match marginal ns/img ÷ marginal Ir/img from the two representative hosts. |
| 2 | “MLP instruction took 1.9–2.5× as long as a baseline one” | **SUPPORTED** | EPYC ratios 2.28–2.56×; arm64 ratios 1.86–1.93×; overall span ≈1.9–2.6×. |
| 3 | “mlp64 used 0.43× linear’s instructions on x86-64 (0.65× on arm64), yet it was slower than linear or thermo, past 2 SE, on all 7 hosts” | **SUPPORTED** (inclusive “or”) | Ir ratios correct. On every host mlp64 is slower than **at least one** baseline by >2 SE (ubuntu-24.04-3: slower than thermo only; test-local: slower than linear only). |
| 4 | Cache-miss model error drops 30–46% → 22–32%; mlp16 has ~2.4 misses/img | **SUPPORTED** | Per-host RMS errors: Ir-only 30.0–45.6%; Ir+L1 22.4–31.5%. mlp16 marginal D1miss 2.37–2.39. |
| 5 | Branch mispredictions: mlp16 3.7 vs linear 4.1; mlp64 14.6 | **SUPPORTED** | Cachegrind totals: linear 4.10, mlp16 3.67, mlp64 14.58. |
| 6 | EPYC: linear ~11 inst/ns, MLPs ~4.4–4.7 inst/ns | **SUPPORTED** | ubuntu-24.04-1 marginal: linear 11.22, mlp16 4.46, mlp32 4.38, mlp64 4.67. |
| 7 | “Plausible cause is dependent table lookups; needs perf; no host exposes it” | **NOT CHECKABLE FROM THIS DATA** | Hypothesis / future work. |
| 8 | Proposed rule: “cost claim must state both meters; passes only if cheaper on Ir/correct **and** >2 SE faster natively on every claimed architecture; disagreement → inconclusive” | **AMBIGUOUS** | “On every architecture it claims” is not operationalised: the data has multiple hosts per architecture, paired SE is per-host, and no aggregation rule is given. |
| 9 | “Plan 2 predates this rule and was judged by its own” | **NOT CHECKABLE FROM THIS DATA** | Historical/administrative. |
|10| Phantom energy counter reads 0 J idle & loaded | **SUPPORTED** | Raw data confirms perf opens OK, reads 0 raw units in both cases. |
|11| Outcome inconclusive (ledger entry 3) | **NOT CHECKABLE FROM THIS DATA** | Administrative. |
|12| Claims A & B descriptions | **NOT CHECKABLE FROM THIS DATA** | These are the claims under test. |
|13| Verifier/CI ran on 3 x86-64 + 3 arm64; scoreboards valid; retrained byte-identically; predictions identical | **SUPPORTED** | 7 hosts present; accuracies identical to 6 decimals on all. |
|14| Test accuracy / Ir/img table | **SUPPORTED** | All values match host ubuntu-24.04-1 (x86) and ubuntu-24.04-arm-1 (arm) within rounding. |
|15| “Part A passes on every host. mlp16 uses 0.19–0.29× baselines’ instructions; accuracy 0.06 pts below thermo, inside 0.5-pt floor” | **PARTLY NOT CHECKABLE** | Ir ratios 0.185–0.289 ≈ 0.19–0.29; accuracy gap 0.0556% ≈ 0.06 pts. “Beats both on Ir per correct answer past 2 SE” – no SE for derived metric Ir/acc provided in data. |
|16| Part B inconclusive: mlp32/64 beat linear (z 4.3, 3.5), lead over thermo +0.56 (z 1.62) & +0.11 (z 0.33) <2 SE | **SUPPORTED** | Acc diffs and SEs from ubuntu-24.04-1 pairs yield exactly those z-scores. |
|17| “Code reading: no test data carried in, no env-dependent path” | **NOT CHECKABLE FROM THIS DATA** | Source code not supplied. |
|18| “mlp16 does the job with ~1/5 (x86) to 1/4 (arm) of simulated instructions, startup included” | **SUPPORTED** | Total Ir ratios: mlp16/thermo = 0.198 (x86), 0.289 (arm). |
|19| “Natively 17–31% faster than both baselines on every host (e.g. 520±5 vs 706/711 on EPYC); startup removed 1.9–2.3× faster, not 5×” | **OVERSTATED** | On cited EPYC host (ubuntu-24.04-1) marginal speedup = 339.8/144.7 = **2.35×** vs linear, 332.6/144.7 = **2.30×** vs thermo – exceeds claimed 2.3× ceiling. Across all hosts marginal speedup ranges 1.80–2.63×. |
|20| “mlp64 at 0.43× linear’s instructions was slower than at least one baseline on all 7 hosts; native time wasn’t in the rule” | **SUPPORTED** (first clause) | Numerically slower than ≥1 baseline on every host (ubuntu-24.04-3: slower than both; test-local: slower than linear). |
|21| “Bigger MLP’s accuracy edge over thermo is within noise; re-run needs new plan” | **SUPPORTED** | mlp32/thermo z=1.62, mlp64/thermo z=0.33 < 2. |

---

### Rule & code assessment

**Proposed rule** (“both meters, >2 SE on every architecture”) is **not fully computable** from the given fields because:
- “On every architecture it claims” lacks a host-aggregation rule (per-host paired SEs exist, but no meta-analysis method is specified).
- “Simulated instructions per correct answer” requires a derived SE not present in the data.

**Code function `cost_disagreements`** correctly implements its docstring: it flags pairs where `Ir_ratio < 1` (a has fewer instructions) yet `ns_per_image_diff > 2 * SE` (a is slower by >2 SE). No logic errors; assumes input pairs already have numeric `ns_per_image_diff` and `ns_per_image_diff_se` fields (the raw text shows “±” strings, but the schema implies parsed floats).

(model: nvidia/nemotron-3-ultra-550b-a55b:free, generation id: gen-1794354707-z8IpfFJw09UgDb15wvgj, input tokens: 6346, output tokens: 11257, reasoning tokens: 9509)
