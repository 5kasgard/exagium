# Pilot 2: what slows mlp16, and a learned-wiring logic network (2026-11-15, dev only)

Follow-on to `pilots/lgn-20261115/`. Nothing was registered, and the test set was never scored.

## Findings
1. **Dependency chains don't explain mlp16's low instructions per ns.** Interleaving 2, 4 or 8 images kept predictions bit-identical but didn't speed up the steady state: 74–84 ns/img against 76.5, with overlapping IQRs. The interleaved versions cost 3–33% more instructions at steady state. At task size they were not faster either: NB=4 was slower by 111 ± 30 ns.
2. **The native verdict depends on run size.** At 956 images per run (loop.py's dev), fixed per-run costs dominate, and at 95,600 images the ranking changes. First-touch page faults on the working set are part of the fixed cost, but they don't explain all of it (see section 2). The logic-gate networks also cost in steps of 1,024-image blocks. So a classifier's cost needs at least two numbers: one per run and one per image.
3. **Learned wiring (Top-K) cuts gates sharply.** On the inner split, against random wiring of lgn_a's shape with 4 thresholds (10,187 gates, 0.9805):
   - lgn_c reaches 0.9833 with 72% of those gates;
   - lgn_d reaches 0.9791 with 41%.

   lgn_d has the lowest instruction count and native time per image at steady state of anything measured: 0.45× mlp16's instructions and 0.63× its time. That is from medians, without a paired SE.

   At task size it uses 0.66× mlp16's instructions and is 27.9 ± 14.9 ns faster. That is 1.9 SE, below the 2 SE line, so the time win at task size is **not established**.

   Its dev accuracy is 0.73 ± 0.50 points below mlp16's, which is not significant in either direction.

## 1. The dependency-chain hypothesis
**Setup.** `mlpx/infer_x.c` is mlp16's `infer.c` with NB images classified side by side. Each image's sums run in the same order, so predictions are bit-identical; checked on dev and on dev tiled 100×. A branchless layer 2 is optional, and exact too, since s + (±0) = s.

**Steady state.** Dev tiled 100× (95,600 images), 15 interleaved rounds, pinned:

Second run, 21 rounds, saved as `bench_nb.txt` (the first run's output wasn't saved, and gave the same picture):

| | ref | NB=1 | NB=1 branchless | NB=2 | NB=2 branchless | NB=4 | NB=4 branchless | NB=8 |
|---|---|---|---|---|---|---|---|---|
| ns/img (median) | 76.5 | 76.4 | 79.0 | 74.3 | 76.5 | 76.0 | 77.8 | 84.4 |
| IQR | 75–83 | 74–81 | 75–81 | 73–77 | 74–79 | 74–93 | 76–82 | 82–93 |

No variant is faster beyond noise. So mlp16 isn't latency-bound on its add chains.

One correction on exactness: "s + (±0) = s" isn't bitwise true when s = −0, but it can't change the argmax, and predictions were identical in every test.

The instructions-per-ns gap itself is real at steady state: linear 18.6, thermo 16.1, mlp16 7.0, mlp32 7.5, mlp64 7.0. Its cause is still unidentified; hardware counters would be needed.

## 2. Run size changes the native verdict
These are medians of (t(N) − t(0)) / N from 21–25 interleaved rounds, pinned. A Python harness times whole processes here, so N=956 is too noisy to read (IQRs of ±200 ns); loop.py's in-jail timer is the precise one at that size.

| ns/img | N = 9,560 | N = 95,600 | Ir/img at 95,600 |
|---|---|---|---|
| thermo | 184 | 189 | 3,555 |
| mlp16 | 82 | 84 | 645 |
| lgn_a (pilot 1) | 99 | 87 | 625 |
| lgn_b (pilot 1) | 81 | 66 | 431 |
| lgn_c | 74 | 66 | 414 |
| lgn_d | 61 | 53 | 291 |

**Why the LGNs are worse at small N: partly explained.**
- **Page faults.** They touch 125–370 KB of signal slots on every run. The reviewer measured 93 more first-touch page faults for lgn_a than for mlp16 at N=956.
- **Faults don't account for everything.** Using the reviewer's estimates, lgn_c has 53 extra faults but about 141 µs of extra fixed cost, while lgn_d (31 faults) and mlp16 (about 15) both have about 41 µs.
- **The estimator is uneven.** (t(N) − t(0)) / N charges mlp16's table loading to its N = 0 run, so it is subtracted. An LGN touches its slot array only when N > 0, so that cost isn't subtracted.
- **Costs come in steps.** An LGN's cost rises with ceil(N / 1024). At N = 956 every LGN runs exactly one block.
- **Sessions drift.** mlp16's steady state varied between sessions (76, 84, 90, 93 ns/img), so compare rows only within one run. Each table here comes from a single run.

## 3. Cutting cost: three measured levers
On lgn_a's shape, each lever is measured on the inner split (rows 0–2,149 train, 2,150–2,866 judge):

| change | gates | inner acc | exact? |
|---|---|---|---|
| none (16 thresholds, random wiring) | 11,847 | 0.9791 | — |
| NOT gates folded into their readers | about −9% | same predictions | yes |
| 8 thresholds | 10,415 | 0.9791 | no |
| 4 thresholds (2, 6, 10, 14) | 10,187 | 0.9805 | no |

The first row already has NOTs folded. Without that, lgn_a is 13,123 gates.

**NOT folding**, measured on pilot 1's nets, gave identical predictions on dev and on dev × 100. lgn_a went from 13,123 to 11,967 gates (731 → 691 Ir/img on dev; 90 → 86 ns at steady state).

**Learned wiring (Top-K, K = 8, after LILogicNet, arXiv 2511.12340),** all on the inner split:

| widths | τ | thresholds | inner acc | gates |
|---|---|---|---|---|
| 1000-1000-500 | 4 | 8 | 0.9637 | 4,191 |
| 2000-1000-500 | 4 | 8 | 0.9763 | 4,360 |
| 1000-1000-1000-500 | 4 | 8 | 0.9707 | 4,375 |
| 2000-2000-1000 | 6 | 8 | 0.9833 | 7,520 |
| **2000-2000-1000 (lgn_c)** | 6 | 4 | **0.9833** | 7,344 |
| **2000-1000-500 (lgn_d)** | 4 | 4 | **0.9791** | 4,197 |

On the same split, thermo scored 0.9721 and mlp16 0.9763. lgn_c and lgn_d were fixed before dev was scored.

## Dev results
**Setup.** All entries were retrained on the first 75% (`train75.csv`). Evaluation was `loop.py evaluate --test dev --rounds 60 --seed 17`, confine jail, valid scoreboard (`scoreboard-dev.json`).

| | dev acc | Ir/img | ns/img | marginal Ir | marginal ns |
|---|---|---|---|---|---|
| thermo | 0.9749 | 3,695 | 1,363 ± 30 | 3,556 | 227 ± 26 |
| mlp16 | 0.9801 | 788 | 1,247 ± 29 | 646 | 128 ± 21 |
| lgn_a | 0.9780 | 876 | 1,364 ± 28 | 731 | 217 ± 23 |
| lgn_b | 0.9707 | 669 | 1,312 ± 31 | 524 | 196 ± 21 |
| lgn_c | 0.9738 | 652 | 1,321 ± 32 | 507 | 214 ± 25 |
| lgn_d | 0.9728 | 521 | 1,219 ± 28 | 376 | 95 ± 18 |

**Paired comparisons:**

| | acc diff | Ir ratio | ns/img diff |
|---|---|---|---|
| lgn_d vs mlp16 | −0.73 ± 0.50 pt | 0.66× | −27.9 ± 14.9 (1.9 SE) |
| lgn_d vs thermo | −0.21 ± 0.47 pt | 0.14× | −144.2 ± 19.7 |
| lgn_c vs mlp16 | −0.63 ± 0.51 pt | 0.83× | +74.0 ± 26.7 |

**The inner-split edge didn't carry over.** lgn_c and lgn_d led mlp16 by 0.7 and 0.3 points on the inner split, and trail it by 0.6–0.7 on dev. Both gaps are within about 1.5 SE.

The likely cause is selection: they were the best 2 of 6 learned-wiring settings, chosen on 717 rows, so their inner scores are biased upward. Plan 2 saw the same shrinkage on new writers.

mlp16's in-jail time also moved by about 51 ns between two runs of this pilot (`mlpx/sb1.json` and this scoreboard). That is more than the 27.9 ns gap, which is one more reason not to read 1.9 SE as a win.

## Limits
- **One host** (x86-64 Xeon VM, no hardware counters) and **dev only**. The steady-state numbers come from a Python harness timing whole processes, not loop.py's trusted timer. They are medians with IQRs, not paired SEs.
- **Steady state tiles dev 100×.** A candidate that memoised repeated images would gain from that. These candidates don't, as their `infer.c` shows, but a meter built on this idea needs distinct inputs.
- **One seed.** lgn_c and lgn_d were trained with seed 1 only. Under random wiring, seed changed lgn_a's inner accuracy by 0.28 points.
- **Determinism.** lgn_d was retrained on the full training file (4,327 gates, 4 min 18 s), then `loop.py reproduce` retrained it in the jail: identical weights (`reproduce_d.log`). That was on one host; arm64 is untested. An earlier draft of this README stated the result before the run had finished. The independent reviewer caught it, and it is now true.
- **The thresholds evidence is weaker than first written.** LILogicNet's 7-vs-4 comparison is on CIFAR-10, and 7 thresholds beat 4 in every pair there, by 0.1–1.1 points. Here, 4 thresholds cost nothing measurable on the inner split.
- **Gate counts aren't comparable with the literature's.** LILogicNet's gate counts (98.45% on MNIST with 8k) cover learned gates only. Ours include the popcount, argmax and threshold circuitry.
- **The gradients were checked.** The reviewer matched `train_topk`'s gradients against finite differences on a tiny network: largest absolute error 4e-9.

## What would make a claim
**A pre-registered plan on fresh data.** The optdigits test set is spent. MNIST is the literature's comparison (LILogicNet: 98.45% with 8k gates).

**Two meters, both stated:**
- the plan's own task-size meter;
- a steady-state per-image meter on distinct inputs.

**Two claims, so neither is overclaimed:**
- lgn_d's architecture matches mlp16-class accuracy within a stated floor;
- it beats mlp16 on both meters.

**Before that plan, choose architectures on more than one seed and a bigger inner split,** so the selection bias seen here doesn't repeat.

## Files
- `lgn_c/`, `lgn_d/`: `train.py`, `infer.c` (identical to pilot 1's), and `weights.bin` trained on `train75.csv` (the first 2,867 lines of `optdigits.tra`).
- `scoreboard-dev.json`, `dev_eval60.txt`, `records/`: the dev evaluation and the counter records it cites.
- `train_full_d.log`, `reproduce_d.log`: lgn_d trained on the full file, then reproduced in the jail.
- `mlp_interleave/`:
  - `infer_x.c` (build with `-DNB=` and `-DBRANCHLESS=`);
  - `bench.py`, `bench_nb.txt`: steady-state timing;
  - `sb1.json`: the task-size scoreboard.
- `timing/`: `bench2.py` (run sizes 956, 9,560 and 95,600), its JSON outputs, and the instruction counts at steady state (`ipc.json`, `steady3.json`). Paths inside point at the original workspace.
- `sweeps/`: inner-split drivers and logs.
  - `exp2.py`, `t*.log`: thresholds;
  - `exp3.py`, `k*.log`: Top-K learned wiring.
- `not_folding/`: the task-size scoreboard for pilot 1's nets with NOT folding (`lgn_an`, `lgn_bn`).
- The data files `dev.bin` (dev, 956 × 64 bytes), `mid.bin` (×10) and `big.bin` (×100) aren't committed. Build them from rows 2,867 to the end of `optdigits.tra`.
