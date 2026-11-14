# Pilot: logic-gate network on optdigits (2026-11-15, dev only)

**Outcome: the hypothesis is not supported.** No logic-gate network here beats mlp16 on both meters at mlp16's accuracy. Nothing was registered, and the test set was never scored (`loop.py` only hashes `optdigits.tes` as a frozen input).

## The hypothesis (from `forge-next-item-2026-11-15`)
A logic-gate network (Petersen et al. 2022, arXiv 2210.08277) on thermo's bits can reach thermo/mlp16 accuracy on dev with:
- fewer simulated instructions than mlp16, **and**
- a lower marginal native time than mlp16.

## What was built (only `train.py` and `infer.c`, no teacher)
**`train.py`:**
- Thermometer bits (1,024) feed layers of 2-input gates with fixed random wiring. Each gate learns a softmax over the 16 binary ops, relaxed to reals.
- The last layer is split into 10 groups, and a class score is its group sum / τ. Training uses Adam with deterministic explicit sums, as in mlp16.
- After training, each gate keeps its most likely op. The whole classifier is then **compiled into one circuit** of 2-input gates: the threshold comparators on pixel bit-planes, the learned gates, a full-adder popcount per class, and an argmax (first maximum wins).
- The circuit is constant-folded, hashed, pruned and levelled. `train.py` asserts that it predicts exactly what the network does on every training row.

**`infer.c`, bit-sliced:**
- A block of 1,024 images (KV = 8 vectors of 128 bits per signal) is transposed into bit-planes with SWAR 8×8 bit and byte transposes, about 100 Ir/img.
- The circuit then runs one op-group at a time, with no per-gate dispatch.
- Finally the 4 answer bits are read back.
- Pixels > 16 are rejected. The slot array spans every uint16, so a weights file can't index outside it.

**Training time.** 3.5 min on 2,867 rows on this host, far inside the 1,800 s limit. The sceptic's worry that byte-identical training would cap the net at a few thousand gates didn't hold: one epoch of a 3,000-gate net took 0.34 s.

## Choosing the variants without dev
- Shapes and τ were swept on an **inner split of the first 75%**: train on rows 0–2,149, judge on rows 2,150–2,866 (717 rows).
- On that inner split thermo scored 0.972 and mlp16 0.976.
- Two variants were then fixed, before dev was scored:

| | widths | τ | inner acc | gates compiled |
|---|---|---|---|---|
| lgn_a | 4000-4000-2000-1000 | 6 | 0.979 (seed 1), 0.982 (seed 2) | 13,123 |
| lgn_b | 4000-4000-2000-500 | 4 | 0.974 | 7,980 |

Other shapes are in the sweep log below.

## Dev results
**Setup.** All seven entries were retrained on the first 75% only, so dev is out of sample for every one, baselines included. The scoreboard labels the split `dev-in-sample`, which is `loop.py`'s fixed label and is wrong for these retrained entries. Evaluation was `loop.py evaluate --test dev --rounds 60 --seed 7` on one host (Xeon @ 2.30 GHz, confine jail). The scoreboard is valid (`scoreboard-dev.json`).

| | dev acc | Ir/img | marginal Ir | ns/img | marginal ns |
|---|---|---|---|---|---|
| linear | 0.9770 | 3,955 | 3,813 | 1,214 ±26 | 193 ±29 |
| thermo | 0.9749 | 3,695 | 3,556 | 1,243 ±20 | 204 ±18 |
| mlp16 | 0.9801 | 788 | 646 | 1,134 ±16 | 103 ±17 |
| lgn_a | 0.9780 | 876 | 731 | 1,279 ±18 | 238 ±17 |
| lgn_b | 0.9707 | 669 | 524 | 1,207 ±21 | 156 ±25 |
| lgn_a_kv2 * | 0.9780 | 1,416 | 1,271 | 1,178 ±15 | 130 ±16 |
| lgn_b_kv2 * | 0.9707 | 1,003 | 858 | 1,140 ±17 | 115 ±13 |

\* KV = 2 (256 images per block) is a one-line change to `infer.c`. It was tried **after** seeing the KV = 8 dev numbers, so it is exploratory.

**Paired against mlp16** (accuracy diff, Ir ratio, ns/img diff ± SE):

| | acc diff | Ir ratio | ns/img diff | meters |
|---|---|---|---|---|
| lgn_a | −0.21 ± 0.47 pt | 1.11× | +144 ± 21 | loses both |
| lgn_b | −0.94 ± 0.60 pt | 0.85× | +72 ± 25 | wins Ir only; accuracy gap under 2 SE |
| lgn_a_kv2 | −0.21 ± 0.47 pt | 1.80× | +43 ± 20 | loses both |
| lgn_b_kv2 | −0.94 ± 0.60 pt | 1.27× | +6 ± 19 | loses Ir, ties on time |

**Against thermo:**
- Every LGN uses 0.18–0.38× thermo's instructions, and every accuracy is within 1 SE of thermo's.
- Natively, lgn_b is faster by 36 ± 20 ns/img, under 2 SE. The pre-chosen lgn_a is *slower* by 36 ± 25.
- Only the exploratory KV = 2 variants are clearly faster: −65 ± 19 and −103 ± 17.

**The near miss is lgn_b.** It is within 2 SE of both baselines' accuracy and uses 0.85× mlp16's instructions, but it is slower than mlp16 by 72 ± 25 ns/img. No variant beats mlp16 on both meters.

## Why it doesn't win
1. **Counting is expensive in logic.** These figures are from the reviewer's count on lgn_a.
   - The class score is a popcount. Exactly, that takes about 5 gates per input bit, so 4,730 gates for 10 × 100 bits.
   - The argmax adds 425 after pruning.
   - The threshold comparators add 2,360 of a possible 2,368, because the random wiring uses every input bit.
   - So 7,515 of lgn_a's 13,123 gates (57%) are bookkeeping that a float MLP gets almost free. Only 43% are learned gates.
2. **Interpreted gates are memory-bound.**
   - Each gate loads two 128-byte signals and stores one: about 40 Ir per gate per 1,024 images. That is the instruction meter's best case.
   - The live set, though, is 240–370 KB, which doesn't fit L1. Cachegrind, startup excluded, counts 50 D1 misses per image for lgn_a and 27 for lgn_b, against 2.7 for mlp16.
3. **The two meters trade off through block width.**
   - KV = 8 minimises instructions. KV = 2 makes the working set 4× smaller and roughly halves marginal native time (lgn_a 238 → 130 ns), but adds 60% more instructions.
   - Neither width tried (KV = 8 or 2) won both meters against mlp16. KV = 4 was not tried.
4. **The published speed relies on compiled code.** Petersen et al.'s 625 ns per MNIST image comes from compiling each net to straight-line C, with operands in registers. A fixed `infer.c` reading a netlist from `weights.bin` can't do that without generating code at run time. That is outside the pilot's scope, and at odds with the trust model, since a verifier reads `infer.c`, not the weights.

The scope note named two ways to fail: accuracy at trainable size, or startup swamping the win. **Neither was the cause.** Accuracy matched mlp16's within noise, and the loss shows in marginal time, not startup. The cause is cost per gate, plus the counting overhead, against a 16-unit MLP that is already tiny.

## Limits
- One host (x86-64), dev only (956 rows, the last 25% of the training file). The sizes and τ were tuned on 717 inner rows, where two seeds of lgn_a differed by 0.28 points.
- Native times are from one VM. Marginal times are differences of noisy totals.
- **Determinism.** lgn_b was trained on the full training file (8,073 gates, 4 min 22 s), then `loop.py reproduce` retrained it in the jail: identical weights. That holds on one x86-64 host and numpy 2.5.3; arm64 is untested.
- Teacher distillation (Suh & Jha, arXiv 2611.09024) was not tried. Per the scope, a teacher must be retrained inside the jail on the first 75% only.
- **Untried fixes, all inside the scope.** The reviewer estimates these could bring lgn_b to roughly a tie with mlp16, with no plausible flip for lgn_a:
  - Absorb the standalone NOT gates into their consumers, in `train.py` only. They are 1,329 of lgn_a's gates and 759 of lgn_b's, saving about 50 Ir/img.
  - Try KV = 4.
  - Order gates for a smaller live set. That halves peak slots but breaks the grouping by op (324 groups become about 9,700).
- **One dev-exposure channel.** A compiler correctness test (`ctest.py`) trained a 5-epoch, 2,500-gate net on rows 0–1,999 and printed its accuracy on rows 2,000 onward, which includes dev (0.862). It checked the compiler and played no part in choosing sizes or τ.

## Files
- `lgn_a/`, `lgn_b/`: `train.py`, `infer.c`, and `weights.bin` trained on the first 75% (`python3 -s -P train.py train75.csv weights.bin`, where `train75.csv` = the first 2,867 lines of `optdigits.tra`).
- `scoreboard-dev.json`, `dev_eval60.txt`: the evaluation above.
- `sweep.md`: the inner-split sweep.
- `reproduce.log`: lgn_b retrained on the full training file, then reproduced byte for byte in the jail.
- `records/`: the counter records cited (with SHA-256) by `scoreboard-dev.json`. The scoreboard's `dir` paths point at the original workspace, `/home/claude/pilot/`.
- The KV = 2 variants are `lgn_a`/`lgn_b` with `#define KV 8` changed to `#define KV 2` in `infer.c`.
- The baselines and mlp16 were retrained on `train75.csv` with their unchanged `train.py`. Their weights are not committed; the command above rebuilds them.
