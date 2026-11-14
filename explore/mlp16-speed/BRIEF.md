# Brief: why does mlp16 retire so few instructions per nanosecond?

You are an independent investigator. This folder is everything you need. Find the cause with evidence, or rule causes out with evidence. A well-supported "ruled out" is as useful as a cause. Don't guess: every claim in your report needs a number you produced, or a line you can point to in the files here.

## The puzzle (measured, x86-64 Xeon @ 2.30 GHz VM, 48 KB L1d per core, gcc 13.3, `-O2 -std=c11 -ffp-contract=off`, no `-march`)
Steady state means 95,600 images per run, startup subtracted. The data is in `results/reference-xeon-2.3.json`.

| program | ns/img | instructions/img | instructions/ns |
|---|---|---|---|
| linear | 177.0 | 3,812 | 21.5 |
| thermo | 184.8 | 3,555 | 19.2 |
| **mlp16** | **72.7** | **645** | **8.9** |
| mlp32 | 102.2 | 949 | 9.3 |
| mlp64 | 205.4 | 1,610 | 7.8 |

mlp16 does about 1/6 of linear's instructions, yet it is only 2.4× faster. Each MLP instruction costs about 2.4× a linear one. **Why?**

The programs are in `../../loop/baselines/{linear,thermo}/infer.c` and `../../loop/candidates/mlp{16,32,64}/infer.c`, each about 60–90 lines of C.

## Already established (don't redo; you may challenge it with evidence)
- **Not dependency chains.** `src/mlp_interleave.c` classifies NB images side by side, giving bit-identical predictions. NB = 1, 2, 4 and 8 run at 74–84 ns/img against 76.5, so overlapping the chains doesn't help.
- **Not a branchless fix.** Removing the skip of zero hidden units (`-DBRANCHLESS=1`) is exact, and not faster.
- **No FP16, no `pmaddwd`, no horizontal adds.**
  - mlp16's layer 1 is int16 table adds (`paddw`). Layer 2 is float32 `mulps`/`addps`.
  - `pxor` already breaks `cvtsi2ss`'s false dependency.
  - See `reference/mlp16.objdump.txt`.
- **Cache size alone doesn't fit, on this host.**
  - mlp16's layer-1 table is 34,816 B and fits the 48 KB L1d. mlp64's table is 139 KB and doesn't, yet both run at about 8–9 instructions/ns.
  - Simulated L1d misses are 2.1 per image for mlp16 (`reference/mlp16.cachegrind-summary.txt`), mostly the input stream.
  - On 32 KB-L1d hosts (AMD EPYC 7763, many cloud x86), mlp16's table does *not* fit. **That is a live cross-host question.**
- **Instruction counts** come from valgrind/cachegrind. They are deterministic and equal the native instruction stream for these programs (no CPU dispatch in the hot loops).

## Hypotheses still open (test any; add your own)
- **H1 Load-port throughput.**
  - Layer 1 per image: 64 byte loads + 64 × H/8 16-byte table loads + address arithmetic. Count loads per image from the disassembly and compare to 2–3 loads/cycle.
  - Test: a variant with the same loads but a constant table row, or with half the loads.
- **H2 Address dependence.**
  - Each table address depends on a just-loaded pixel byte (load → shift/add → load).
  - Test: precompute all 64 row pointers for an image first, then do the adds.
- **H3 Layer 2 and argmax cost.**
  - Per hidden unit: `cwtl`, `cvtsi2ss`, `shufps` and three `mulps`/`addps`, plus W2 rows reloaded from the stack. Then a 10-way argmax with branches.
  - Test by ablation: time layer 1 only (keep its result observable, for example XOR all accumulators into a printed checksum), then layer 2 only.
- **H4 Store-to-load forwarding.** `h[]` is written as 16-byte vectors with `memcpy`, then read back as 2-byte scalars. Test a variant that extracts lanes from registers instead.
- **H5 Front end.** The unrolled code is large (`#pragma GCC unroll 64`). Test less unrolling, and compare the code size of `main` in the objdump.
- **H6 Linear is the outlier, not mlp16.** linear and thermo are simple scalar float loops, perhaps unusually friendly to this CPU at 21 instructions/ns. Test: what instructions/ns does a trivially parallel loop reach on your host? That gives a ceiling for comparison.
- **H7 From the outside suggestion: VM noise and frequency scaling.** Test: repeat the bench at different times. Include a CPU-bound control such as linear in every run; the bench interleaves programs, so shared noise largely cancels.
- **H8 32 KB L1.** On a 32 KB-L1d host, does mlp16's instructions/ns drop relative to mlp32 and mlp64? Test: run the bench there and report `lscpu`.

## How to run (Linux, gcc, python3; valgrind optional but wanted)
```
git clone -b pilot/lgn-20261115 https://github.com/5kasgard/exagium && cd exagium/explore/mlp16-speed
python3 make_data.py          # builds data/ from the committed training file only
sh build.sh                   # compiles every program with the project's flags into bin/
python3 bench.py programs.json --rounds 21 --tag <yourname>-<host>
```
To test a hypothesis:
1. Copy a program into `src/`, change one thing, and add a line to `build.sh` and an entry to a spec JSON.
2. Run `bench.py` with the original in the same run. Programs are timed interleaved, so only compare within one run.

## Rules
1. **Change one thing per variant**, and say what it is.
   - If predictions must stay the same, `bench.py` shows which programs print identical output. Ablations that change the output are fine as long as you say so.
   - An ablation must keep its work observable, or gcc may delete it. Prove it didn't: show `Ir/img` from valgrind, or the hot loop in `objdump`.
2. **Report medians with IQR** over at least 15 rounds, with the original program in the same run. Don't call a difference real if the IQRs overlap.
3. **Report your host:** `lscpu` (or equivalent), kernel, compiler version, whether you have hardware counters, and whether you're in a VM.
4. **If you do have hardware counters** (`perf stat -e cycles,instructions,...` works), say so and use them. That alone settles a lot. Report cycles, instructions and, if available, frequency.
5. **Don't use `../../loop/task/optdigits.tes`** (the test file). Nothing here needs it.
6. **No claim without evidence.** "Probably X" is allowed only as a labelled hypothesis, with the test that would decide it.

## Deliverable
Write `reports/<yourname>-<date>.md` containing:
1. your host;
2. a table of each hypothesis you tested: the variant, what changed, ns/img median and IQR against the original in the same run, Ir/img, and your verdict (supported, ruled out, or inconclusive);
3. what you think the cause is, with the confidence the data justifies;
4. anything in this brief you found wrong.

Put your `results/*.json` and variant sources next to the report. If you can't push, return the report text and the JSON files to the owner.

## Chat-only track (no code execution)
Read `reference/`:
- `*.objdump.txt` and `*.s`: the compiled code;
- `*.cachegrind-lines.txt`: instructions per source line at N = 95,600;
- `*.cachegrind-summary.txt`: totals with simulated cache and branch counts;
- `host-lscpu.txt`.

Do a static analysis:
- count loads, stores, ALU and vector ops per image in mlp16's and linear's hot loops;
- map them to ports for this CPU class and estimate a cycles/image floor for each;
- compare those floors with the measured ns/img (the clock under turbo is unknown; give a range).

Then name the experiment from H1–H8, or your own, that would best decide between the remaining causes, written as a variant someone with a shell can build. Label all static estimates as estimates.
