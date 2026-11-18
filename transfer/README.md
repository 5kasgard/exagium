# transfer — spec card

**What it is.** Step 1 of widening the Forge beyond one task: the same model families, trained by one fixed recipe,
on three unlike tasks given as byte records, with every item's outcome kept. It asks one question, fixed in
[`PLAN.md`](PLAN.md) before any run: does the order of families change from one kind of task to another?

**Answer so far (2026-11-18): yes.** See [`results/2026-11-18/`](results/2026-11-18/README.md): 3 accuracy flips
(2 survive a 4× training budget), 4 instruction-count flips, 2 native-time flips. The logic-gate network's cost lead
over the small MLP holds on digits only.

**Parts.**
- `tasks.py`: builds optdigits, text (next character) and adult (UCI income) as byte records; downloads pinned by sha256.
- `families.py`: linear, mlp16/32/64 and a logic-gate network; one encoding rule per feature kind; exports what the
  C programs run, plus a numpy run of the exported form.
- `infer_table.c`, `infer_lgn.c`: the measured programs (table lookups; bit-sliced gates).
- `meter.py`: per-item outcomes (C must equal numpy byte for byte), instructions per item (cachegrind, start-up
  removed), native ns per item (rounds, random order, one CPU).
- `decide.py`: the fixed rule. `.github/workflows/transfer.yml` runs it all on Actions (manual; read-only token;
  checks the run's files against the registered commit).

**Run locally.**
```
python3 transfer/tasks.py /tmp/tasks
python3 transfer/families.py /tmp/tasks/text mlp16 1 /tmp/w/mlp16-s1.bin > /tmp/w/mlp16-s1.json
gcc -O2 -std=c11 -ffp-contract=off transfer/infer_table.c -o /tmp/w/infer_table   # and infer_lgn.c
python3 transfer/meter.py /tmp/tasks/text /tmp/w /tmp/result-text.json
python3 transfer/decide.py /tmp/result-*.json
```
Needs Python 3 with numpy, gcc, valgrind.

**Limits.** No jail (every program here was written for the plan, none proposed); three tasks; one host per task;
3 seeds; no tuning, so capacity and regularisation effects can look like structural ones. Not lgn_d's compiled
circuit. Step 2 replaces this with a general task contract and a mixed suite.
