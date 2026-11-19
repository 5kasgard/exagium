"""Ask a free model for new candidates (queue P5). Holds the key; runs no generated code.
python3 -I runner/proposals/propose.py OUT_DIR N START
Writes OUT_DIR/<k>/{task, prompt.md, reply.md} for k in 0..N-1 (task k is TASKS[(START + k) % 3]) and appends one line
per call to OUT_DIR/calls.jsonl (for worker/ledger.py record). One call per proposal, no retry (PLAN.md)."""
import glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "router"))
import engines  # noqa: E402

ENGINE, MODEL, MAX_TOKENS = "nvidia", "nvidia/nemotron-3-ultra-550b-a55b", 16000
TASKS = ("optdigits", "text", "adult")
EXAMPLE = "label-linear"


def section(md, title):
    s = md.split("## " + title, 1)[1]
    return "## " + title + s.split("\n## ", 1)[0]


def families():
    out = []
    for d in sorted(glob.glob(os.path.join(ROOT, "runner", "candidates", "*"))):
        name = os.path.basename(d)
        if name.startswith("sat-"):
            continue
        doc = open(os.path.join(d, "train.py")).read().split('"""')[1].strip().split("\n")[0]
        out.append(f"- {name}: {doc}")
    return "\n".join(out)


def scores(task):
    """The latest committed dev lines for the task (P3 step 1 if present, else P2)."""
    for p in sorted(glob.glob(os.path.join(ROOT, "runner", "results", "*", "ci", f"dev-{task}.txt")), reverse=True):
        return "\n".join(l for l in open(p).read().splitlines() if l.strip()[:1].isalpha() or l.startswith("  "))[:3000]
    return "(none recorded)"


def prompt(task):
    contract = open(os.path.join(ROOT, "runner", "CONTRACT.md")).read()
    card = open(os.path.join(ROOT, "tasks", task, "card.json")).read()
    ex = os.path.join(ROOT, "runner", "candidates", EXAMPLE)
    return f"""You write a new candidate model for a public research repository that measures small machine-learning
models on accuracy and CPU cost. Your code is compiled and run in a sandbox with no network; it is judged on a held-back
dev split by a fixed checker.

# The contract
{section(contract, "A candidate (`runner/candidates/<name>/`): code only, no weights")}

Rules that the checks enforce:
- train.py: Python 3 with the standard library and numpy only; deterministic (fixed seeds); single thread; it must finish
  within 20 minutes on one CPU core and within 8 GiB of memory; it reads the card and training records and writes
  weights.bin. Running it twice must give byte-identical weights.bin.
- infer.c: C11, libc and libm only, compiled with gcc -O2 -std=c11 -ffp-contract=off; `./infer weights.bin in.rec N`
  reads N input records and writes exactly N output records to stdout (4-byte little-endian length, then the bytes).
  For this task an output record is one byte: the class index. Validate the weights file size; never read past buffers.
- If you choose settings, choose them on the card's tuning split only (last 20% of train.in, file order), then train
  on all of train.

# The task card
```json
{card}
```

# A working example candidate ({EXAMPLE}): follow its input and output handling exactly
```python
{open(os.path.join(ex, "train.py")).read()}
```
```c
{open(os.path.join(ex, "infer.c")).read()}
```

# Families that already exist (propose a different one)
{families()}

# Current dev results on this task (accuracy, instructions per item, ns per item)
{scores(task)}

# Your answer
Propose ONE new model family for this task that you expect to beat label-linear on dev accuracy, or to match it at a
lower cost per item. Answer with exactly three fenced blocks, in this order and nothing else:
```hypothesis
(at most 10 lines: the family, why it should do well here, and what result would show it does not)
```
```python
(the complete train.py)
```
```c
(the complete infer.c)
```
"""


def main():
    out, n, start = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    engines.TIMEOUT = 600   # Nemotron reasons before answering (worker release test 1)
    for k in range(n):
        task = TASKS[(start + k) % len(TASKS)]
        d = os.path.join(out, str(k))
        os.makedirs(d, exist_ok=True)
        p = prompt(task)
        open(os.path.join(d, "task"), "w").write(task)
        open(os.path.join(d, "prompt.md"), "w").write(p)
        r = engines.compat_call(ENGINE, MODEL, p, max_tokens=MAX_TOKENS)
        open(os.path.join(d, "reply.md"), "w").write(r.get("text") or f"(no reply: status {r.get('status')}: {r.get('error')})")
        with open(os.path.join(out, "calls.jsonl"), "a") as f:
            f.write(json.dumps({k2: r.get(k2) for k2 in ("t", "engine", "model", "ok", "status", "latency_s", "usage")}) + "\n")
        print(f"proposal {k}: task {task}, call ok {r.get('ok')}, status {r.get('status')}")


if __name__ == "__main__":
    main()
