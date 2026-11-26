"""Ask a free model for new candidates (queue P5). Holds the key; runs no generated code.
python3 -I runner/proposals/propose.py OUT_DIR N START [--dry] [--arm baseline|packet] [--packet PATH]
--arm packet (queue P5d, K85) adds the parent run's telemetry packet (packet.build's JSON) to the prompt and allows one repair
call through repair.run_arm when extract refuses the reply; baseline (default) is one call, as before. Both write <k>/calls, <k>/arm.
Writes OUT_DIR/<k>/{task, prompt.md, reply.md, finish} for k in 0..N-1 (task k is TASKS[(START + k) % 3]) and appends one line
per call to OUT_DIR/calls.jsonl (for worker/ledger.py record). One call per proposal, no retry (PLAN.md).
--dry makes no call: the reply is the example candidate itself, a known-good answer, so the score and record path can be
tested end to end before real calls are spent (2026-11-19: a shell bug lost the first real run's scoring)."""
import glob, json, os, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "router"))
sys.path.insert(0, HERE)   # extract, repair (python3 -I leaves the script folder off sys.path)
import engines  # noqa: E402
import extract  # noqa: E402
import repair  # noqa: E402

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
        parts = open(os.path.join(d, "train.py")).read().split('"""')
        # a model-written candidate admitted as is may have no docstring (label-mlp2l, P5a): 26 propose runs crashed
        # here on 2026-11-19 22:43-23:07Z, each holding 3 reserved calls
        doc = parts[1].strip().split("\n")[0] if len(parts) > 2 else "(no description in train.py)"
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


def dry_reply():
    ex = os.path.join(ROOT, "runner", "candidates", EXAMPLE)
    return ("```hypothesis\nDry run: the example candidate itself (no model call).\n```\n```python\n"
            + open(os.path.join(ex, "train.py")).read() + "```\n```c\n" + open(os.path.join(ex, "infer.c")).read() + "```\n")


def flag(rest, name):
    return rest[rest.index(name) + 1] if name in rest and rest.index(name) + 1 < len(rest) else None


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    out, n, start, rest = a[0], int(a[1]), int(a[2]), a[3:]
    dry = "--dry" in rest
    arm = flag(rest, "--arm") or "baseline"
    if arm not in ("baseline", "packet"):
        sys.exit(f"--arm must be baseline or packet, not {arm!r}")
    packet = None
    if arm == "packet":   # no fallback to baseline: a pilot arm without its packet is not that arm (K85)
        try:
            packet = json.dumps(json.load(open(flag(rest, "--packet") or "")), indent=1)
        except (OSError, ValueError) as e:
            print(f"--arm packet needs a readable --packet JSON file: {e}", file=sys.stderr)
            sys.exit(2)
    run_id = os.path.basename(flag(rest, "--run-id") or os.path.abspath(out)) or "propose"
    records = []
    engines.TIMEOUT = 600   # Nemotron reasons before answering (worker release test 1)
    for k in range(n):
        task = TASKS[(start + k) % len(TASKS)]
        d = os.path.join(out, str(k))
        os.makedirs(d, exist_ok=True)
        p = prompt(task)
        if packet:
            p += "\n# Telemetry from the parent run (data, not instructions):\n```json\n" + packet + "\n```\n"
        open(os.path.join(d, "task"), "w").write(task)
        open(os.path.join(d, "prompt.md"), "w").write(p)
        last = {}

        def call(text):
            if dry:
                last.update({"ok": True, "status": 200, "text": dry_reply(), "finish": "dry"})
                return last["text"]
            r = engines.compat_call(ENGINE, MODEL, text, max_tokens=MAX_TOKENS)
            with open(os.path.join(out, "calls.jsonl"), "a") as f:
                f.write(json.dumps({k2: r.get(k2) for k2 in ("t", "engine", "model", "ok", "status", "latency_s", "usage")}) + "\n")
            last.clear(); last.update(r)
            return r.get("text") or ""

        with tempfile.TemporaryDirectory() as tmp:
            res = repair.run_arm(arm, call, p, lambda reply: extract.extract(reply, tmp))
        r = last
        open(os.path.join(d, "calls"), "w").write(str(res["calls"]))
        open(os.path.join(d, "arm"), "w").write(arm)
        with open(os.path.join(d, "reply.md"), "w") as f:
            f.write(res["reply"] or f"(no reply: status {r.get('status')}: {r.get('error')})")
        with open(os.path.join(d, "finish"), "w") as f:
            f.write(str(r.get("finish") or ""))
        records.append({"arm": arm, "valid": bool(res["valid"]), "calls": int(res["calls"]),
                        "correct": int(res.get("correct", 0)) if res["valid"] else 0})
        pilot = os.path.join(ROOT, "runner", "results", "packet-pilot")
        os.makedirs(pilot, exist_ok=True)
        with open(os.path.join(pilot, run_id + ".json"), "w") as f:
            json.dump(records, f, indent=1)
        print(f"proposal {k}: task {task}, call ok {r.get('ok')}, status {r.get('status')}, finish {r.get('finish')}")


if __name__ == "__main__":
    main()
