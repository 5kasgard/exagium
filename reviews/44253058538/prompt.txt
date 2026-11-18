You are reviewing a new general task runner for a project that measures small AI artifacts. Be adversarial and specific. No praise. For each problem: location (file, function), what is wrong, a concrete input or scenario that shows it, why it matters, a fix. Grade critical / major / minor. End with: ship as is / fix then ship / do not ship.

Focus, in order:
1. Can a candidate's output (untrusted bytes, written by code that may be adversarial) make a checker or the runner give a wrong score, crash in a way that is counted as success, or read/write outside its sandbox? (The jail itself is loop.py's, reviewed before; review how this runner uses it.)
2. Is the claim "no task-specific code in run.py" true, and does the contract really cover verifier-scored tasks with variable-size inputs and outputs?
3. Statistical or measurement errors: SE of the mean score, marginal instructions per item (Ir at N minus Ir at N=0, over N), native ns per item, floors.
4. The test gate: can --split test be reached without a valid plan, or with weights that are not reproducible?
5. The WalkSAT program: correctness bugs (it is a candidate, not trusted, but a wrong baseline misleads).
Not in scope: style, naming, the choice of tasks.

=== runner/CONTRACT.md
# The task contract (queue P2): bytes in, bytes out, a checker

**Status: hypothesis, being tested.** Written 2026-11-18 after transfer step 1 showed one-task results don't carry
over (`../transfer/results/2026-11-18/`).

## Hypothesis
One contract expresses every kind of task the Forge means to hold, with **no task-specific code in the runner**:
labels (vision, language, tabular) and tasks scored by a verifier with no answer key (SAT first; proofs and
emitted programs later).

**It fails for a kind of task if** adding that kind needs a change to `runner/run.py` beyond the contract, or if the
same artifact scores differently through the runner than through a direct run (the equivalence check below).

## A task (`tasks/<name>/card.json`, data built by `tasks/build.py`, every download pinned by sha256)
- **Records.** Every input and output is a file of records: a 4-byte little-endian length, then that many bytes.
  Fixed-size tasks just use one length. Inputs and outputs may vary in size per item (a SAT instance, a proof).
- **Splits.** `train`, `dev`, `test`; each has `<split>.in` (inputs) and `<split>.key` (what only the checker sees:
  labels, or empty records when the checker needs no key). Training sees `train.in`, `train.key` and the card.
- **Checker.** `"label"` (the built-in: an output must equal its key byte for byte) or a task's own `check.py`:
  `python3 -I check.py IN KEY OUT` prints one JSON line per item `{"score": s}` with 0 ≤ s ≤ 1. The checker is task
  code, trusted, run outside the jail; it treats the candidate's output as untrusted bytes.
- **Floor.** A named baseline whose dev score a candidate must beat to count (majority class; all-false assignment).
- **Baselines.** Named candidates that every evaluation of the task must include.
- **Test gate.** The test split is scored only under a ledger plan that freezes the runner, the card, the built data
  and the candidates' code, after the ledger verifies (as `loop.py` does).

## A candidate (`runner/candidates/<name>/`): code only, no weights
- `train.py`: `python3 -s -P train.py card.json train.in train.key weights.bin` (stdlib + numpy, deterministic,
  one process, single thread). A candidate with nothing to learn still writes a weights file.
- `infer.c`: `./infer weights.bin in.rec N` writes exactly N output records to stdout. N = 0 is start-up only.
- Weights are made by training in the jail at run time. `reproduce` trains twice and requires identical bytes;
  the test gate requires it.

## What the runner measures (`runner/run.py`; reuses `loop.py`'s jail, timer and counter by import; `loop.py` is not edited)
Per candidate and task, on one host: every item's output and score (kept), the mean score with its SE, simulated
instructions (cachegrind via `counter/count.py`) at N items and at N = 0, and native time at N and at 0 over R
rounds in random order, pinned to one CPU.

## Open questions, to settle with data, not here
- Whether one checker call per split is fast enough for slow verifiers (Lean): per-item calls may be needed.
- Whether a fixed per-task output bound is enough, or proofs and programs need a resource budget per item.
- Tuning budgets: transfer step 1 showed capacity effects can pass for structural ones. A candidate may split
  `train.in` itself; whether the card should fix a tuning split for everyone is open.
=== runner/run.py
#!/usr/bin/env python3
"""The general runner (queue P2; runner/CONTRACT.md): train, run, check and meter any candidate on any task card.

    python3 runner/run.py evaluate --data BUILT --task NAME --cand NAME... [--split dev] [--rounds 5] [--seed 1]
                                   [--plan SEQ] --out RESULT.json [--records DIR]
    python3 runner/run.py reproduce --data BUILT --task NAME --cand NAME...

BUILT is tasks/build.py's output. Nothing here knows any task: the card names the checker, the output bound, the
floor and the baselines. Reuses loop.py by import (not edited): its jail (bwrap, else confine), its trusted timer
and its counter call pattern.
  - train: the candidate's train.py, jailed, sees only train.py, card.json, train.in and train.key.
  - outputs: one jailed run of infer on the split, under loop's timer, pinned to one CPU; the bytes it writes must be
    exactly one record per item, each within the card's bound; the checker scores each item (0..1).
  - instructions: counter/count.py (cachegrind, reference cache) on the same jailed run at N items and at N = 0.
  - native time: the timed run at N and at 0, R rounds, candidates in a fresh random order each round.
  - --split test refuses unless ledger entry SEQ is a plan freezing every digest `digests` gives now, the ledger
    verifies, and each candidate's training reproduces byte for byte here.
Exit codes: 0 done; 1 a check failed (see the message); 3 failure.
"""
import argparse, base64, hashlib, json, math, os, random, shutil, statistics, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "loop"))
sys.path.insert(0, HERE)
import loop as L  # noqa: E402
import records  # noqa: E402

TASKS = os.path.join(REPO, "tasks")
CANDS = os.path.join(HERE, "candidates")
CODE = ("train.py", "infer.c")
Failed = L.Failed


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def rel(p):
    return os.path.relpath(os.path.abspath(p), REPO)


def task(name, data):
    card_path = os.path.join(TASKS, name, "card.json")
    card = json.load(open(card_path))
    d = os.path.join(data, name)
    for f, want in card["data"].items():
        if sha_file(os.path.join(d, f)) != want:
            raise Failed(f"{name}: built {f} does not match the card's digest; rebuild with tasks/build.py")
    checker = (os.path.join(HERE, "checkers", "label.py") if card["checker"] == "label"
               else os.path.join(TASKS, name, card["checker"]))
    return dict(name=name, card=card, card_path=card_path, dir=d, checker=checker)


def cand(name):
    d = os.path.join(CANDS, name)
    for f in CODE:
        p = os.path.join(d, f)
        if os.path.islink(p) or not os.path.isfile(p):
            raise Failed(f"candidate {name}: needs a regular {f}")
    extra = sorted(set(os.listdir(d)) - set(CODE) - {"README.md"})
    if extra:
        raise Failed(f"candidate {name}: unexpected files {extra}")
    return dict(name=name, dir=d)


def digests(t, cands):
    tools = [os.path.abspath(__file__), os.path.join(HERE, "records.py"), t["checker"], t["card_path"],
             os.path.join(REPO, "loop", "loop.py"), L.TIMER, L.CONFINE, L.SECCOMP, L.COUNT, L.RECORD]
    out = [{"name": rel(p), "sha256": sha_file(p)} for p in tools]
    out += [{"name": f"built/{t['name']}/{f}", "sha256": h} for f, h in sorted(t["card"]["data"].items())]
    out += [{"name": rel(os.path.join(c["dir"], f)), "sha256": sha_file(os.path.join(c["dir"], f))} for c in cands for f in CODE]
    return out


def require_plan(seq, t, cands):
    path = os.path.join(REPO, "ledger", "entries", f"{seq:06d}.json")
    if not os.path.isfile(path):
        raise Failed(f"no ledger entry {seq}")
    e = json.load(open(path, "rb"))
    if e.get("kind") != "plan":
        raise Failed(f"ledger entry {seq} is not a plan")
    have = {(x["name"], x["sha256"]) for x in e["body"].get("inputs", [])}
    missing = [x["name"] for x in digests(t, cands) if (x["name"], x["sha256"]) not in have]
    if missing:
        raise Failed(f"plan {seq} does not freeze these as they are now: {', '.join(missing)}")
    for sub in ("verify", "check-anchors"):
        r = subprocess.run([sys.executable, "-I", L.RECORD, sub], capture_output=True, text=True, cwd=REPO)
        if r.returncode:
            raise Failed(f"record.py {sub} exit {r.returncode}: the plan can't be trusted here\n{r.stdout[-1500:]}{r.stderr[-1500:]}")
    return {"seq": seq, "sha256": sha_file(path)}


def train(c, t):
    """Jailed training; returns the weights' bytes."""
    with tempfile.TemporaryDirectory(prefix="exagium-rtrain-") as sb:
        shutil.copyfile(os.path.join(c["dir"], "train.py"), os.path.join(sb, "train.py"))
        shutil.copyfile(t["card_path"], os.path.join(sb, "card.json"))
        for f in ("train.in", "train.key"):
            shutil.copyfile(os.path.join(t["dir"], f), os.path.join(sb, f))
        r = L.jail([sys.executable, "-s", "-P", "train.py", "card.json", "train.in", "train.key", "weights.bin"],
                   [(sb, "/w", True)], L.TRAIN_TIMEOUT)
        if r.returncode:
            raise Failed(f"{c['name']} on {t['name']}: train.py exited {r.returncode}:\n{r.stderr.decode(errors='replace')[-2000:]}")
        w = os.path.join(sb, "weights.bin")
        if os.path.islink(w) or not os.path.isfile(w):
            raise Failed(f"{c['name']} on {t['name']}: train.py wrote no weights.bin")
        return open(w, "rb").read()


def timed(c, inp, n, timer, cpu):
    """One jailed run under loop's timer: (output bytes, ns)."""
    with tempfile.TemporaryDirectory(prefix="exagium-rrun-") as w:
        shutil.copy2(c["exe"], os.path.join(w, "infer"))
        shutil.copy2(c["weights_path"], os.path.join(w, "weights.bin"))
        shutil.copy2(timer, os.path.join(w, "timer"))
        shutil.copy2(inp, os.path.join(w, "in.rec"))
        sc = ["--seccomp"] if L.JAIL["mode"] == "confine" else []
        r = L.jail(["./timer", *sc, str(cpu), "./infer", "weights.bin", "in.rec", str(n)], [(w, "/w", False)], 1800,
                   seccomp=not sc)
    out, sep, tail = r.stdout.rpartition(b"\nexagium-timer ")
    f = tail.split()
    if r.returncode or not sep or len(f) != 4 or not tail.endswith(b"\n") or not all(x.isdigit() for x in f[:3]) \
            or not (f[3].isdigit() or f[3] == b"-1"):
        raise Failed(f"{c['name']}: timer failed (exit {r.returncode}): {r.stderr.decode(errors='replace')[-300:]}")
    ns, status, cpu_ns, pids = (int(x) for x in f)
    if status:
        raise Failed(f"{c['name']}: infer exited {status}")
    if pids != 1 and L.JAIL["mode"] == "bwrap":
        raise Failed(f"{c['name']}: the jail handed out {pids} pid(s); candidates must be one single-threaded process")
    if cpu_ns > L.CPU_SLACK[0] * ns + L.CPU_SLACK[1]:
        raise Failed(f"{c['name']}: used more than one CPU ({cpu_ns} ns CPU in {ns} ns wall)")
    return out, ns


def count(c, inp, n, outdir):
    """Counter record for one jailed run (loop.count's pattern with record files): returns Ir."""
    os.makedirs(outdir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="exagium-rcnt-") as tool:
        for src, name in ((c["exe"], "infer"), (c["weights_path"], "weights.bin"), (L.COUNT, "count.py"), (inp, "in.rec")):
            shutil.copy2(src, os.path.join(tool, name))
        tail = ["--reps", "2", "--input", "/tool/infer", "--input", "/tool/weights.bin", "--input", "/tool/in.rec", "--",
                "./infer", "weights.bin", "in.rec", str(n)]
        if L.JAIL["mode"] == "confine":
            ro = [x for flag, p in L.read_allow() for x in (("--confine-ro", p) if flag == "--ro" else ("--confine-ls", p))]
            with tempfile.TemporaryDirectory(prefix="exagium-rcnt-cwd-") as cwd:
                r = subprocess.run([sys.executable, "-I", os.path.join(tool, "count.py"), "--stdout-record",
                                    "--confine", L.JAIL["confine"], *ro, *[x.replace("/tool", tool, 1) for x in tail]],
                                   capture_output=True, timeout=3600, cwd=cwd, env=dict(L.ENV, HOME=cwd, TMPDIR=cwd))
        else:
            r = L.jail([sys.executable, "-I", "/tool/count.py", "--stdout-record", "--single-process", *tail],
                       [(tool, "/tool", False)], 3600, cwd="/tmp")
    if r.returncode not in (0, 1) or r.stdout.count(b"\n") != 1:
        raise Failed(f"{c['name']}: counter failed (exit {r.returncode}):\n{r.stderr.decode(errors='replace')[-1000:]}")
    rec = json.loads(r.stdout)
    body = {k: v for k, v in rec.items() if k != "digest_sha256"}
    if L.sha256_bytes(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()) != rec["digest_sha256"]:
        raise Failed(f"{c['name']}: counter record digest does not match")
    with open(os.path.join(outdir, f"{c['name']}-n{n}-{rec['digest_sha256'][:16]}.json"), "wb") as f:
        f.write(r.stdout)
    return rec["reps"][0]["counts"]["Ir"], r.returncode


def check(t, split, out):
    """Score the candidate's output bytes with the task's checker: per-item dicts."""
    n = t["card"]["items"][split]
    try:
        records.unpack(out, max_len=t["card"]["output"]["max_len"], count=n)
    except ValueError as e:
        raise Failed(f"output is malformed: {e}")
    with tempfile.NamedTemporaryFile(prefix="exagium-out-", delete=False) as f:
        f.write(out)
    try:
        r = subprocess.run([sys.executable, "-I", t["checker"], os.path.join(t["dir"], f"{split}.in"),
                            os.path.join(t["dir"], f"{split}.key"), f.name], capture_output=True, text=True, timeout=3600)
    finally:
        os.unlink(f.name)
    if r.returncode:
        raise Failed(f"checker rejected the output (exit {r.returncode}): {r.stderr[-500:]}")
    items = [json.loads(line) for line in r.stdout.splitlines()]
    if len(items) != n or not all(0 <= x["score"] <= 1 for x in items):
        raise Failed(f"checker gave {len(items)} scores for {n} items, or a score outside 0..1")
    return items


def mean_se(xs):
    m = statistics.fmean(xs)
    return m, (statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None)


def prepare(a):
    t = task(a.task, a.data)
    cs = [cand(x) for x in a.cand]
    return t, cs


def cmd_reproduce(a):
    L.check_jail()
    t, cs = prepare(a)
    out = []
    for c in cs:
        w1, w2 = train(c, t), train(c, t)
        out.append(dict(candidate=c["name"], task=t["name"], sha256=[hashlib.sha256(w).hexdigest() for w in (w1, w2)],
                        identical=w1 == w2, jail=L.JAIL["mode"]))
        print(f"{c['name']:14s} {t['name']:10s} identical={w1 == w2}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    if not all(x["identical"] for x in out):
        raise Failed("training is not deterministic for: " + ", ".join(x["candidate"] for x in out if not x["identical"]))


def cmd_evaluate(a):
    L.check_jail()
    t, cs = prepare(a)
    card = t["card"]
    missing = [b for b in card["baselines"] if b not in a.cand]
    if missing:
        raise Failed(f"{t['name']}: every evaluation must include the baselines {card['baselines']}; missing {missing}")
    if card["floor"] not in a.cand:
        raise Failed(f"{t['name']}: the floor baseline {card['floor']} must be evaluated")
    plan = None
    if a.split == "test":
        if a.plan is None:
            raise Failed("--split test needs --plan SEQ")
        plan = require_plan(a.plan, t, cs)
    elif a.split != "dev":
        raise Failed("--split must be dev or test")
    work = tempfile.mkdtemp(prefix="exagium-runner-")
    timer = os.path.join(work, "timer")
    L.build(L.TIMER, timer)
    cpu = sorted(os.sched_getaffinity(0))[-1]
    inp = os.path.join(t["dir"], f"{a.split}.in")
    empty = os.path.join(work, "empty.rec")
    open(empty, "wb").close()
    n = card["items"][a.split]
    res = {}
    for c in cs:
        w = train(c, t)
        if plan and train(c, t) != w:
            raise Failed(f"{c['name']}: training does not reproduce byte for byte; the test gate refuses")
        c["weights_path"] = os.path.join(work, f"{c['name']}.weights")
        open(c["weights_path"], "wb").write(w)
        c["exe"] = os.path.join(work, f"{c['name']}.infer")
        L.build(os.path.join(c["dir"], "infer.c"), c["exe"])
        out, _ = timed(c, inp, n, timer, cpu)
        items = check(t, a.split, out)
        rec_dir = a.records or os.path.join(work, "records")
        ir_n, rc_n = count(c, inp, n, rec_dir)
        ir_0, rc_0 = count(c, empty, 0, rec_dir)
        res[c["name"]] = dict(weights_sha256=hashlib.sha256(w).hexdigest(), weights_bytes=len(w),
                              code={f: sha_file(os.path.join(c["dir"], f)) for f in CODE},
                              items=items, output_b64=base64.b64encode(out).decode(),
                              ir_n=ir_n, ir_0=ir_0, ir_per_item=(ir_n - ir_0) / n,
                              counter_exit=[rc_n, rc_0], ns_n=[], ns_0=[])
    rng = random.Random(a.seed)
    order = list(cs)
    for _ in range(a.rounds):
        rng.shuffle(order)
        for c in order:
            o, ns = timed(c, inp, n, timer, cpu)
            if o != base64.b64decode(res[c["name"]]["output_b64"]):
                raise Failed(f"{c['name']}: output differs between runs")
            res[c["name"]]["ns_n"].append(ns)
            res[c["name"]]["ns_0"].append(timed(c, empty, 0, timer, cpu)[1])
    floor = statistics.fmean(x["score"] for x in res[card["floor"]]["items"])
    for name, r in res.items():
        r["mean"], r["se"] = mean_se([x["score"] for x in r["items"]])
        r["above_floor"] = r["mean"] > floor if name != card["floor"] else None
        r["ns_per_item"] = statistics.fmean((x - y) / n for x, y in zip(r["ns_n"], r["ns_0"]))
    out = dict(task=t["name"], split=a.split, n=n, metric=card["metric"], floor=dict(baseline=card["floor"], mean=floor),
               plan=plan, jail=L.JAIL["mode"], host=L.versions(), rounds=a.rounds, seed=a.seed,
               run_id=os.environ.get("GITHUB_RUN_ID", ""), digests=digests(t, cs), candidates=res)
    json.dump(out, open(a.out, "w"), indent=1, sort_keys=True)
    shutil.rmtree(work, ignore_errors=True)
    print(f"{t['name']} / {a.split} ({n} items, {card['metric']}; floor {card['floor']} = {floor:.4f}; jail {L.JAIL['mode']})")
    for name, r in sorted(res.items(), key=lambda kv: -kv[1]["mean"]):
        print(f"  {name:14s} {r['mean']:.4f} ± {r['se'] or 0:.4f}  Ir/item {r['ir_per_item']:10.1f}  "
              f"ns/item {r['ns_per_item']:9.1f}  above floor: {r['above_floor']}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("evaluate", "reproduce"):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True)
        p.add_argument("--task", required=True)
        p.add_argument("--cand", nargs="+", required=True)
        p.add_argument("--out")
        if name == "evaluate":
            p.add_argument("--split", default="dev")
            p.add_argument("--rounds", type=int, default=5)
            p.add_argument("--seed", type=int, default=1)
            p.add_argument("--plan", type=int)
            p.add_argument("--records")
    a = ap.parse_args(argv)
    if a.cmd == "evaluate" and not a.out:
        ap.error("evaluate needs --out")
    try:
        (cmd_evaluate if a.cmd == "evaluate" else cmd_reproduce)(a)
    except Failed as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
=== runner/records.py
"""Record files: each record is a 4-byte little-endian length followed by that many bytes (runner/CONTRACT.md)."""
import struct


def pack(recs):
    return b"".join(struct.pack("<I", len(r)) + bytes(r) for r in recs)


def write(path, recs):
    with open(path, "wb") as f:
        f.write(pack(recs))


def unpack(data, max_len=None, count=None):
    """Parse bytes into records. Raises ValueError on a truncated file, an over-long record or a wrong count."""
    out, i, n = [], 0, len(data)
    while i < n:
        if i + 4 > n:
            raise ValueError(f"truncated length at byte {i}")
        k = struct.unpack_from("<I", data, i)[0]
        i += 4
        if max_len is not None and k > max_len:
            raise ValueError(f"record {len(out)} is {k} bytes, over the bound {max_len}")
        if i + k > n:
            raise ValueError(f"record {len(out)} truncated")
        out.append(data[i:i + k])
        i += k
        if count is not None and len(out) > count:
            raise ValueError(f"more than {count} records")
    if count is not None and len(out) != count:
        raise ValueError(f"{len(out)} records, expected {count}")
    return out


def read(path, **kw):
    with open(path, "rb") as f:
        return unpack(f.read(), **kw)
=== runner/checkers/label.py
"""The built-in "label" checker: python3 -I label.py IN KEY OUT  (runner/CONTRACT.md).

One JSON line per item: {"score": 1 if the output record equals the key record byte for byte, else 0}.
Exit 1 if OUT is not exactly one record per key, or a record is longer than the longest key.
"""
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import records  # noqa: E402


def main():
    keys = records.read(sys.argv[2])
    try:
        outs = records.read(sys.argv[3], max_len=max(map(len, keys), default=0), count=len(keys))
    except ValueError as e:
        print(f"malformed output: {e}", file=sys.stderr)
        sys.exit(1)
    for k, o in zip(keys, outs):
        print(json.dumps({"score": int(k == o)}))


if __name__ == "__main__":
    main()
=== tasks/sat/check.py
"""SAT checker: python3 -I check.py IN KEY OUT  (runner/CONTRACT.md). KEY is unused: the checker verifies.

Prints one JSON line per item: {"score": 1 if the output assigns every variable 0 or 1 and satisfies every clause,
else 0, "satisfied": fraction of clauses satisfied (0 when the output is malformed), "malformed": bool}.
Exit 1 if OUT is not exactly one record per input; the output is untrusted bytes.
"""
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runner"))
import records  # noqa: E402


def main():
    ins = records.read(sys.argv[1])
    try:
        outs = records.read(sys.argv[3], max_len=255, count=len(ins))
    except ValueError as e:
        print(f"malformed output: {e}", file=sys.stderr)
        sys.exit(1)
    for x, o in zip(ins, outs):
        n, m = x[0], x[1] | (x[2] << 8)
        lits = [v - 256 if v > 127 else v for v in x[3:3 + 3 * m]]
        if len(o) != n or any(b > 1 for b in o):
            print(json.dumps({"score": 0, "satisfied": 0.0, "malformed": True}))
            continue
        sat = sum(any((o[abs(l) - 1] == 1) == (l > 0) for l in lits[3 * c:3 * c + 3]) for c in range(m))
        print(json.dumps({"score": int(sat == m), "satisfied": sat / m, "malformed": False}))


if __name__ == "__main__":
    main()
=== tasks/sat/card.json
{
 "task": "sat",
 "kind": "verifier-scored: constraint satisfaction",
 "description": "planted random 3-SAT, 50 variables, 210 clauses; every instance is satisfiable",
 "input": {
  "bytes": "3 + 3m",
  "meaning": "n, m (2 bytes little-endian), then 3m signed int8 literals (+v / -v, v in 1..n)"
 },
 "output": {
  "max_len": 255,
  "meaning": "n bytes, each 0 or 1: the value of variables 1..n"
 },
 "checker": "check.py",
 "metric": "solved (all clauses satisfied)",
 "baselines": [
  "sat-zeros"
 ],
 "floor": "sat-zeros",
 "sources": [
  "generated by tasks/build.py (seeds 101, 102, 103)"
 ],
 "splits": {
  "train": "train.in + train.key, seen by train.py",
  "dev": "scored by the runner, never trained on",
  "test": "scored only under a ledger plan (test_gate)"
 },
 "test_gate": "plan",
 "items": {
  "dev": 200,
  "test": 200,
  "train": 100
 },
 "data": {
  "dev.in": "646c4fbce215b008593b8286ed709605aa0faa8463bee94f0bf99416b5450904",
  "dev.key": "67042dfda5683aead81b6055d19c4dba238341f9dd82f49c0e7cc0c19c5f10d1",
  "test.in": "db3810cc9906d6f638bcb563753e6f1dfffc28395114a7e8f68f4b6ea3e17745",
  "test.key": "67042dfda5683aead81b6055d19c4dba238341f9dd82f49c0e7cc0c19c5f10d1",
  "train.in": "f701b15e650d97337f4ec04f32c6b4987ad49c73392fe3d18862d5a61d8eb357",
  "train.key": "7a12e561363385e9dfeeab326368731c030ed4b374e7f5897ac819159d2884c5"
 }
}
=== runner/candidates/sat-walksat/infer.c
/* WalkSAT (Selman, Kautz and Cohen 1994): ./infer weights.bin in.rec N
 * weights: uint32 noise in 1/1000, uint32 max flips per instance. From a random start, repeat: if all clauses hold,
 * stop; pick an unsatisfied clause (the first-found one in a random scan); with probability noise flip a random
 * variable of it, else the one whose flip breaks the fewest satisfied clauses. Deterministic: xorshift seeded per
 * instance by its index. Output: n bytes 0/1 (the best assignment seen if no solution is found). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *slurp(const char *p, long *n) {
    FILE *f = fopen(p, "rb");
    if (!f) { perror(p); exit(3); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *b = malloc(*n ? *n : 1);
    if (fread(b, 1, *n, f) != (size_t)*n) { fprintf(stderr, "short read %s\n", p); exit(3); }
    fclose(f);
    return b;
}

/* next record of a record file: 4-byte little-endian length, then the bytes. */
static const unsigned char *next_rec(const unsigned char *buf, long n, long *pos, uint32_t *len) {
    if (*pos + 4 > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    uint32_t k = buf[*pos] | buf[*pos + 1] << 8 | buf[*pos + 2] << 16 | (uint32_t)buf[*pos + 3] << 24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k >> 8 & 255, k >> 16 & 255, k >> 24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

static uint64_t rs;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 11); }

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *w = slurp(argv[1], &nw), *x = N ? slurp(argv[2], &nx) : NULL;
    if (nw != 8) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t noise = w[0] | w[1] << 8 | w[2] << 16 | (uint32_t)w[3] << 24;
    uint32_t maxf = w[4] | w[5] << 8 | w[6] << 16 | (uint32_t)w[7] << 24;
    for (long it = 0; it < N; it++) {
        uint32_t k; const unsigned char *r = next_rec(x, nx, &pos, &k);
        if (k < 3) { fprintf(stderr, "bad instance\n"); return 3; }
        int n = r[0], m = r[1] | r[2] << 8;
        if ((long)k != 3 + 3L * m || n < 1) { fprintf(stderr, "bad instance\n"); return 3; }
        const signed char *L = (const signed char *)(r + 3);
        unsigned char a[256], best[256];
        int *nt = calloc(m, sizeof(int));  /* true literals per clause */
        rs = 0x9E3779B97F4A7C15ULL ^ (uint64_t)(it + 1) * 0xBF58476D1CE4E5B9ULL;
        for (int v = 0; v < n; v++) a[v] = rnd() & 1;
        int unsat = 0, bestu = m + 1;
        for (int c = 0; c < m; c++) {
            for (int j = 0; j < 3; j++) { int l = L[3 * c + j]; nt[c] += (a[abs(l) - 1] == 1) == (l > 0); }
            unsat += nt[c] == 0;
        }
        for (uint32_t f = 0; ; f++) {
            if (unsat < bestu) { bestu = unsat; memcpy(best, a, n); }
            if (!unsat || f >= maxf) break;
            int start = rnd() % m, c = -1;
            for (int s = 0; s < m; s++) { int cc = (start + s) % m; if (!nt[cc]) { c = cc; break; } }
            int pick;
            if (rnd() % 1000 < noise) pick = abs(L[3 * c + rnd() % 3]) - 1;
            else {
                int bb = 1 << 30; pick = abs(L[3 * c]) - 1;
                for (int j = 0; j < 3; j++) {
                    int v = abs(L[3 * c + j]) - 1, brk = 0;
                    for (int d = 0; d < m; d++)
                        for (int q = 0; q < 3; q++) {
                            int l = L[3 * d + q];
                            if (abs(l) - 1 == v && nt[d] == 1 && (a[v] == 1) == (l > 0)) brk++;
                        }
                    if (brk < bb) { bb = brk; pick = v; }
                }
            }
            for (int d = 0; d < m; d++)
                for (int q = 0; q < 3; q++) {
                    int l = L[3 * d + q];
                    if (abs(l) - 1 != pick) continue;
                    int was = nt[d];
                    nt[d] += ((a[pick] == 1) == (l > 0)) ? -1 : 1;
                    unsat += (nt[d] == 0) - (was == 0);
                }
            a[pick] ^= 1;
        }
        free(nt);
        put_rec(stdout, best, n);
    }
    return 0;
}
=== runner/test_runner.py
"""Tests for the contract's untrusted-byte paths: python3 -m unittest runner/test_runner.py (stdlib + numpy)."""
import json, os, subprocess, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import records  # noqa: E402

SAT_CHECK = os.path.join(REPO, "tasks", "sat", "check.py")
LABEL_CHECK = os.path.join(HERE, "checkers", "label.py")


def run_check(checker, ins, keys, out_bytes):
    with tempfile.TemporaryDirectory() as d:
        p = lambda n: os.path.join(d, n)
        records.write(p("in"), ins); records.write(p("key"), keys)
        open(p("out"), "wb").write(out_bytes)
        r = subprocess.run([sys.executable, "-I", checker, p("in"), p("key"), p("out")], capture_output=True, text=True)
    return r.returncode, [json.loads(x) for x in r.stdout.splitlines()]


def sat_instance(n, clauses):
    lits = [l & 0xFF for c in clauses for l in c]
    return bytes([n, len(clauses) % 256, len(clauses) // 256] + lits)


class Records(unittest.TestCase):
    def test_roundtrip(self):
        recs = [b"", b"a", bytes(range(256)) * 3]
        self.assertEqual(records.unpack(records.pack(recs)), recs)

    def test_rejects(self):
        good = records.pack([b"ab", b"c"])
        for bad, kw in ((good[:-1], {}), (good + b"\x01", {}), (good, {"max_len": 1}), (good, {"count": 3}),
                        (good, {"count": 1})):
            with self.assertRaises(ValueError):
                records.unpack(bad, **kw)


class LabelChecker(unittest.TestCase):
    def test_scores(self):
        rc, s = run_check(LABEL_CHECK, [b"x", b"y"], [b"\x01", b"\x02"], records.pack([b"\x01", b"\x03"]))
        self.assertEqual((rc, [x["score"] for x in s]), (0, [1, 0]))

    def test_malformed(self):
        for out in (records.pack([b"\x01"]), records.pack([b"\x01", b"\x02\x02"]), b"\x05\x00"):
            rc, _ = run_check(LABEL_CHECK, [b"x", b"y"], [b"\x01", b"\x02"], out)
            self.assertEqual(rc, 1, out)


class SatChecker(unittest.TestCase):
    inst = sat_instance(3, [[1, 2, -3], [-1, 2, 3]])

    def test_solved_and_not(self):
        rc, s = run_check(SAT_CHECK, [self.inst] * 3, [b""] * 3,
                          records.pack([bytes([0, 1, 0]), bytes([1, 0, 0]), bytes([1, 0, 1])]))
        self.assertEqual(rc, 0)
        self.assertEqual([x["score"] for x in s], [1, 0, 1])
        self.assertEqual(s[1]["satisfied"], 0.5)

    def test_malformed_item_scores_zero(self):
        rc, s = run_check(SAT_CHECK, [self.inst] * 2, [b""] * 2, records.pack([bytes([0, 1]), bytes([0, 2, 0])]))
        self.assertEqual(rc, 0)
        self.assertEqual([x["malformed"] for x in s], [True, True])
        self.assertEqual([x["score"] for x in s], [0, 0])

    def test_wrong_count(self):
        rc, _ = run_check(SAT_CHECK, [self.inst] * 2, [b""] * 2, records.pack([bytes([0, 1, 0])]))
        self.assertEqual(rc, 1)


class NoTaskCodeInRunner(unittest.TestCase):
    def test_runner_names_no_task(self):
        src = open(os.path.join(HERE, "run.py")).read()
        names = [n for n in os.listdir(os.path.join(REPO, "tasks")) if os.path.isdir(os.path.join(REPO, "tasks", n))]
        self.assertTrue(names)
        for n in names:
            self.assertNotIn(f'"{n}"', src, f"run.py names task {n}")


if __name__ == "__main__":
    unittest.main()
