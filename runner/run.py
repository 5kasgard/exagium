#!/usr/bin/env python3
"""The general runner (queue P2; runner/CONTRACT.md): train, run, check and meter any candidate on any task card.

    python3 runner/run.py evaluate --data BUILT --task NAME --cand NAME... [--split dev] [--rounds 5] [--seed 1]
                                   [--plan SEQ] --out RESULT.json [--records DIR]
    python3 runner/run.py reproduce --data BUILT --task NAME --cand NAME...

BUILT is tasks/build.py's output. Nothing here knows any task: the card names the checker, the output bound, the
floor and the baselines. Reuses loop.py by import (not edited): its jail (bwrap, else confine) through runner/jail.py, which adds Q10's
limits and seccomp filter, its trusted timer and its counter call pattern.
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
import jail as J  # noqa: E402  (loop's jail plus Q10's limits)
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
    tools = [os.path.abspath(__file__), os.path.join(HERE, "records.py"), os.path.join(HERE, "jail.py"),
             os.path.join(HERE, "limits.c"), t["checker"], t["card_path"],
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
        r = J.jail([sys.executable, "-s", "-P", "train.py", "card.json", "train.in", "train.key", "weights.bin"],
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
        # the timer forks, so it runs without the filter and adds it to its child, in both jail modes (Q10)
        r = J.jail(["./timer", "--seccomp", str(cpu), "./infer", "weights.bin", "in.rec", str(n)], [(w, "/w", False)],
                   1800, seccomp=False)
    out, sep, tail = r.stdout.rpartition(b"\nexagium-timer ")
    f = tail.split()
    if r.returncode or not sep or len(f) != 4 or not tail.endswith(b"\n") or not all(x.isdigit() for x in f[:3]) \
            or not (f[3].isdigit() or f[3] == b"-1"):
        raise Failed(f"{c['name']}: timer failed (exit {r.returncode}): {r.stderr.decode(errors='replace')[-300:]}")
    ns, status, cpu_ns, pids = (int(x) for x in f)
    if status:
        raise Failed(f"{c['name']}: infer exited {status}")
    if pids != -1:   # -1: the child ran under the seccomp filter, which refuses every new process or thread
        raise Failed(f"{c['name']}: the timer did not filter its child (pids {pids})")
    if cpu_ns > L.CPU_SLACK[0] * ns + L.CPU_SLACK[1]:
        raise Failed(f"{c['name']}: used more than one CPU ({cpu_ns} ns CPU in {ns} ns wall)")
    return out, ns


def count(c, inp, n, outdir):
    """Counter record for one jailed run (loop.count's pattern with record files): returns (Ir, stdout sha256).
    Only exit 0 is a clean count: count.py exits 1 when reps differ, stdout differs between reps, or the program
    issued client requests that could switch counting off (GLM F051)."""
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
            # count.py forks valgrind, so it runs under the rlimits only and starts valgrind inside confine (runner/jail.py)
            r = J.jail([sys.executable, "-I", "/tool/count.py", "--stdout-record", "--single-process",
                        *J.counter_confine("/tool"), *tail],
                       [(tool, "/tool", False)], 3600, cwd="/tmp", seccomp=False)
    if r.returncode != 0 or r.stdout.count(b"\n") != 1:
        raise Failed(f"{c['name']}: counter failed (exit {r.returncode}):\n{r.stderr.decode(errors='replace')[-1000:]}")
    rec = json.loads(r.stdout)
    body = {k: v for k, v in rec.items() if k != "digest_sha256"}
    if L.sha256_bytes(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()) != rec["digest_sha256"]:
        raise Failed(f"{c['name']}: counter record digest does not match")
    path = os.path.join(outdir, f"{c['name']}-n{n}-{rec['digest_sha256'][:16]}.json")
    with open(path, "wb") as f:
        f.write(r.stdout)
    return rec["reps"][0]["counts"]["Ir"], rec["reps"][0]["stdout_sha256"], {"name": os.path.basename(path),
                                                                           "sha256": hashlib.sha256(r.stdout).hexdigest()}


def check(t, split, out, inp=None, key=None):
    """Score the candidate's output bytes with the task's checker: per-item dicts. inp/key override the split's files."""
    n = t["card"]["items"][split]
    inp = inp or os.path.join(t["dir"], f"{split}.in")
    key = key or os.path.join(t["dir"], f"{split}.key")
    try:
        records.unpack(out, max_len=t["card"]["output"]["max_len"], count=n)
    except ValueError as e:
        raise Failed(f"output is malformed: {e}")
    with tempfile.NamedTemporaryFile(prefix="exagium-out-", delete=False) as f:
        f.write(out)
    try:
        r = subprocess.run([sys.executable, "-I", t["checker"], inp, key, f.name], capture_output=True, text=True, timeout=3600)
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
    if len(set(a.cand)) != len(a.cand):   # F057: a repeated name would silently overwrite its result
        raise Failed("candidate names must be unique")
    cs = [cand(x) for x in a.cand]
    return t, cs


def cmd_reproduce(a):
    jail = J.check()
    t, cs = prepare(a)
    out = []
    for c in cs:
        w1, w2 = train(c, t), train(c, t)
        out.append(dict(candidate=c["name"], task=t["name"], sha256=[hashlib.sha256(w).hexdigest() for w in (w1, w2)],
                        identical=w1 == w2, jail=jail))
        print(f"{c['name']:14s} {t['name']:10s} identical={w1 == w2}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    if not all(x["identical"] for x in out):
        raise Failed("training is not deterministic for: " + ", ".join(x["candidate"] for x in out if not x["identical"]))


def cmd_evaluate(a):
    jail = J.check()
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
    try:
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
            J.compile_candidate(os.path.join(c["dir"], "infer.c"), c["exe"])   # Q56: never compile candidate code on the host
            out, _ = timed(c, inp, n, timer, cpu)
            items = check(t, a.split, out)
            out0, _ = timed(c, empty, 0, timer, cpu)
            if out0 != b"":   # F053: with no items there is nothing to answer
                raise Failed(f"{c['name']}: with N = 0 it must write no records (wrote {len(out0)} bytes)")
            rec_dir = a.records or os.path.join(work, "records")
            ir_n, sha_n, rec_n = count(c, inp, n, rec_dir)
            ir_0, sha_0, rec_0 = count(c, empty, 0, rec_dir)
            # F052: the counted run must be the run that was scored, else the instruction count measures another program
            if sha_n != hashlib.sha256(out).hexdigest() or sha_0 != hashlib.sha256(out0).hexdigest():
                raise Failed(f"{c['name']}: output under the counter differs from the native output")
            why = marginal_problem(ir_n, ir_0)
            if why:
                raise Failed(f"{c['name']}: {why}")
            res[c["name"]] = dict(weights_sha256=hashlib.sha256(w).hexdigest(), weights_bytes=len(w),
                                  code={f: sha_file(os.path.join(c["dir"], f)) for f in CODE},
                                  items=items, output_b64=base64.b64encode(out).decode(),
                                  ir_n=ir_n, ir_0=ir_0, ir_per_item=(ir_n - ir_0) / n,
                                  counter_exit=[0, 0], counter_records=[rec_n, rec_0], ns_n=[], ns_0=[])
        rng = random.Random(a.seed)
        order = list(cs)
        rows = records.read(inp)
        keys = records.read(os.path.join(t["dir"], f"{a.split}.key"))
        perm_in, perm_key = os.path.join(work, "perm.rec"), os.path.join(work, "perm.key")
        base = {c["name"]: records.unpack(base64.b64decode(res[c["name"]]["output_b64"])) for c in cs}
        for _ in range(a.rounds):
            rng.shuffle(order)
            # F053: every timed run sees the items in a fresh order, and each item's answer must not change, so a
            # candidate can't recognise the fixed input file and replay stored answers (loop.py's permuted rows)
            perm = rng.sample(range(n), n)
            records.write(perm_in, [rows[i] for i in perm])
            records.write(perm_key, [keys[i] for i in perm])
            for c in order:
                o, ns = timed(c, perm_in, n, timer, cpu)
                try:
                    got = records.unpack(o, count=n)
                except ValueError as e:
                    raise Failed(f"{c['name']}: a timing run's output is malformed: {e}")
                if got != [base[c["name"]][i] for i in perm]:
                    # A verifier-scored task may take a different valid answer (walksat seeds by position); what
                    # must not change is each item's score. A label task's answer is its score, so this covers both.
                    sc = [x["score"] for x in check(t, a.split, o, perm_in, perm_key)]
                    if sc != [res[c["name"]]["items"][i]["score"] for i in perm]:
                        raise Failed(f"{c['name']}: with the items reordered its per-item scores change")
                    res[c["name"]]["reordered_answers_differ"] = True
                res[c["name"]]["ns_n"].append(ns)
                res[c["name"]]["ns_0"].append(timed(c, empty, 0, timer, cpu)[1])
        floor = statistics.fmean(x["score"] for x in res[card["floor"]]["items"])
        for name, r in res.items():
            r["mean"], r["se"] = mean_se([x["score"] for x in r["items"]])
            r["above_floor"] = r["mean"] > floor if name != card["floor"] else None
            r["ns_per_item"] = statistics.fmean((x - y) / n for x, y in zip(r["ns_n"], r["ns_0"]))
            r["ns_per_ir"] = statistics.fmean(r["ns_n"]) / r["ir_n"]
        # F053, reported not enforced: in runner results so far honest trivial baselines (majority, sat-zeros) reach
        # 20-46x the lowest candidate's native ns per counted instruction (startup dominates), so loop.py's band of 30
        # would refuse honest runs. Flag it for a reader; decide on a gate once flagged runs exist.
        low = min(r["ns_per_ir"] for r in res.values())
        for r in res.values():
            r["ns_per_ir_vs_lowest"] = r["ns_per_ir"] / low
            r["ratio_flag"] = r["ns_per_ir_vs_lowest"] > L.RATIO_BAND
        out = dict(task=t["name"], split=a.split, n=n, metric=card["metric"], floor=dict(baseline=card["floor"], mean=floor),
                   plan=plan, jail=jail, host=L.versions(), rounds=a.rounds, seed=a.seed,
                   run_id=os.environ.get("GITHUB_RUN_ID", ""), digests=digests(t, cs), candidates=res)
        json.dump(out, open(a.out, "w"), indent=1, sort_keys=True)
    finally:   # F056: a failed check must not leave candidate binaries and data in /tmp
        shutil.rmtree(work, ignore_errors=True)
    print(f"{t['name']} / {a.split} ({n} items, {card['metric']}; floor {card['floor']} = {floor:.4f}; jail {L.JAIL['mode']})")
    for name, r in sorted(res.items(), key=lambda kv: -kv[1]["mean"]):
        print(f"  {name:14s} {r['mean']:.4f} ± {r['se'] or 0:.4f}  Ir/item {r['ir_per_item']:10.1f}  "
              f"ns/item {r['ns_per_item']:9.1f}  above floor: {r['above_floor']}")


def marginal_problem(ir_n, ir_0):
    """Audit 2026-11-21 (R4 F2, GLM F054): the run with N items does all the N = 0 run does and more, so it cannot count
    fewer instructions. A candidate that burns instructions only when N = 0 got -250,724 Ir per item and was accepted,
    and a negative cost heads the leaderboard front. None when the pair is possible."""
    if not (isinstance(ir_n, int) and isinstance(ir_0, int)) or ir_0 < 0:
        return f"instruction counts are not counts (N: {ir_n!r}, N = 0: {ir_0!r})"
    if ir_n <= ir_0:
        return f"the run with N items counted {ir_n} instructions, no more than the N = 0 run's {ir_0}: work was moved into the N = 0 run"
    return None


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
    except Exception:   # F059: the documented 3, not a bare traceback's 1, so callers can tell a fault from a check
        import traceback
        traceback.print_exc()
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
