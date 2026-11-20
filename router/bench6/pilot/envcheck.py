"""P28 pilot, environment check (PLAN.md). No model calls.
python3 -I envcheck.py sample OUT.json          # freeze the 12 candidates (needs network to HF datasets-server)
python3 -I envcheck.py run CANDS.json ID OUTDIR # inside CI: pull image, negative and positive known-answer runs
python3 -I envcheck.py grade DIR                # summary over result-*.json files under DIR
"""
import ast, json, os, random, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SEED, N = 20261121, 12
LIMIT_S = 1500   # whole check inside the job's 30-minute limit (PLAN.md), leaving time to upload


def parse_list(v):
    return v if isinstance(v, list) else ast.literal_eval(v or "[]")


def sample(out):
    sys.path.insert(0, os.path.join(HERE, "..", "illusion"))
    import illusion
    seen = set(json.load(open(os.path.join(HERE, "..", "illusion", "frozen-ids.json")))["fresh"])
    ds, splits = illusion.FRESH
    pool = sorted((dict(r, _split=s) for s in splits for r in illusion.rows(ds, s) if r["instance_id"] not in seen),
                  key=lambda r: r["instance_id"])
    perm = random.Random(SEED).sample(pool, len(pool))
    keep = ("instance_id", "repo", "base_commit", "patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "docker_image",
            "created_at", "_split")
    cands = []
    for r in perm[:N]:
        c = {k: r.get(k) for k in keep}
        c["test_cmd"] = (r["install_config"] if isinstance(r["install_config"], dict) else ast.literal_eval(r["install_config"]))["test_cmd"]
        c["FAIL_TO_PASS"], c["PASS_TO_PASS"] = parse_list(c["FAIL_TO_PASS"]), parse_list(c["PASS_TO_PASS"])
        cands.append(c)
    json.dump({"seed": SEED, "pool": len(pool), "excluded_seen": len(seen), "order": [r["instance_id"] for r in perm],
               "candidates": cands}, open(out, "w"), indent=1)
    print(f"pool {len(pool)} (minus {len(seen)} shown to the model), froze {len(cands)}")


STATUSES = ("PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL", "XPASS")


def statuses(log, ids):
    """pytest -rA short summary lines -> {test id: status}, matched against the known ids (an id may itself contain
    " - ", which pytest also uses to append a failure message)."""
    ids = set(ids)
    out = {}
    for line in log.splitlines():
        st, _, rest = line.strip().partition(" ")
        if st not in STATUSES:
            continue
        rest = rest.strip()
        if rest in ids:
            out[rest] = st
        else:
            hit = [i for i in ids if rest.startswith(i + " - ")]
            if hit:
                out[max(hit, key=len)] = st
    return out


def sh(cmd, timeout):
    t = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr, round(time.time() - t, 1)
    except subprocess.TimeoutExpired as e:
        return "timeout", str(e.stdout or "")[-2000:], round(time.time() - t, 1)


def judge(c, neg, pos):
    """The PLAN.md rule, pure: -> (usable, reason)."""
    f2p, p2p = c["FAIL_TO_PASS"], c["PASS_TO_PASS"]
    if not f2p:
        return False, "no FAIL_TO_PASS tests"
    if all(neg.get(t) == "PASSED" for t in f2p):
        return False, "negative control: every FAIL_TO_PASS test already passes without the fix"
    miss = [t for t in f2p + p2p if pos.get(t) != "PASSED"]
    if miss:
        return False, f"positive control: {len(miss)} of {len(f2p) + len(p2p)} tests not PASSED with the gold patch, e.g. {miss[0][:120]}"
    return True, "ok"


def run(cands, iid, outdir):
    c = next(x for x in json.load(open(cands))["candidates"] if x["instance_id"] == iid)
    os.makedirs(outdir, exist_ok=True)
    res = {"id": iid, "image": c["docker_image"], "steps": {}}
    start = time.time()
    left = lambda: max(60, LIMIT_S - (time.time() - start))
    rc, log, s = sh(["docker", "pull", "-q", c["docker_image"]], left())
    res["steps"]["pull"] = {"rc": rc, "s": s}
    if rc != 0:
        res.update(usable=False, reason=f"image pull failed: {log[-300:]}")
    else:
        work = os.path.join(outdir, "w"); os.makedirs(work, exist_ok=True)
        for name in ("patch", "test_patch"):
            open(os.path.join(work, name + ".diff"), "w").write(c[name])
        json.dump(c["FAIL_TO_PASS"] + c["PASS_TO_PASS"], open(os.path.join(work, "tests.json"), "w"))
        open(os.path.join(work, "test_cmd"), "w").write(c["test_cmd"])
        inner = os.path.join(work, "inner.sh")
        open(inner, "w").write(INNER)
        rc, log, s = sh(["docker", "run", "--rm", "-v", f"{os.path.abspath(work)}:/w",
                         c["docker_image"], "bash", "/w/inner.sh"], left())
        res["steps"]["tests"] = {"rc": rc, "s": s}
        open(os.path.join(outdir, f"log-{iid}.txt"), "w").write(log)
        parts = log.split("=====FORGE-POSITIVE=====")
        if "=====FORGE-NEGATIVE=====" not in log or len(parts) < 2 or rc == "timeout":
            res.update(usable=False, reason=f"run did not reach both controls (rc {rc}): {log[-300:]}")
        else:
            ids = c["FAIL_TO_PASS"] + c["PASS_TO_PASS"]
            neg, pos = statuses(parts[0], ids), statuses(parts[1], ids)
            res["neg_counts"] = {k: list(neg.values()).count(k) for k in set(neg.values())}
            res["pos_counts"] = {k: list(pos.values()).count(k) for k in set(pos.values())}
            res["usable"], res["reason"] = judge(c, neg, pos)
    res["seconds"] = round(time.time() - start, 1)
    json.dump(res, open(os.path.join(outdir, f"result-{iid}.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("id", "usable", "reason", "seconds")}))
    return 0


# Runs inside the task's image. SWE-bench-style images keep the repo in /testbed and a conda env named testbed.
INNER = r"""
set -u
[ -f /opt/miniconda3/bin/activate ] && . /opt/miniconda3/bin/activate testbed 2>/dev/null
cd /testbed || { echo "no /testbed"; exit 3; }
git apply -v /w/test_patch.diff >/dev/null 2>&1 || { echo "test_patch did not apply"; exit 4; }
run() { python - <<'PY'
import json, shlex, subprocess
cmd = shlex.split(open("/w/test_cmd").read()) + json.load(open("/w/tests.json"))
subprocess.run(cmd)
PY
}
echo "=====FORGE-NEGATIVE====="; run
git apply -v /w/patch.diff >/dev/null 2>&1 || { echo "gold patch did not apply"; exit 5; }
echo "=====FORGE-POSITIVE====="; run
"""


def grade(d):
    rs = []
    for root, _, files in os.walk(d):
        rs += [json.load(open(os.path.join(root, f))) for f in files if f.startswith("result-") and f.endswith(".json")]
    rs.sort(key=lambda r: r["id"])
    for r in rs:
        print(f"{'USABLE' if r['usable'] else 'no    '} {r['id']:<45} {r['seconds']:>7}s {r['reason'][:110]}")
    u = sum(r["usable"] for r in rs)
    print(f"usable {u} of {len(rs)}; rule: at least 8 of 12 before any model attempt -> {'met' if len(rs) == N and u >= 8 else 'not met' if len(rs) == N else 'incomplete'}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "sample":
        sample(a[1])
    elif a and a[0] == "run":
        sys.exit(run(a[1], a[2], a[3]))
    elif a and a[0] == "grade":
        grade(a[1])
    else:
        print(__doc__); sys.exit(1)
