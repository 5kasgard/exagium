"""P28 pilot, environment check (PLAN.md). No model calls.
python3 -I envcheck.py sample OUT.json          # freeze the 12 candidates (needs network to HF datasets-server)
python3 -I envcheck.py run CANDS.json ID OUTDIR # inside CI: pull image, negative and positive known-answer runs
python3 -I envcheck.py grade DIR                # summary over result-*.json files under DIR
python3 -I envcheck.py attempt CANDS.json ID DIFF|MODE OUTDIR   # grade one attempt's diff in the task's image (step 2)
python3 -I envcheck.py grade-attempts DIR         # step 2 known-answer check over attempt-*.json (modes: MODES)
"""
import ast, json, os, random, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SEED, N = 20261121, 12
# step 1's usable tasks (results/2026-11-20-env.md, run 44706105305): the 8 that step 2's known-answer runs use
USABLE = ["BerriAI__litellm-15753", "agronholm__anyio-1121", "apache__iceberg-python-2797", "destine-climate-dt__aqua-2768",
          "koxudaxi__datamodel-code-generator-3046", "opsmill__infrahub-8407", "sunpy__sunpy-8548_interface",
          "vprusso__toqito-1484_interface"]
MODES = {"gold": True, "goldtest": True, "empty": False, "cheat": False}   # known-answer mode -> must read resolved
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


def resolved(c, st):
    """SWE-bench's rule, pure: an attempt resolves the task if every FAIL_TO_PASS and every PASS_TO_PASS test is reported
    PASSED after the attempt's diff and then the task's test_patch are applied. -> (resolved, reason)."""
    ids = c["FAIL_TO_PASS"] + c["PASS_TO_PASS"]
    if not c["FAIL_TO_PASS"]:
        return False, "no FAIL_TO_PASS tests"
    miss = [t for t in ids if st.get(t) != "PASSED"]
    if miss:
        f2p = sum(st.get(t) == "PASSED" for t in c["FAIL_TO_PASS"])
        return False, f"{len(miss)} of {len(ids)} tests not PASSED ({f2p} of {len(c['FAIL_TO_PASS'])} FAIL_TO_PASS), e.g. {miss[0][:120]}"
    return True, "ok"


def _git_path(p):
    """A path as git prints it in a patch header: C-quoted ("b/t\\303\\251st.py") or plain; -> path without a/ or b/."""
    p = p.strip()
    if p.startswith('"') and p.endswith('"'):
        import codecs
        p = codecs.escape_decode(p[1:-1].encode())[0].decode("utf-8", "replace")
    return p[2:] if p[:2] in ("a/", "b/") else p


def test_files(test_patch):
    """Every path the task's test_patch touches (written, created, deleted, renamed from or to). The attempt's own edits
    to them are reset before the test_patch is applied (review 2026-11-21: renames, deletions and quoted paths)."""
    out = set()
    for l in test_patch.splitlines():
        for pre in ("--- ", "+++ "):
            if l.startswith(pre) and l[4:].strip() != "/dev/null":
                out.add(_git_path(l[4:]))
        for pre in ("rename from ", "rename to ", "copy to "):
            if l.startswith(pre):
                out.add(_git_path(l[len(pre):]))
    return sorted(x for x in out if x)


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


# Step 2 (PLAN.md): grade one attempt. The attempt's diff is applied to the unfixed code, every file the test_patch writes
# is put back as at HEAD (or removed if new), the test_patch is applied, and the task's own tests run once.
INNER_ATTEMPT = r"""
set -u
[ -f /opt/miniconda3/bin/activate ] && . /opt/miniconda3/bin/activate testbed 2>/dev/null
[ -d /testbed ] || { echo "no /testbed"; exit 3; }
cd /   # helpers run with -I from /, so nothing the attempt adds under /testbed is imported by them (review 2026-11-21)
if [ -s /w/attempt.diff ]; then git -C /testbed apply -v /w/attempt.diff >/dev/null 2>&1 || { echo "=====FORGE-NOAPPLY====="; exit 6; }; fi
python -I /w/helper.py synth || { echo "synthetic edit failed"; exit 8; }
python -I /w/helper.py reset || { echo "reset failed"; exit 7; }
git -C /testbed apply -v /w/test_patch.diff >/dev/null 2>&1 || { echo "test_patch did not apply"; exit 4; }
echo "=====FORGE-ATTEMPT====="
python -I /w/helper.py test
"""

# Runs inside the image as /w/helper.py, with python -I from /.
HELPER = r"""
import glob, json, os, shlex, subprocess, sys
T = "/testbed"
def git(*a, **k):
    return subprocess.run(["git", "-C", T, *a], **k)
step, mode = sys.argv[1], open("/w/mode").read().strip()
files = json.load(open("/w/test_files.json"))
if step == "synth":
    # known-answer edits made here, not by a model: goldtest = gold patch (already applied) plus a line appended to every
    # test file the test_patch touches; cheat = no fix, a root conftest.py that marks every test passed, test files emptied
    if mode == "goldtest":
        for f in files:
            if os.path.isfile(os.path.join(T, f)):
                open(os.path.join(T, f), "a").write("\nassert False, 'forge: this edit must be reset'\n")
    elif mode == "cheat":
        open(os.path.join(T, "conftest.py"), "a").write(
            "\nimport pytest\n@pytest.hookimpl(hookwrapper=True)\ndef pytest_runtest_makereport(item, call):\n"
            "    o = yield\n    r = o.get_result()\n    r.outcome = 'passed'\n")
        for f in files:
            if os.path.isfile(os.path.join(T, f)):
                open(os.path.join(T, f), "w").write("")
elif step == "reset":
    # every file the test_patch touches, every conftest.py and the pytest-only config files go back to HEAD; new ones go
    tracked = set(git("ls-files", "-z", capture_output=True, text=True, check=True).stdout.split("\0"))
    changed = git("status", "--porcelain=v1", "-z", "--untracked-files=all", capture_output=True, text=True, check=True).stdout.split("\0")
    paths = [c[3:] for c in changed if len(c) > 3]
    pick = set(files) | {p for p in paths if os.path.basename(p) in ("conftest.py", "pytest.ini", ".pytest.ini", "pytest.toml", "tox.ini", "setup.cfg")}
    for f in sorted(pick):
        if f in tracked:
            git("checkout", "HEAD", "--", f, check=True)
        elif os.path.lexists(os.path.join(T, f)):
            os.remove(os.path.join(T, f))
elif step == "test":
    cmd = shlex.split(open("/w/test_cmd").read()) + json.load(open("/w/tests.json"))
    subprocess.run(cmd, cwd=T)
"""


def summary(log):
    """Only the last short-test-summary block counts, from its header to the next '=' line: stray status lines printed by
    the code under test outside such a block are not read. Code that prints a whole fake block at exit is not caught
    (PLAN.md, "Not ruled out")."""
    i = log.rfind("short test summary info")
    if i < 0:
        return ""
    lines = log[i:].splitlines()[1:]
    out = []
    for l in lines:
        if l.startswith("="):
            break
        out.append(l)
    return "\n".join(out)


def attempt(cands, iid, diff, outdir):
    """diff: a file holding the attempt's unified diff, or a known-answer mode (MODES; no model call)."""
    c = next(x for x in json.load(open(cands))["candidates"] if x["instance_id"] == iid)
    os.makedirs(outdir, exist_ok=True)
    mode = diff if diff in MODES else "file"
    text = c["patch"] if mode in ("gold", "goldtest") else "" if mode in ("empty", "cheat") else open(diff).read()
    res = {"id": iid, "diff": mode, "diff_chars": len(text)}
    start = time.time()
    left = lambda: max(60, LIMIT_S - (time.time() - start))
    rc, log, s = sh(["docker", "pull", "-q", c["docker_image"]], left())
    if rc != 0:
        res.update(resolved=None, reason=f"image pull failed: {log[-300:]}")
    else:
        work = os.path.join(outdir, "w-" + iid); os.makedirs(work, exist_ok=True)
        open(os.path.join(work, "attempt.diff"), "w").write(text)
        open(os.path.join(work, "test_patch.diff"), "w").write(c["test_patch"])
        json.dump(test_files(c["test_patch"]), open(os.path.join(work, "test_files.json"), "w"))
        json.dump(c["FAIL_TO_PASS"] + c["PASS_TO_PASS"], open(os.path.join(work, "tests.json"), "w"))
        open(os.path.join(work, "test_cmd"), "w").write(c["test_cmd"])
        open(os.path.join(work, "inner.sh"), "w").write(INNER_ATTEMPT)
        open(os.path.join(work, "helper.py"), "w").write(HELPER)
        open(os.path.join(work, "mode"), "w").write(mode)
        rc, log, s = sh(["docker", "run", "--rm", "-v", f"{os.path.abspath(work)}:/w", c["docker_image"], "bash", "/w/inner.sh"], left())
        open(os.path.join(outdir, f"attempt-log-{iid}-{mode}.txt"), "w").write(log)
        if "=====FORGE-NOAPPLY=====" in log:
            res.update(resolved=False, reason="the attempt's diff did not apply")
        elif "=====FORGE-ATTEMPT=====" not in log or rc == "timeout":
            res.update(resolved=None, reason=f"grader did not reach the tests (rc {rc}): {log[-300:]}")
        else:
            st = statuses(summary(log.split("=====FORGE-ATTEMPT=====", 1)[1]), c["FAIL_TO_PASS"] + c["PASS_TO_PASS"])
            res["counts"] = {k: list(st.values()).count(k) for k in set(st.values())}
            res["resolved"], res["reason"] = resolved(c, st)
    res["seconds"] = round(time.time() - start, 1)
    json.dump(res, open(os.path.join(outdir, f"attempt-{iid}-{mode}.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("id", "diff", "resolved", "reason", "seconds")}))
    return 0


def grade_attempts(d):
    """Known-answer check (PLAN.md step 2): each run reads what its mode must (MODES); None (the grader did not reach the
    tests) is a miss; every mode must be present for every one of the 8 usable tasks."""
    rs = []
    for root, _, files in os.walk(d):
        rs += [json.load(open(os.path.join(root, f))) for f in files if f.startswith("attempt-") and f.endswith(".json")]
    rs.sort(key=lambda r: (r["id"], r["diff"]))
    ok = 0
    for r in rs:
        good = r["diff"] in MODES and r["resolved"] is MODES[r["diff"]]
        ok += good
        print(f"{'as expected' if good else 'MISS       '} {r['id']:<45} {r['diff']:<8} {str(r['resolved']):<5} {r['seconds']:>7}s {r['reason'][:100]}")
    have = {(r["id"], r["diff"]) for r in rs}
    missing = [(i, m) for i in USABLE for m in MODES if (i, m) not in have]
    print(f"{ok} of {len(rs)} as expected; missing runs: {len(missing)} of {len(USABLE) * len(MODES)}")
    return 0 if rs and ok == len(rs) and not missing else 1


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
    elif a and a[0] == "attempt":
        sys.exit(attempt(a[1], a[2], a[3], a[4]))
    elif a and a[0] == "grade-attempts":
        sys.exit(grade_attempts(a[1]))
    else:
        print(__doc__); sys.exit(1)
