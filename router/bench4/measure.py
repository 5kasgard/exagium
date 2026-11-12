"""bench-4 measures for one attempt, from its own files. Standard library only.

usage: measure.py OUT_DIR WORKSPACE FIX_TEST HIDDEN_TEST P2P_JSON META_JSON > result.json
OUT_DIR holds run.json, meter.jsonl, egress.json and snaps/ (see attempt.py).

Runs inside a container with no network, no key and no token (see bench4.yml): grading imports the agent's
code, so it gets nothing worth taking. Git runs with hooks and fsmonitor off.
Model calls and tokens come from the meter, which every harness goes through, so they are counted the same
way for every arm. Each snapshot of record/ is graded by the agent's own test file ("visible") and by the
grader ("hidden"); the clock is the number of model calls forwarded so far.
"""
import hashlib, json, os, re, shutil, subprocess, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decide, grade

out, ws, fix_test, hidden, p2p_file, meta = sys.argv[1:6] + [json.loads(sys.argv[6])]
run = json.load(open(out + "/run.json")) if os.path.exists(out + "/run.json") else {}
p2p = json.load(open(p2p_file))
calls = [json.loads(l) for l in open(out + "/meter.jsonl")] if os.path.exists(out + "/meter.jsonl") else []
egress = json.load(open(out + "/egress.json")) if os.path.exists(out + "/egress.json") else {}


def judge(res):
    return res.get(hidden) == "pass" and not [t for t in p2p if res.get(t) != "pass"]


GIT = ["git", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null", "-c", "safe.directory=*"]
# A pass whose non-test code reaches into the test machinery, or a file that shadows a standard module, is
# flagged: grading imports the agent's code, which could otherwise report a pass it didn't earn.
TAMPER = re.compile(r"unittest|TestResult|addSuccess|sys\.modules|builtins|__import__|os\._exit|@@RESULTS@@|sys\.exit|atexit")
STDLIB = {"json", "sys", "os", "unittest", "io", "re", "shutil", "tempfile", "hashlib", "contextlib", "subprocess", "site", "sitecustomize", "usercustomize"}


def tamper_signs(workdir):
    signs = []
    recd = os.path.join(workdir, "record")
    for f in sorted(os.listdir(recd)) if os.path.isdir(recd) else []:
        p = os.path.join(recd, f)
        if f.endswith(".py") and f[:-3] in STDLIB:
            signs.append(f"shadows stdlib: {f}")
        if f.endswith(".pth"):
            signs.append(f"path file: {f}")
    d = subprocess.run(GIT + ["-C", workdir, "diff", "HEAD", "--", "record"], capture_output=True, text=True).stdout
    cur = None
    for line in d.splitlines():
        if line.startswith("+++ "):
            cur = line[6:]
        elif line.startswith("+") and cur and not os.path.basename(cur).startswith("test_") and TAMPER.search(line):
            signs.append(f"{cur}: {line[1:].strip()[:120]}")
    return signs[:20]


def visible(workdir):
    try:
        p = subprocess.run([sys.executable, "-I", "-c", grade.RUNNER], cwd=os.path.join(workdir, "record"),
                           capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return False
    for line in p.stdout.splitlines():
        if line.startswith("@@RESULTS@@"):
            r = json.loads(line[len("@@RESULTS@@"):])
            return "pass" in r.values() and not {"fail", "error"} & set(r.values())
    return False


# the diff first: grading copies the fix commit's test file over the agent's
files = {}
if os.path.isdir(os.path.join(ws, ".git")):
    subprocess.run(GIT + ["-C", ws, "add", "-A"], capture_output=True)
    d = subprocess.run(GIT + ["-C", ws, "diff", "--cached", "--numstat", "HEAD"], capture_output=True, text=True).stdout
    for line in d.splitlines():
        a, r, f = line.split("\t", 2)
        files[f] = [a, r]
signs = tamper_signs(ws) if os.path.isdir(os.path.join(ws, ".git")) else []
final = grade.run_tests(ws, fix_test) if os.path.isdir(os.path.join(ws, "record")) else {}
broken = [t for t in p2p if final.get(t) != "pass"]
graded_pass = final.get(hidden) == "pass" and not broken
passed = graded_pass and not signs

snaps = run.get("snaps") or []
tmp = tempfile.mkdtemp()
if snaps:
    shutil.copytree(ws, tmp + "/w", ignore=shutil.ignore_patterns(".git", "__pycache__"))
base_py, seen, budget_end, graded_snaps = None, {}, time.time() + 900, True
for s in snaps:
    d = os.path.join(out, "snaps", f"{s['i']:03d}")
    py = open(d + "/record.py", "rb").read() if os.path.isfile(d + "/record.py") and not os.path.islink(d + "/record.py") else b""
    base_py = py if s["i"] == 0 else base_py
    s["changed"] = py != base_py
    code = hashlib.sha256(b"".join(open(os.path.join(d, f), "rb").read() for f in sorted(os.listdir(d))
                                   if f.endswith(".py") and os.path.isfile(os.path.join(d, f)) and not os.path.islink(os.path.join(d, f)))).hexdigest()
    if code in seen:          # same code as an earlier snapshot: same verdicts
        s["visible_green"], s["hidden_pass"] = seen[code]
        continue
    if time.time() > budget_end:
        graded_snaps = False
        s["visible_green"] = s["hidden_pass"] = None
        continue
    shutil.rmtree(tmp + "/w/record"); shutil.copytree(d, tmp + "/w/record", symlinks=True)
    s["visible_green"] = visible(tmp + "/w")
    s["hidden_pass"] = judge(grade.run_tests(tmp + "/w", fix_test))
    seen[code] = (s["visible_green"], s["hidden_pass"])
shutil.rmtree(tmp, ignore_errors=True)
green = next((s for s in snaps if s["changed"] and s["visible_green"]), None)
meter_calls = (run.get("meter") or {}).get("calls")
first_hidden = next((s for s in snaps if s["hidden_pass"]), None)

forwarded = [c for c in calls if isinstance(c.get("status"), int)]
ok = [c for c in forwarded if c["status"] == 200]
upstream_errors = [c for c in forwarded if c["status"] != 200 or c.get("stream_error") or c.get("truncated")]


def tsum(k):
    v = [(c.get("usage") or {}).get(k) for c in ok]
    return sum(x for x in v if x is not None), sum(1 for x in v if x is None)


prompt, p_missing = tsum("prompt")
completion, c_missing = tsum("completion")
cached, _ = tsum("cached")
reasoning, _ = tsum("reasoning")
err_text = ""
p = os.path.join(out, "harness", "harness.err")
if os.path.exists(p):
    err_text = open(p, errors="replace").read()[-600:]

print(json.dumps(dict(meta, **{
    "end": decide.classify(run, calls, passed, control=bool(meta.get("control"))),
    "passed": passed, "graded_pass": graded_pass, "tamper_signs": signs, "hidden": final.get(hidden, "missing"), "pass_to_pass_broken": broken,
    "meter_calls": meter_calls, "calls_unlogged": None if meter_calls is None else meter_calls - len(forwarded),
    "all_snapshots_graded": graded_snaps,
    "calls": len(forwarded), "calls_ok": len(ok), "upstream_errors": [[c["status"], (c.get("error") or c.get("stream_error") or ("truncated" if c.get("truncated") else ""))[:200]] for c in upstream_errors][:10],
    "refused_at_cap": sum(1 for c in calls if c.get("status") == "refused_cap"),
    "refused_path": sum(1 for c in calls if c.get("status") == "refused_path"),
    "tokens": {"prompt": prompt, "completion": completion, "cached": cached, "reasoning": reasoning,
               "total": prompt + completion, "calls_missing_usage": max(p_missing, c_missing)},
    "ending": run.get("ending"), "harness_rc": run.get("harness_rc"), "wall_s": run.get("wall_s"),
    "diff_files": files, "n_snaps": len(snaps), "snapshot_errors": run.get("snapshot_errors") or [],
    "green_calls": green["calls"] if green else None, "green_kept": green["hidden_pass"] if green else None,
    "first_hidden_pass_calls": first_hidden["calls"] if first_hidden else None,
    "snaps": [{k: s.get(k) for k in ("i", "calls", "t", "changed", "visible_green", "hidden_pass")} for s in snaps],
    "egress": egress, "harness_err_tail": err_text, "grade_results": final})))
