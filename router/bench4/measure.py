"""bench-4 measures for one attempt, from its own files. Standard library only.

usage: measure.py OUT_DIR WORKSPACE FIX_TEST HIDDEN_TEST P2P_JSON META_JSON > result.json
OUT_DIR holds run.json, meter.jsonl, egress.json and snaps/ (see attempt.py).

Model calls and tokens come from the meter, which every harness goes through, so they are counted the same
way for every arm. Each snapshot of record/ is graded by the agent's own test file ("visible") and by the
grader ("hidden"); the clock is the number of model calls forwarded so far.
"""
import json, os, shutil, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decide, grade

out, ws, fix_test, hidden, p2p_file, meta = sys.argv[1:6] + [json.loads(sys.argv[6])]
run = json.load(open(out + "/run.json")) if os.path.exists(out + "/run.json") else {}
p2p = json.load(open(p2p_file))
calls = [json.loads(l) for l in open(out + "/meter.jsonl")] if os.path.exists(out + "/meter.jsonl") else []
egress = json.load(open(out + "/egress.json")) if os.path.exists(out + "/egress.json") else {}


def judge(res):
    return res.get(hidden) == "pass" and not [t for t in p2p if res.get(t) != "pass"]


def visible(workdir):
    try:
        p = subprocess.run([sys.executable, "-c", grade.RUNNER], cwd=os.path.join(workdir, "record"),
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
    subprocess.run(["git", "-C", ws, "add", "-A"], capture_output=True)
    d = subprocess.run(["git", "-C", ws, "diff", "--cached", "--numstat", "HEAD"], capture_output=True, text=True).stdout
    for line in d.splitlines():
        a, r, f = line.split("\t", 2)
        files[f] = [a, r]
final = grade.run_tests(ws, fix_test) if os.path.isdir(os.path.join(ws, "record")) else {}
broken = [t for t in p2p if final.get(t) != "pass"]
passed = final.get(hidden) == "pass" and not broken

snaps = run.get("snaps") or []
tmp = tempfile.mkdtemp()
if snaps:
    shutil.copytree(ws, tmp + "/w", ignore=shutil.ignore_patterns(".git", "__pycache__"))
base_py = None
for s in snaps:
    d = os.path.join(out, "snaps", f"{s['i']:03d}")
    shutil.rmtree(tmp + "/w/record"); shutil.copytree(d, tmp + "/w/record")
    py = open(d + "/record.py", "rb").read() if os.path.exists(d + "/record.py") else b""
    base_py = py if s["i"] == 0 else base_py
    s["changed"] = py != base_py
    s["visible_green"] = visible(tmp + "/w")
    s["hidden_pass"] = judge(grade.run_tests(tmp + "/w", fix_test))
shutil.rmtree(tmp, ignore_errors=True)
green = next((s for s in snaps if s["changed"] and s["visible_green"]), None)
first_hidden = next((s for s in snaps if s["hidden_pass"]), None)

forwarded = [c for c in calls if isinstance(c.get("status"), int)]
ok = [c for c in forwarded if c["status"] == 200]
upstream_errors = [c for c in forwarded if c["status"] != 200]


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
    "passed": passed, "hidden": final.get(hidden, "missing"), "pass_to_pass_broken": broken,
    "calls": len(forwarded), "calls_ok": len(ok), "upstream_errors": [[c["status"], c.get("error", "")[:200]] for c in upstream_errors][:10],
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
