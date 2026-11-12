"""bench-3 measures for one attempt, from its own files. Standard library only.

usage: measure.py OUT_DIR WORKSPACE FIX_TEST HIDDEN_TEST P2P_JSON META_JSON > result.json
OUT_DIR holds run.json, events.jsonl, agent.diff, egress.json and snaps/ (see run_agent.py).
META_JSON: {"task","arm","model","rep","commit","run_id","pilot","control"}.

Fixes bench-2's three measurement bugs:
  - the diff counts content lines only (agent.diff is made with core.fileMode=false, without conversations/);
  - model calls = distinct llm_response_id in the events (the SDK's token_usages list was empty);
  - the control gets its own label, and every ending gets exactly one (decide.classify).
New: each snapshot of record/ is graded twice, by the agent's own tests ("visible") and by the grader
("hidden"), to find where the attempt first went green and whether stopping there would have kept the pass.
"""
import json, os, re, shutil, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decide, grade

out, ws, fix_test, hidden, p2p_file, meta = sys.argv[1:6] + [json.loads(sys.argv[6])]
run = json.load(open(out + "/run.json"))
events = [json.loads(l) for l in open(out + "/events.jsonl") if l.strip()]
p2p = json.load(open(p2p_file))


def judge(res):
    return res.get(hidden) == "pass" and not [t for t in p2p if res.get(t) != "pass"]


def visible(workdir):
    """The agent's own test file, as it left it: green means at least one pass and no fail or error."""
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


# final grade (same grader and rule as bench-1 and bench-2)
final = grade.run_tests(ws, fix_test)
broken = [t for t in p2p if final.get(t) != "pass"]
passed = final.get(hidden) == "pass" and not broken

# shadow grades, one per snapshot
snaps = run.get("snaps") or []
tmp = tempfile.mkdtemp()
shutil.copytree(ws, tmp + "/w", ignore=shutil.ignore_patterns(".git", "conversations", "__pycache__"))
base_py = None
for s in snaps:
    d = os.path.join(out, "snaps", f"{s['i']:03d}")
    shutil.rmtree(tmp + "/w/record"); shutil.copytree(d, tmp + "/w/record", symlinks=True)
    py = open(d + "/record.py", "rb").read() if os.path.isfile(d + "/record.py") and not os.path.islink(d + "/record.py") else b""
    base_py = py if s["i"] == 0 else base_py
    s["changed"] = py != base_py
    s["visible_green"] = visible(tmp + "/w")          # before the grader overwrites the test file
    s["hidden_pass"] = judge(grade.run_tests(tmp + "/w", fix_test))
shutil.rmtree(tmp, ignore_errors=True)
green = next((s for s in snaps if s["changed"] and s["visible_green"]), None)
first_hidden = next((s for s in snaps if s["hidden_pass"]), None)

# the agent's own trail
kinds, fetched, outside, ids = {}, [], [], set()
for e in events:
    k = e.get("kind", "?"); kinds[k] = kinds.get(k, 0) + 1
    if e.get("llm_response_id"):
        ids.add(e["llm_response_id"])
    if k == "ActionEvent":
        s = json.dumps(e.get("action", {}))
        fetched += re.findall(r"https?://[^\s\"'\\]+", s)
        outside += re.findall(r"/home/runner[^\s\"'\\]*|router/bench[^\s\"'\\]*", s)
    if k == "ObservationEvent":
        outside += re.findall(r"/home/runner/work[^\s\"'\\]*", json.dumps(e.get("observation", {})))

# content lines changed, per file
files, cur = {}, None
if os.path.exists(out + "/agent.diff"):
    for line in open(out + "/agent.diff", errors="replace"):
        m = re.match(r"diff --git a/(.*) b/", line)
        if m:
            cur = files.setdefault(m.group(1), [0, 0])
        elif cur is not None and line.startswith("+") and not line.startswith("+++"):
            cur[0] += 1
        elif cur is not None and line.startswith("-") and not line.startswith("---"):
            cur[1] += 1
egress = json.load(open(out + "/egress.json")) if os.path.exists(out + "/egress.json") else {}

print(json.dumps(dict(meta, **{
    "end": decide.classify(run, events, control=bool(meta.get("control"))),
    "passed": passed, "hidden": final.get(hidden, "missing"), "pass_to_pass_broken": broken,
    "clean": not fetched and not outside,
    "web_fetches": sorted(set(fetched))[:20], "outside_paths": sorted(set(outside))[:20],
    "calls": len(ids) if events else None, "actions": kinds.get("ActionEvent", 0), "event_kinds": kinds,
    "status": run.get("status"), "error": (run.get("error") or "")[:500],
    "error_codes": [e.get("code") for e in events if e.get("kind") == "ConversationErrorEvent"],
    "tokens": run.get("tokens"), "wall_s": run.get("wall_s"), "sdk_version": run.get("sdk_version"), "image": run.get("image"),
    "diff_files": files, "n_snaps": len(snaps), "snapshot_errors": run.get("snapshot_errors") or [],
    "green_actions": green["actions"] if green else None, "green_kept": green["hidden_pass"] if green else None,
    "first_hidden_pass_actions": first_hidden["actions"] if first_hidden else None,
    "snaps": [{k: s[k] for k in ("i", "actions", "t", "changed", "visible_green", "hidden_pass")} for s in snaps],
    "egress_drops": egress.get("drops"), "egress": egress,
    "requests_left_before": run.get("requests_left_before"),
    "grade_results": final})))
