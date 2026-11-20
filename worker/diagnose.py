"""Failed runs, classified by script (owner 2026-11-20: Claude should not have to find every fault by reading logs).

python3 -I worker/diagnose.py scan HOURS OUT_JSON          (needs GH_TOKEN; reads failed runs' job logs)
python3 -I worker/diagnose.py file QUEUE_WT DIAG_JSON       (files one escalated item per new code-bug signature)

Every failed workflow run in the window gets one class from the first rule that matches its failing job's log tail.
Classes and what the system does with each:
  budget        a cap or allowance reached (ours or a provider's)          -> nothing: the worker waits or falls back
  rate_limited  provider throttling (429 upstream)                          -> nothing: the fallback chain moves on
  model_gone    404/410 on a model                                          -> the wiring check names the dependant
  code_bug      a Python traceback in our code (file:line)                  -> an escalated queue item, one per signature
  fact_changed  facts/check.py found a changed registered fact             -> the wiring check's own alarm
  test_failed   unittest failures                                           -> an escalated item (tests must pass on main)
  push_race     git push rejected (another writer first)                    -> nothing: the writers retry
  transient     a network or server error (5xx, timeouts, package mirrors)  -> nothing: a rerun clears it
  auth          401/403 from a provider or GitHub                           -> an escalated item (a key or a permission)
  unknown       none of the above                                           -> an escalated item, so it gets a rule
Rules are ordered from specific to general; the samples they were written from are the failed runs of 2026-11-19/20.
"""
import json, os, re, subprocess, sys

RULES = [
    ("auth", re.compile(r"\b(401|403)\b.*(Unauthorized|Forbidden)|HTTP Error 40[13]", re.I)),
    ("fact_changed", re.compile(r"^CHANGED \S+: registered", re.M)),
    ("budget", re.compile(r"cap reached|daily free allocation|no budget|daily dollar cap|budget headroom", re.I)),
    ("rate_limited", re.compile(r"rate-limited upstream|\b429\b|Too Many Requests", re.I)),
    ("model_gone", re.compile(r"\b(410|404)\b.*model|model .*(end of life|not found|deprecated)", re.I)),
    ("test_failed", re.compile(r"^FAILED \((failures|errors)=\d+", re.M)),
    ("code_bug", re.compile(r'File "[^"]*/(worker|router|runner|facts|transfer)/[^"]+\.py", line \d+')),
    ("transient", re.compile(r"\b(502|503|504)\b|Service Unavailable|Temporary failure|Connection reset|timed out|Unable to fetch some archives", re.I)),
    ("push_race", re.compile(r"\[rejected\].*\(fetch first\)|failed to push some refs", re.I)),
]
ACT = {"transient": "rerun (no action; repeats become unknown)", "auth": "escalated item (a key or permission: often the owner's)", "budget": "wait or fall back (no action)", "rate_limited": "fallback chain (no action)",
       "model_gone": "wiring check names the dependant", "fact_changed": "wiring check alarm",
       "test_failed": "escalated item", "code_bug": "escalated item", "push_race": "writers retry (no action)",
       "unknown": "escalated item"}


def classes():
    """Sorted list of every failure class name in RULES plus 'unknown'."""
    return sorted([name for name, _ in RULES] + ["unknown"])


def classify(log):
    tail = "\n".join(log.splitlines()[-400:])
    for name, rx in RULES:
        if rx.search(tail):
            return name
    return "unknown"


def signature(log):
    """For a code bug: the last frame in our code and the exception line, as a stable signature."""
    frames = re.findall(r'File "[^"]*/((?:worker|router|runner|facts|transfer)/[^"]+\.py)", line (\d+), in (\S+)', log)
    exc = re.findall(r"^(?:\S+\s)?(\w+(?:Error|Exception)): (.{0,120})", log, re.M)
    if not frames:
        return None
    f = frames[-1]
    return f"{f[0]}:{f[2]} {exc[-1][0] if exc else ''}".strip()


def error_tail(log):
    """The lines just before the first '##[error]' (the run's cleanup lines come after it and say nothing)."""
    ls = [l for l in log.splitlines() if l.strip() and "[36;1m" not in l]
    k = next((n for n, l in enumerate(ls) if "##[error]" in l), len(ls))
    return ls[max(0, k - 6):k + 1]


def fixed_since(path, at):
    """True if main changed the file after the failed run started (the bug may be fixed: not filed)."""
    t = subprocess.run(["git", "log", "-1", "--format=%cI", "--", path], capture_output=True, text=True).stdout.strip()
    if not t:
        return False
    import datetime as dt
    return dt.datetime.fromisoformat(t) > dt.datetime.fromisoformat(at.replace("Z", "+00:00"))


def gh(*a):
    return subprocess.run(["gh", "api", *a], capture_output=True, text=True).stdout


def scan(hours, out):
    import datetime as dt
    repo = os.environ.get("GITHUB_REPOSITORY", "5kasgard/exagium")
    since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=float(hours))).strftime("%Y-%m-%dT%H:%M:%SZ")
    runs = json.loads(gh(f"repos/{repo}/actions/runs?per_page=100&status=failure&created=>={since}", "-q",
                         "[.workflow_runs[] | {id, name, created_at}]") or "[]")
    rows = []
    for r in runs:
        jid = gh(f"repos/{repo}/actions/runs/{r['id']}/jobs", "-q", '.jobs[] | select(.conclusion=="failure") | .id').split()
        log = gh(f"repos/{repo}/actions/jobs/{jid[0]}/logs") if jid else ""
        log = re.sub(r"^\S+Z ", "", log, flags=re.M)   # drop timestamps
        c = classify(log)
        rows.append({"run": r["id"], "workflow": r["name"], "at": r["created_at"], "class": c,
                     "signature": signature(log) if c == "code_bug" else None,
                     "tail": error_tail(log) if c in ("unknown", "code_bug", "test_failed", "auth") else []})
    json.dump({"since": since, "runs": rows}, open(out, "w"), indent=1)
    by = {}
    for x in rows:
        by[x["class"]] = by.get(x["class"], 0) + 1
    print(f"diagnose: {len(rows)} failed runs since {since}: " + ", ".join(f"{k} {v}" for k, v in sorted(by.items())))
    return 0


def report(diag):
    """Digest lines: failures by class, with what the system does with each."""
    rows = diag.get("runs", [])
    by = {}
    for x in rows:
        by.setdefault(x["class"], []).append(x)
    out = [f"- {len(rows)} failed runs"]
    for c, xs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        names = sorted({x["workflow"] for x in xs})
        sigs = sorted({x["signature"] for x in xs if x.get("signature")})
        out.append(f"- {c} {len(xs)} ({', '.join(names)}) -> {ACT[c]}" + (f"; {'; '.join(sigs)}" if sigs else ""))
    return out


def file(wt, diag_path):
    """One escalated item per new code-bug signature or unknown failure (deduplicated against existing items)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import queue_ops as Q
    diag = json.load(open(diag_path))
    want = {}
    for x in diag["runs"]:
        if x["class"] in ("code_bug", "test_failed", "unknown", "auth"):
            if x.get("signature") and fixed_since(x["signature"].split(":")[0], x["at"]):
                continue   # the file changed on main after this run: probably fixed; a new failure will file it again
            key = x.get("signature") or f"{x['workflow']}:{x['class']}"
            want.setdefault(key, x)
    if not want:
        print("diagnose: nothing to file"); return 0
    for _ in range(Q.TRIES):
        Q.sync(wt)
        q = Q.load(wt)
        have = {i.get("diagnosis") for i in q["items"] if i.get("diagnosis")}
        new = [(k, x) for k, x in want.items() if k not in have]
        if not new:
            print("diagnose: all signatures already filed"); return 0
        n = max([int(m.group(1)) for i in q["items"] for m in [re.fullmatch(r"D(\d+)", str(i["id"]))] if m] or [0]) + 1
        made = []
        for k, x in new[:3]:
            iid = f"D{n}"; n += 1
            q["items"].append({"id": iid, "title": f"Failed run, {x['class']}: {k}"[:160], "kind": "fix", "status": "escalated",
                               "origin": "generated", "diagnosis": k, "generated_at": Q.iso(Q.now()),
                               "next": f"Filed by worker/diagnose.py from run {x['run']} ({x['workflow']}). Log tail: "
                                       + " | ".join(x.get("tail") or [])[:600],
                               "refs": [f"https://github.com/{os.environ.get('GITHUB_REPOSITORY', '5kasgard/exagium')}/actions/runs/{x['run']}"],
                               "touches": [k.split(":")[0]] if x.get("signature") else [".github/workflows/"], "severity": "minor", "needs": "free", "cost": 0, "value": "medium",
                               "blocked_by": [], "batch": "B3"})
            made.append(iid)
        Q.save(wt, q)
        if Q.commit_push(wt, f"diagnose: {', '.join(made)} filed (escalated)"):
            print("diagnose: filed", made); return 0
    return 1


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit(scan(a[1], a[2]) if a[0] == "scan" else file(a[1], a[2]))
