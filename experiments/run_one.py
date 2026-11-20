#!/usr/bin/env python3
"""Run one experiment and apply its own rule (experiments/README.md). python3 -I experiments/run_one.py NAME OUT_DIR
Exit 0 with OUT_DIR/result.json and verdict.json whatever the verdict; exit 1 if the experiment could not run or its
rule gave no valid verdict (verdict.json then says invalid, with the reason)."""
import importlib.util, json, os, re, subprocess, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
VERDICTS = ("holds", "fails", "invalid")


def load(name):
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise SystemExit(f"bad experiment name: {name!r}")
    path = os.path.join(HERE, name + ".py")
    spec = importlib.util.spec_from_file_location(f"experiment_{name}", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for k in ("PLAN", "run", "rule"):
        if not hasattr(m, k):
            raise SystemExit(f"{name}: no {k} (see experiments/README.md)")
    return m, path


def main(name, out):
    os.makedirs(out, exist_ok=True)
    m, path = load(name)
    commit = subprocess.run(["git", "log", "-1", "--format=%H %cI", "--", path], capture_output=True, text=True, cwd=HERE).stdout.strip()
    meta = {"name": name, "plan": m.PLAN, "file": os.path.relpath(path, os.path.dirname(HERE)), "file_commit": commit,
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        result = m.run()
        json.dumps(result)
    except Exception:
        v = {"verdict": "invalid", "why": "run() failed: " + traceback.format_exc()[-1500:]}
        json.dump(dict(meta, verdict=v), open(os.path.join(out, "verdict.json"), "w"), indent=1)
        print(json.dumps(v)); return 1
    json.dump(result, open(os.path.join(out, "result.json"), "w"), indent=1)
    try:
        v = m.rule(result)
        ok = isinstance(v, dict) and v.get("verdict") in VERDICTS and isinstance(v.get("why"), str)
    except Exception:
        v, ok = {"why": "rule() failed: " + traceback.format_exc()[-1500:]}, False
    if not ok:
        v = {"verdict": "invalid", "why": f"rule gave no valid verdict: {str(v)[:500]}"}
    json.dump(dict(meta, verdict=v, finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
              open(os.path.join(out, "verdict.json"), "w"), indent=1)
    print(json.dumps(v))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
