"""Measure Cloudflare Containers against the Forge's sandbox criteria (audit/fabric/sandbox-decision-2026-11-25.md,
sec. 5; issue #191), the same protocol as e2b_eval.py, through the forge-cfbox Worker (fabric/cfbox/worker.js).
Standard library only. Two carry-over modes run side by side (see worker.js): snap (disk snapshot, then stop) and
native (left to stop by itself after 10 idle minutes). State: one JSON file.

  python3 cf_eval.py create STATE     # both modes: jail and CPU checks, marker file, tool install, ticking process
  python3 cf_eval.py check STATE      # both modes: what survived since the last call
  python3 cf_eval.py kill STATE       # end both

Needs CFBOX_TOKEN (the Worker's BOX_TOKEN). recorded_by is written into every record. Kind: measured."""
import json, os, sys, urllib.request, urllib.error
from datetime import datetime, timezone

URL = os.environ.get("CFBOX_URL", "https://cfbox.exagium.xyz")
RUN = f"{os.environ.get('GITHUB_WORKFLOW', 'local')}#{os.environ.get('GITHUB_RUN_ID', '-')}"
WHO = "fabric/sandbox/cf_eval.py (written by Claude Opus 5.5), run " + RUN
MODES = ("snap", "native")


def now():
    return datetime.now(timezone.utc).isoformat()[:19] + "Z"


def call(op, mode):
    req = urllib.request.Request(f"{URL}/{op}?mode={mode}", method="POST", data=b"",
                                 headers={"authorization": "Bearer " + os.environ["CFBOX_TOKEN"],
                                          "user-agent": "forge-sandbox-eval"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:   # create installs a tool on the smallest size: minutes
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"ok": False, "http": e.code, "error": e.read().decode(errors="replace")[:400]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}


def idle_h(since):
    t = datetime.fromisoformat(since.replace("Z", "+00:00"))
    return round((datetime.now(timezone.utc) - t).total_seconds() / 3600, 2)


def create(path):
    rec = {"recorded_by": WHO, "kind": "measured", "provider": "cloudflare-containers", "plan": "Workers Paid",
           "worker": "fabric/cfbox", "created_at": now()}
    for m in MODES:
        r = call("create", m)
        rec[m] = {"create": r, "last_at": now(), "cycles": []}
        print(json.dumps({m: r}, indent=1))
    if not all(rec[m]["create"].get("ok") for m in MODES):   # no state on a failed setup, so a re-create is allowed
        for m in MODES:
            call("kill", m)
        sys.exit("create failed; nothing recorded (review 2026-11-26)")
    json.dump(rec, open(path, "w"), indent=1)


def check(path):
    rec = json.load(open(path))
    for m in MODES:
        cyc = {"at": now(), "recorded_by": WHO, "idle_h": idle_h(rec[m]["last_at"])} | call("check", m)
        if "files_ok" not in cyc:   # the call itself failed: not evidence about survival (review 2026-11-26)
            cyc["infra_error"] = True
        rec[m]["cycles"].append(cyc); rec[m]["last_at"] = now()
        print(json.dumps({m: cyc}, indent=1))
    json.dump(rec, open(path, "w"), indent=1)


def kill(path):
    rec = json.load(open(path))
    killed = all(call("kill", m).get("killed") for m in MODES)
    for m in MODES:
        cs = [c for c in rec[m]["cycles"] if not c.get("infra_error")]
        print(f"{m}: files carried over in {sum(bool(c.get('files_ok')) for c in cs)}/{len(cs)} cycles, "
              f"process in {sum(bool(c.get('ok')) for c in cs)}/{len(cs)}; "
              f"{len(rec[m]['cycles']) - len(cs)} calls failed")
    if not killed:   # leave no killed_at, so the next run tries again (a live container or snapshot keeps billing)
        sys.exit("kill failed; state left open")
    rec["killed_at"] = now(); json.dump(rec, open(path, "w"), indent=1)


if __name__ == "__main__":
    {"create": create, "check": check, "kill": kill}[sys.argv[1]](sys.argv[2])
