"""First experiment, and a canary for the executor: how many live queue items does the classifier mark already_done?"""
import json, os, subprocess

PLAN = ("Measure: of the live queue items (not done or parked) that worker/classify.py has judged, the share with "
        "already_done >= 0.5. Rule, fixed before the run: holds if that share is at least 0.15 (status drift is a "
        "material lever, as Claude's first reading of 54 untyped items suggested: about 12); fails if under 0.15; "
        "invalid if fewer than 20 items have been judged.")


def run():
    raw = subprocess.run(["git", "show", "origin/pilot/lgn-20261115:audit/2026-11-17/queue.json"], capture_output=True, text=True,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    q = json.loads(raw.stdout)
    live = [i for i in q["items"] if i.get("status") not in ("done", "parked") and i.get("work_types")]
    drift = [i["id"] for i in live if i["work_types"].get("already_done", 0) >= 0.5]
    return {"judged": len(live), "already_done": drift, "share": round(len(drift) / len(live), 3) if live else None}


def rule(r):
    if r["judged"] < 20:
        return {"verdict": "invalid", "why": f"only {r['judged']} items judged (need 20)"}
    return {"verdict": "holds" if r["share"] >= 0.15 else "fails", "why": f"{len(r['already_done'])} of {r['judged']} ({r['share']})"}
