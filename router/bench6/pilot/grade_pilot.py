"""P28 step 3: apply PLAN.md's pilot rule to agent-*-model.json and the meter logs beside them. Endings come from the
meter's own log (a 'refused_cap' line means the attempt reached the cap), not from the attempt record's label.
python3 -I grade_pilot.py DIR"""
import glob, json, os, sys

KEEP_MIN, N = 6, 8   # PLAN.md, step 3


def grade(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "**", "agent-*-model.json"), recursive=True)):
        r = json.load(open(f))
        m = glob.glob(os.path.join(os.path.dirname(f), f"meter-{r['id']}.jsonl"))
        log = [json.loads(l) for l in open(m[0]) if l.strip()] if m else []
        calls = sum(1 for x in log if x.get("status") == 200)
        capped = any(x.get("status") == "refused_cap" for x in log)
        started = r["ending"] in ("exited", "timeout", "call_cap")
        end = "not started" if not started else "call_cap" if capped else "timeout" if r["ending"] == "timeout" or r["wall_s"] > r["wall_cap_s"] else "exited"
        status = (r.get("harness_end") or {}).get("status") or (r.get("harness_end") or {}).get("error", "")[:40]
        rows.append((r["id"], end, calls, r["wall_s"], r["resolved"], status))
        print(f"{r['id']:<42} {end:<11} calls {calls:>3}  wall {r['wall_s']:>7}s  resolved {str(r['resolved']):<5} {status}")
    kept = sum(1 for x in rows if x[1] == "exited")
    started = sum(1 for x in rows if x[1] != "not started")
    print(f"kept {kept} of {started} started ({len(rows)} records); resolved {sum(1 for x in rows if x[4])} of {len(rows)}; "
          f"rule: at least {KEEP_MIN} of {N} kept -> " + ("met" if started == N and kept >= KEEP_MIN else
          "not met: the fresh set is too hard for this pilot" if started == N else "incomplete"))


if __name__ == "__main__":
    grade(sys.argv[1])
