"""bench-2 measures for one attempt, from its own logs. Standard library only.

usage: measure.py RUN_DIR GRADE_JSON DIFFSTAT > result.json
Fixes bench-1's three measurement bugs:
  - diff comes from `git diff --shortstat HEAD` (passed in);
  - leak = the agent's own actions or observations name a path outside its workspace
    (/home/runner, router/bench) or fetch a URL; README text no longer counts;
  - rate limits are counted from error records only.
"""
import json, re, sys

run_dir, grade, diffstat = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
run = json.load(open(run_dir + "/run.json"))
events = [json.loads(l) for l in open(run_dir + "/events.jsonl") if l.strip()]
kinds, fetched, outside, rate = {}, [], [], 0
for e in events:
    k = e.get("kind", "?"); kinds[k] = kinds.get(k, 0) + 1
    if k == "ActionEvent":
        s = json.dumps(e.get("action", {}))
        fetched += re.findall(r"https?://[^\s\"'\\]+", s)
        outside += re.findall(r"/home/runner[^\s\"'\\]*|router/bench[^\s\"'\\]*", s)
    if k == "ObservationEvent":
        s = json.dumps(e.get("observation", {}))
        outside += re.findall(r"/home/runner/work[^\s\"'\\]*", s)
    if k == "ConversationErrorEvent" and re.search(r"RateLimit|429|RESOURCE_EXHAUSTED", json.dumps(e)):
        rate += 1
if run.get("error") and re.search(r"RateLimit|429|RESOURCE_EXHAUSTED", run["error"]):
    rate += 1
err = run.get("error") or ""
end = ("harness_crash" if err.startswith("harness") else
       "provider_error" if rate else
       "agent_error" if err else
       "max_iterations" if "max" in str(run.get("status", "")).lower() or kinds.get("ActionEvent", 0) >= run.get("max_iterations", 10**9) else
       "finished")
print(json.dumps({
    "passed": grade["hidden"] == "pass" and not grade["pass_to_pass_broken"],
    "hidden": grade["hidden"], "pass_to_pass_broken": grade["pass_to_pass_broken"],
    "end": end, "status": run.get("status"), "error": err[:500],
    "llm_calls": run.get("llm_calls"), "actions": kinds.get("ActionEvent", 0), "event_kinds": kinds,
    "tokens": run.get("tokens"), "wall_s": run.get("wall_s"), "sdk_version": run.get("sdk_version"),
    "image": run.get("image"), "rate_limit_errors": rate, "diff": diffstat.strip(),
    "web_fetches": sorted(set(fetched))[:20], "outside_paths": sorted(set(outside))[:20],
    "clean": not fetched and not outside,
    "grade_results": grade["results"]}))
