"""bench-2: one agent attempt in a sealed Agent Server container (OpenHands SDK, pinned).

usage: run_agent.py TASK_MD WORKSPACE_DIR OUT_DIR IMAGE MAX_ITERATIONS
Only WORKSPACE_DIR is mounted (at /workspace). Writes OUT_DIR/events.jsonl and OUT_DIR/run.json.
Model and key come from LLM_MODEL and LLM_API_KEY; the key is never written out.
"""
import json, os, sys, time, traceback

task_md, ws, out, image, maxit = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
os.makedirs(out, exist_ok=True)
run = {"image": image, "max_iterations": maxit, "model": os.environ.get("LLM_MODEL"), "error": None}
t0 = time.time()
events = []
try:
    from pydantic import SecretStr
    from openhands.sdk import LLM, Conversation
    from openhands.tools.preset.default import get_default_agent
    from openhands.workspace import DockerWorkspace
    import openhands.sdk as sdk
    run["sdk_version"] = getattr(sdk, "__version__", None)
    llm = LLM(usage_id="agent", model=os.environ["LLM_MODEL"], api_key=SecretStr(os.environ["LLM_API_KEY"]))
    task = open(task_md).read().replace("of this repository", "of this repository, which is in /workspace")
    with DockerWorkspace(server_image=image, volumes=[f"{ws}:/workspace"]) as w:
        conv = Conversation(agent=get_default_agent(llm=llm), workspace=w, max_iteration_per_run=maxit)
        conv.send_message(task)
        try:
            conv.run()
        except Exception as e:  # the agent loop failing is a result, not a harness crash
            run["error"] = f"run: {type(e).__name__}: {e}"[:2000]
        try:
            for e in conv.state.events:
                try:
                    events.append(json.loads(e.model_dump_json()))
                except Exception:
                    events.append({"kind": type(e).__name__, "unserialisable": True})
        except Exception as e:
            run["events_error"] = f"{type(e).__name__}: {e}"[:500]
        try:
            m = conv.conversation_stats.get_combined_metrics()
            run["llm_calls"] = len(getattr(m, "token_usages", []) or [])
            tu = getattr(m, "accumulated_token_usage", None)
            run["tokens"] = json.loads(tu.model_dump_json()) if tu is not None else None
            run["cost_usd"] = getattr(m, "accumulated_cost", None)
        except Exception as e:
            run["metrics_error"] = f"{type(e).__name__}: {e}"[:500]
        try:
            run["status"] = str(conv.state.execution_status)
        except Exception:
            pass
except Exception as e:
    run["error"] = run["error"] or f"harness: {type(e).__name__}: {e}"[:2000]
    run["traceback"] = traceback.format_exc()[-3000:]
run["wall_s"] = round(time.time() - t0, 1)
with open(os.path.join(out, "events.jsonl"), "w") as f:
    for e in events:
        f.write(json.dumps(e) + "\n")
json.dump(run, open(os.path.join(out, "run.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in run.items() if k != "traceback"})[:3000])
