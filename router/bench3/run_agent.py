"""bench-3: one agent attempt in a sealed Agent Server container (OpenHands SDK, pinned).

usage: run_agent.py TASK_MD WORKSPACE_DIR OUT_DIR IMAGE MAX_ITERATIONS
Only WORKSPACE_DIR is mounted (at /workspace). Writes OUT_DIR/events.jsonl, OUT_DIR/run.json and
OUT_DIR/snaps/NNN/: a copy of WORKSPACE_DIR/record each time its contents change, with the number of
actions taken so far. The snapshots are graded afterwards (measure.py); the agent never sees them.
Same agent, prompt and cap as bench-2; the only addition is the snapshot callback.
Model and key come from LLM_MODEL and LLM_API_KEY; the key is never written out.
"""
import hashlib, json, os, shutil, sys, time, traceback

task_md, ws, out, image, maxit = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
os.makedirs(os.path.join(out, "snaps"), exist_ok=True)
run = {"image": image, "max_iterations": maxit, "model": os.environ.get("LLM_MODEL"), "error": None,
       "snaps": [], "snapshot_errors": []}
t0 = time.time()
events, state = [], {"actions": 0, "last": None}
rec = os.path.join(ws, "record")


def tree_hash(d):
    h = hashlib.sha256()
    for root, dirs, files in os.walk(d):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__")
        for f in sorted(files):
            p = os.path.join(root, f)
            if os.path.isfile(p) and os.path.getsize(p) < 2_000_000:
                h.update(os.path.relpath(p, d).encode() + b"\0" + open(p, "rb").read() + b"\0")
    return h.hexdigest()


def snapshot():
    try:
        h = tree_hash(rec)
        if h != state["last"]:
            i = len(run["snaps"])
            shutil.copytree(rec, os.path.join(out, "snaps", f"{i:03d}"), ignore=shutil.ignore_patterns("__pycache__"))
            run["snaps"].append({"i": i, "actions": state["actions"], "hash": h, "t": round(time.time() - t0, 1)})
            state["last"] = h
    except Exception as e:  # a snapshot must never break an attempt
        run["snapshot_errors"].append(f"{type(e).__name__}: {e}"[:300])


def on_event(ev):
    kind = getattr(ev, "kind", type(ev).__name__)
    if kind == "ActionEvent":
        state["actions"] += 1
    elif kind == "ObservationEvent":
        snapshot()


try:
    from pydantic import SecretStr
    from openhands.sdk import LLM, Conversation
    from openhands.tools.preset.default import get_default_agent
    from openhands.workspace import DockerWorkspace
    import openhands.sdk as sdk
    run["sdk_version"] = getattr(sdk, "__version__", None)
    llm = LLM(usage_id="agent", model=os.environ["LLM_MODEL"], api_key=SecretStr(os.environ["LLM_API_KEY"]))
    task = open(task_md).read().replace("of this repository", "of this repository, which is in /workspace")
    snapshot()  # snapshot 0: the untouched base
    with DockerWorkspace(server_image=image, volumes=[f"{ws}:/workspace"]) as w:
        conv = Conversation(agent=get_default_agent(llm=llm), workspace=w, max_iteration_per_run=maxit,
                            callbacks=[on_event])
        conv.send_message(task)
        try:
            conv.run()
        except Exception as e:  # the agent loop failing is a result, not a harness crash
            run["error"] = f"run: {type(e).__name__}: {e}"[:2000]
        snapshot()
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
print(json.dumps({k: v for k, v in run.items() if k not in ("traceback", "snaps")})[:3000])
