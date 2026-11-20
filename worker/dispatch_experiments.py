#!/usr/bin/env python3
"""Start the experiments that open queue items name (experiments/README.md), by script; no model.
    python3 -I worker/dispatch_experiments.py QUEUE_WORKTREE REPO [MAX]
An item carries {"experiment": {"name": NAME}}; when experiments/NAME.py exists on main and the item is open, it moves
open -> scheduled (outcome dispatch) and experiment.yml starts with the item; that run's record job moves it on."""
import json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import queue_ops as Q  # noqa: E402


def ready(q, repo):
    out = []
    for i in q["items"]:
        n = (i.get("experiment") or {}).get("name", "")
        if i.get("status") == "open" and re.fullmatch(r"[a-z0-9_]+", n or "") and os.path.isfile(os.path.join(repo, "experiments", n + ".py")) \
                and all(next((x for x in q["items"] if x["id"] == b), {}).get("status") == "done" for b in i.get("blocked_by", [])):
            out.append((i["id"], n))
    return out


def main(wt, repo, n=3, start=None):
    Q.sync(wt)
    todo = ready(Q.load(wt), repo)[:n]
    start = start or (lambda name, item: subprocess.run(["gh", "workflow", "run", "experiment.yml", "-R", os.environ.get("GITHUB_REPOSITORY", ""),
                                                          "--ref", "main", "-f", f"name={name}", "-f", f"item={item}"]).returncode)
    started = []
    for item, name in todo:
        if Q.move(wt, "experiment-dispatch", item, "dispatch", f"experiment {name} started") != 0:
            continue
        if start(name, item) == 0:
            started.append(item)
        else:   # the move happened but the run did not start: the record of a lost run puts it back
            Q.move(wt, "experiment-dispatch", item, "lost", f"experiment {name} could not be started", allow_failed=True)
    print(f"experiments started: {started}")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit(main(a[0], a[1], int(a[2]) if len(a) > 2 else 3))
