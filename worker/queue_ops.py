"""Queue worker, claims and leases (queue P8). Git is the coordinator: a claim is a commit to the queue branch, and
git refuses the second of two concurrent pushes, so two workers can't hold one item.

python3 worker/queue_ops.py claim WT WORKER_ID OUT_JSON      # WT: a worktree of the queue branch
python3 worker/queue_ops.py release WT WORKER_ID ITEM STATUS NOTE_JSON
python3 worker/queue_ops.py ready WT                          # list ready items (no write)

Rules (P8, owner 2026-11-18; a hypothesis to measure, not a final design):
- ready: status "open", every blocker done, the item carries a job the worker can do, and no live claim touches any
  of its "touches". Job types, by what the item carries: "targets" (exact files and line ranges a script can check)
  -> a doc job (worker/doc_job.py); "research" (a question and search queries) -> a research job that writes a
  literature card (worker/research_job.py, queue P7);
- a lease lasts LEASE_HOURS; an expired worker lease reopens its item; a chat's claim ("chat-...") always wins and is
  never expired or taken by a worker;
- order is the queue's own order (it is ordered by how the work actually happens).
Exit codes: 0 done (claim: an item was claimed, or none was ready, see OUT_JSON); 1 lost the race too often; 3 failure.
"""
import datetime as dt, json, os, subprocess, sys

QUEUE = "audit/2026-11-17/queue.json"
RENDER = "audit/2026-11-17/render.py"
QUEUE_MD = "audit/2026-11-17/QUEUE.md"
BRANCH = "pilot/lgn-20261115"
LEASE_HOURS = 2
TRIES = 5


def now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse(s):
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


def git(wt, *a, check=True):
    return subprocess.run(["git", "-C", wt, *a], capture_output=True, text=True, check=check)


def sync(wt):
    git(wt, "fetch", "-q", "--depth", "20", "origin", BRANCH)
    git(wt, "reset", "-q", "--hard", "FETCH_HEAD")


def load(wt):
    return json.load(open(os.path.join(wt, QUEUE)))


def save(wt, q):
    p = os.path.join(wt, QUEUE)
    with open(p, "w") as f:
        json.dump(q, f, indent=1, ensure_ascii=False)
        f.write("\n")
    r = subprocess.run([sys.executable, "-I", os.path.join(wt, RENDER), p], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"render.py refused the queue: {r.stderr[-500:]}")
    open(os.path.join(wt, QUEUE_MD), "w").write(r.stdout)


def live_claim(i, t):
    """True if the item is held: by a chat (always), or by a worker whose lease has not expired."""
    if i.get("status") != "in progress" or not i.get("claimed_by"):
        return False
    if not i["claimed_by"].startswith("worker-"):
        return True
    return parse(i.get("lease_until", "1970-01-01T00:00:00Z")) > t


def expire(q, t):
    """Reopen items whose worker lease has run out. Returns their ids."""
    out = []
    for i in q["items"]:
        if i.get("status") == "in progress" and str(i.get("claimed_by", "")).startswith("worker-") and not live_claim(i, t):
            i["status"] = "open"
            i.setdefault("runs", []).append({"at": iso(t), "worker": i["claimed_by"], "result": "lease expired"})
            for k in ("claimed_by", "claimed_at", "lease_until"):
                i.pop(k, None)
            out.append(i["id"])
    return out


def ready(q, t):
    by = {i["id"]: i for i in q["items"]}
    held = set()
    for i in q["items"]:
        if live_claim(i, t):
            held.update(i.get("touches", []))
    out = []
    for i in q["items"]:
        if i.get("status") != "open" or not (i.get("targets") or i.get("research")):
            continue
        if any(by.get(b, {}).get("status") != "done" for b in i.get("blocked_by", [])):
            continue
        if any(_overlap(x, h) for x in i.get("touches", []) for h in held):
            continue
        out.append(i)
    return out


def _overlap(a, b):
    a, b = a.rstrip("/"), b.rstrip("/")
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def job_type(i):
    return "doc" if i.get("targets") else "research" if i.get("research") else None


def commit_push(wt, msg):
    git(wt, "add", QUEUE, QUEUE_MD)
    git(wt, "-c", "user.name=exagium-worker", "-c", "user.email=worker@exagium.invalid", "commit", "-qm", msg)
    return git(wt, "push", "-q", "origin", f"HEAD:{BRANCH}", check=False).returncode == 0


def claim(wt, worker, out):
    for attempt in range(TRIES):
        sync(wt)
        q, t = load(wt), now()
        expired = expire(q, t)
        r = ready(q, t)
        if not r:
            if expired:
                save(wt, q)
                if not commit_push(wt, f"worker: leases expired {expired}"):
                    continue
            json.dump({"item": None, "expired": expired, "attempts": attempt + 1}, open(out, "w"))
            return 0
        i = r[0]
        i.update(status="in progress", claimed_by=worker, claimed_at=iso(t), lease_until=iso(t + dt.timedelta(hours=LEASE_HOURS)))
        save(wt, q)
        if commit_push(wt, f"worker: {worker} claims {i['id']}"):
            json.dump({"item": i, "expired": expired, "attempts": attempt + 1}, open(out, "w"), indent=1)
            return 0
    json.dump({"item": None, "lost_races": TRIES}, open(out, "w"))
    return 1


def release(wt, worker, item_id, status, note_path):
    note = json.load(open(note_path))
    for _ in range(TRIES):
        sync(wt)
        q = load(wt)
        i = next((x for x in q["items"] if x["id"] == item_id), None)
        if i is None:
            raise RuntimeError(f"no item {item_id}")
        lost = i.get("claimed_by") != worker
        i.setdefault("runs", []).append(dict(note, worker=worker, at=iso(now()), lost_lease=lost))
        if not lost:
            i["status"] = status
            for k in ("claimed_by", "claimed_at", "lease_until"):
                i.pop(k, None)
        save(wt, q)
        if commit_push(wt, f"worker: {worker} {'releases' if not lost else 'records (lease lost)'} {item_id} -> {status}"):
            return 0
    return 1


def main(a):
    if a[0] == "claim":
        return claim(a[1], a[2], a[3])
    if a[0] == "release":
        return release(a[1], a[2], a[3], a[4], a[5])
    if a[0] == "ready":
        for i in ready(load(a[1]), now()):
            print(i["id"], i["title"][:100])
        return 0
    raise SystemExit("usage: see the docstring")


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except (RuntimeError, subprocess.CalledProcessError) as e:
        print("FAILURE", e, getattr(e, "stderr", ""), file=sys.stderr)
        sys.exit(3)
