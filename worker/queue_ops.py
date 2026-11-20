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
import datetime as dt, json, os, re, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transitions as T  # noqa: E402  the only place an item's next status is decided (Q60)

QUEUE = "audit/2026-11-17/queue.json"
RENDER = "audit/2026-11-17/render.py"
QUEUE_MD = "audit/2026-11-17/QUEUE.md"
BRANCH = "pilot/lgn-20261115"
LEASE_HOURS = 2
TRIES = 5
MAX_FAILURES = T.MAX_FAILURES   # kept for readers; the rule lives in transitions.py
ID = re.compile(r"[A-Za-z0-9._-]+")


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
    env = dict(os.environ, EXAGIUM_TRANSITIONS=os.path.join(os.path.dirname(os.path.abspath(__file__)), "transitions.py"))
    r = subprocess.run([sys.executable, "-I", os.path.join(wt, RENDER), p], capture_output=True, text=True, env=env)
    if r.returncode:
        raise RuntimeError(f"render.py refused the queue: {r.stderr[-500:]}")
    open(os.path.join(wt, QUEUE_MD), "w").write(r.stdout)


def live_claim(i, t):
    """True if the item is held: by a chat for CHAT_TTL_H hours from its claim, or by a worker whose lease has not expired."""
    if i.get("status") != "in progress" or not i.get("claimed_by"):
        return False
    if not i["claimed_by"].startswith("worker-"):
        return parse(i.get("claimed_at", "1970-01-01T00:00:00Z")) >= t - dt.timedelta(hours=CHAT_TTL_H)
    return parse(i.get("lease_until", "1970-01-01T00:00:00Z")) > t


CHAT_TTL_H = 24   # GLM review F017: a chat's claim never expired; P3 and P5 (held since 2026-11-19) blocked 10 open items


def expire(q, t):
    """Reopen items whose worker lease has run out, items 'in progress' with no claim at all (F016: Q58 sat there with no
    exit), and chat claims older than CHAT_TTL_H hours (F017). Returns their ids."""
    out = []
    for i in q["items"]:
        if i.get("status") != "in progress":
            continue
        who = str(i.get("claimed_by") or "")
        stale_chat = who and not who.startswith("worker-") and parse(i.get("claimed_at", "1970-01-01T00:00:00Z")) < t - dt.timedelta(hours=CHAT_TTL_H)
        if not who or stale_chat:
            i["status"] = T.next_status("in progress", "lease_expired")
            i.setdefault("runs", []).append({"at": iso(t), "worker": who or "nobody",
                                             "result": "no claim held" if not who else f"chat claim older than {CHAT_TTL_H} h",
                                             "outcome": "lease_expired"})
            for k in ("claimed_by", "claimed_at", "lease_until"):
                i.pop(k, None)
            out.append(i["id"])
            continue
        if who.startswith("worker-") and not live_claim(i, t):
            i.setdefault("runs", []).append({"at": iso(t), "worker": i["claimed_by"], "result": "lease expired", "outcome": "lease_expired"})
            i["status"] = T.next_status("in progress", "lease_expired", failures(i))   # counted with this run (GLM F018)
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
        if i.get("status") != "open" or not job_type(i) or not ID.fullmatch(str(i.get("id"))) or not covered(i):
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
    return "doc" if i.get("targets") else "research" if i.get("research") else "code" if i.get("code") else None


def covered(i):
    """Everything the job writes is declared in touches, so the overlap rule sees it (review m2)."""
    t = set(i.get("touches", []))
    if job_type(i) == "doc":
        return all(isinstance(x, dict) and x.get("path") in t for x in i["targets"])
    if job_type(i) == "code":
        c = i.get("code") or {}
        return bool(c.get("files")) and all(f in t for f in c["files"] + ([c["test"]] if c.get("test") else []))
    return f"research/cards/{i.get('id')}.md" in t


def failures(i):
    """Consecutive failed runs at the end of the item's history (review N1: a budget refusal is not a failure, and a
    success resets the count)."""
    n = 0
    for r in reversed(i.get("runs", [])):
        res = str(r.get("result", ""))
        if res.startswith("not started") or r.get("outcome") in ("budget", "retry"):
            continue
        if r.get("result") == "lease expired":   # the run was killed before it released: a failure (review of Q60, 10)
            n += 1; continue
        if r.get("outcome") in ("failed", "lost") or (not r.get("outcome") and res.startswith("failed")):   # outcome when recorded (Q60)
            n += 1; continue
        break
    return n


def commit_push(wt, msg):
    git(wt, "add", QUEUE, QUEUE_MD)
    git(wt, "-c", "user.name=exagium-worker", "-c", "user.email=worker@exagium.invalid", "commit", "-qm", msg)
    return git(wt, "push", "-q", "origin", f"HEAD:{BRANCH}", check=False).returncode == 0


def claim(wt, worker, out, types=None):
    """types: job types the budget can still pay for today (None = any). A research item short of Cloudflare neurons must
    not stop doc items that run on other pools (2026-11-19 20:24Z: the worker stopped with E1-E4 ready; Q59)."""
    for attempt in range(TRIES):
        sync(wt)
        q, t = load(wt), now()
        expired = expire(q, t)
        r = [i for i in ready(q, t) if types is None or job_type(i) in types]
        if not r:
            if expired:
                save(wt, q)
                if not commit_push(wt, f"worker: leases expired {expired}"):
                    continue
            json.dump({"item": None, "expired": expired, "attempts": attempt + 1}, open(out, "w"))
            return 0
        i = r[0]
        i.update(status=T.next_status(i["status"], "claim"), claimed_by=worker, claimed_at=iso(t), lease_until=iso(t + dt.timedelta(hours=LEASE_HOURS)))
        save(wt, q)
        if commit_push(wt, f"worker: {worker} claims {i['id']}" + (f" (leases expired {expired})" if expired else "")):
            json.dump({"item": i, "expired": expired, "attempts": attempt + 1}, open(out, "w"), indent=1)
            return 0
    json.dump({"item": None, "lost_races": TRIES}, open(out, "w"))
    return 1


def release(wt, worker, item_id, outcome, note_path):
    """outcome: one of transitions.WORKER; the next status comes from transitions.next_status, never from the caller."""
    if outcome not in T.WORKER:
        raise ValueError(f"not a worker outcome: {outcome!r}")
    try:
        note = json.load(open(note_path))
        assert isinstance(note, dict)
    except (OSError, ValueError, AssertionError):
        note = {"result": "failed: the note was missing or not JSON"}
    for _ in range(TRIES):
        sync(wt)
        q = load(wt)
        i = next((x for x in q["items"] if x["id"] == item_id), None)
        if i is None:
            raise RuntimeError(f"no item {item_id}")
        lost = i.get("claimed_by") != worker
        i.setdefault("runs", []).append(dict(note, worker=worker, at=iso(now()), lost_lease=lost))
        i["runs"][-1]["outcome"] = outcome
        status = i["status"]
        if not lost:
            try:
                status = T.next_status(i["status"], outcome, failures(i))
            except T.NotAllowed as e:   # someone moved the item while this run held it: record, do not overwrite
                i["runs"][-1]["refused"] = str(e)
                status = i["status"]
            if outcome in ("failed", "lease_expired") and status == "escalated" and "refused" not in i["runs"][-1]:
                i["runs"][-1]["escalated"] = f"failed {failures(i)} times in a row"
            i["status"] = status
            for k in ("claimed_by", "claimed_at", "lease_until"):
                i.pop(k, None)
        save(wt, q)
        if commit_push(wt, f"worker: {worker} {'releases' if not lost else 'records (lease lost)'} {item_id} -> {status}"):
            print(f"released-status={status if not lost else 'lease lost'}")   # the chain reads the real outcome (review of Q60, 8)
            return 0
    return 1


def held(wt, worker, item_id):
    """0 if this worker still holds a live claim on the item: checked right before a merge or PR (review of Q60, 5)."""
    sync(wt)
    i = next((x for x in load(wt)["items"] if x["id"] == item_id), None)
    return 0 if i and i.get("claimed_by") == worker and live_claim(i, now()) else 1


TIER_OUTCOMES = {"confirm", "redo", "escalate", "solved", "accept", "reject", "decide_done", "decide_open", "park", "decide_blocked",
                 "decide_scheduled", "decide_owner", "dispatch", "verdict", "invalid"}
EXPERIMENT_FAILED = "lost"   # only the experiment record job moves scheduled -> open with it (move(..., allow_failed=True))


def move(wt, actor, item_id, outcome, reason, allow_failed=False):
    """The tier above the workers (the Claude session, the owner's chat) moves an item: the same table, the same record.
    Chat edits used to set status by hand, past transitions.py (Q60 open point)."""
    if outcome not in TIER_OUTCOMES and not (allow_failed and outcome == EXPERIMENT_FAILED):
        raise SystemExit(f"not a tier or owner outcome: {outcome!r} (worker outcomes go through release)")
    for _ in range(TRIES):
        sync(wt)
        q = load(wt)
        i = next((x for x in q["items"] if x["id"] == item_id), None)
        if i is None:
            raise SystemExit(f"no item {item_id}")
        to = T.next_status(i["status"], outcome, failures(i) + 1 if outcome == EXPERIMENT_FAILED else 0)   # raises NotAllowed for a move the table does not have
        i.setdefault("runs", []).append({"at": iso(now()), "worker": actor, "result": reason, "outcome": outcome})
        i["status"] = to
        save(wt, q)
        if commit_push(wt, f"{actor}: {item_id} {outcome} -> {to}"):
            print(f"{item_id}: {to}")
            return 0
    return 1


def main(a):
    if a[0] == "claim":
        return claim(a[1], a[2], a[3], a[4].split(",") if len(a) > 4 and a[4] else None)
    if a[0] == "release":
        return release(a[1], a[2], a[3], a[4], a[5])
    if a[0] == "held":
        return held(a[1], a[2], a[3])
    if a[0] == "move":
        return move(a[1], a[2], a[3], a[4], a[5])
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
