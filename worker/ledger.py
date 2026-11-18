"""Budget ledger for the free workers (queue P8; logs usage per call for the pools it uses, part of Q22).

python3 worker/ledger.py reserve WT RUN_ID POOL:CALLS[,POOL:CALLS...]   # exit 2 if a cap would be passed
python3 worker/ledger.py record WT RUN_ID CALLS_JSONL                     # actual usage of each call, appended
python3 worker/ledger.py check WT POOL:CALLS[,...]                        # no write: exit 2 if the reservation would fail

WT is a worktree of the "worker-runs" branch. ledger/YYYY-MM-DD.json holds, per pool, calls reserved and calls,
tokens and neurons used. A run checks headroom before it claims, then reserves calls before it calls (a commit; git refuses a
second concurrent push, so two runs can't both take the last share). Calls are counted exactly; neurons are a projection
(EST_NEURONS per Cloudflare call) checked against recorded actuals, so one run can overshoot by the difference. Caps are half of each pool's measured daily allowance, so the other half stays for
chats, reviews and probes (P8, "coexists with chat sessions"):
  cloudflare 10,000 neurons/day (measured: 560-780 per long review, router-runs reviews/44253058538, 44257135013)
  -> cap 5,000 neurons and 20 calls; nvidia (NIM) shape unmeasured (bench-5: about 30/min, then refusals) -> cap
  40 calls; groq 1,000 requests/day -> cap 500. Paid pools are not listed, so the worker can't reserve them.
"""
import datetime as dt, json, os, subprocess, sys

BRANCH = "worker-runs"
CAPS = {"cloudflare": {"calls": 20, "neurons": 5000}, "nvidia": {"calls": 40}, "groq": {"calls": 500}}
EST_NEURONS = 1000   # per Cloudflare call, reserved ahead (measured 560-780 for long reviews); actuals recorded after
TRIES = 5


def git(wt, *a, check=True):
    return subprocess.run(["git", "-C", wt, *a], capture_output=True, text=True, check=check)


def sync(wt):
    if git(wt, "fetch", "-q", "--depth", "5", "origin", BRANCH, check=False).returncode == 0:
        git(wt, "reset", "-q", "--hard", "FETCH_HEAD")


def day_path(wt):
    return os.path.join(wt, "ledger", dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d") + ".json")


def push(wt, paths, msg):
    git(wt, "add", *paths)
    git(wt, "-c", "user.name=exagium-worker", "-c", "user.email=worker@exagium.invalid", "commit", "-qm", msg)
    return git(wt, "push", "-q", "origin", f"HEAD:{BRANCH}", check=False).returncode == 0


def load(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def headroom(led, want):
    for pool, n in want.items():
        if pool not in CAPS:
            return f"pool {pool} is not a free pool the worker may use"
        e, cap = led.get(pool, {}), CAPS[pool]
        if e.get("reserved", 0) + n > cap["calls"] or ("neurons" in cap and max(e.get("neurons", 0), e.get("neurons_reserved", 0)) + n * EST_NEURONS > cap["neurons"]):
            return f"{pool}: cap reached ({e.get('reserved', 0)} calls reserved, {e.get('neurons', 0):.0f} neurons used; cap {cap})"
    return None


def check(wt, spec):
    sync(wt)
    why = headroom(load(day_path(wt)), {k: int(v) for k, v in (x.split(":") for x in spec.split(","))})
    print(why or "headroom ok")
    return 2 if why else 0


def reserve(wt, run, spec):
    want = {k: int(v) for k, v in (x.split(":") for x in spec.split(","))}
    for _ in range(TRIES):
        sync(wt)
        p = day_path(wt)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        led = load(p)
        for pool, n in want.items():
            if pool not in CAPS:
                print(f"pool {pool} is not a free pool the worker may use", file=sys.stderr)
                return 2
            e = led.setdefault(pool, {"reserved": 0, "calls": 0, "tokens": 0, "neurons": 0.0, "runs": []})
            cap = CAPS[pool]
            e.setdefault("neurons_reserved", 0)
            if e["reserved"] + n > cap["calls"] or ("neurons" in cap and max(e["neurons"], e["neurons_reserved"]) + n * EST_NEURONS > cap["neurons"]):
                print(f"{pool}: cap reached ({e['reserved']} reserved, {e['neurons']:.0f} neurons; cap {cap})")
                return 2
            e["reserved"] += n
            if "neurons" in cap:
                e["neurons_reserved"] += n * EST_NEURONS
            e["runs"].append(run)
        json.dump(led, open(p, "w"), indent=1, sort_keys=True)
        if push(wt, [p], f"ledger: {run} reserves {spec}"):
            return 0
    return 1


def record(wt, run, calls_path):
    calls = [json.loads(line) for line in open(calls_path) if line.strip()]
    for _ in range(TRIES):
        sync(wt)
        p = day_path(wt)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        led = load(p)
        log = os.path.join(wt, "ledger", "calls.jsonl")
        for c in calls:
            e = led.setdefault(c["engine"], {"reserved": 0, "calls": 0, "tokens": 0, "neurons": 0.0, "runs": []})
            u = c.get("usage") or {}
            e["calls"] += 1
            e["tokens"] += int(u.get("total_tokens") or 0)
            e["neurons"] += float(u.get("neurons") or 0)
        json.dump(led, open(p, "w"), indent=1, sort_keys=True)
        with open(log, "a") as f:
            for c in calls:
                f.write(json.dumps({k: c.get(k) for k in ("t", "engine", "model", "ok", "status", "latency_s", "usage")}
                                   | {"run": run}) + "\n")
        if push(wt, [p, log], f"ledger: {run} used {len(calls)} call(s)"):
            return 0
    return 1


if __name__ == "__main__":
    a = sys.argv[1:]
    try:
        sys.exit(reserve(a[1], a[2], a[3]) if a[0] == "reserve" else check(a[1], a[2]) if a[0] == "check" else record(a[1], a[2], a[3]))
    except subprocess.CalledProcessError as e:
        print("FAILURE", e, e.stderr, file=sys.stderr)
        sys.exit(3)
