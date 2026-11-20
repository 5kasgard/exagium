"""Budget ledger for the free workers (queue P8; logs usage per call for the pools it uses, part of Q22).

python3 worker/ledger.py reserve WT RUN_ID POOL:CALLS[:NEURONS_EACH][,...]   # exit 2 if a cap would be passed
python3 worker/ledger.py record WT RUN_ID CALLS_JSONL                     # actual usage of each call, appended
python3 worker/ledger.py check WT POOL:CALLS[,...]                        # no write: exit 2 if the reservation would fail

WT is a worktree of the "worker-runs" branch. ledger/YYYY-MM-DD.json holds, per pool, calls reserved and calls,
tokens and neurons used. A run checks headroom before it claims, then reserves calls before it calls (a commit; git refuses a
second concurrent push, so two runs can't both take the last share). Calls made are counted exactly, and a run's reserved calls are held as "pending_calls" until it records, then released; neurons are a projection
(per job type) held as "pending" until the run records its actual use, then replaced by it; the cap applies to actual plus
pending, so one run can overshoot by the difference between its projection and its use. Caps are half of each pool's measured daily allowance, so the other half stays for
chats, reviews and probes (P8, "coexists with chat sessions"):
  cloudflare 10,000 neurons/day (measured: 560-780 per long review, router-runs reviews/44253058538, 44257135013)
  -> cap 5,000 neurons; its calls are not limited by Cloudflare, so the call cap is only a backstop (500);
  nvidia (NIM): no daily limit is published and the credit system ended (NVIDIA staff, Sep 2025); the stated limit is
  about 40 requests a minute per key (forum, Jun 2026); bench-5 measured a bucket of about 220 that refills at about 8 a
  minute (project doc forge-free-pools). The old cap of 40 a day read a per-minute figure as daily and stopped both
  the worker and P5 (2026-11-19, 36 of 40 used by midday). Cap 400 a day: far under the measured refill (about
  11,000 a day), kept low because the terms are for development and evaluation; groq 1,000 requests/day -> cap 500.
  Paid pools are not listed, so the worker can't reserve them.
"""
import datetime as dt, json, os, subprocess, sys

BRANCH = "worker-runs"
CAPS = {"cloudflare": {"calls": 500, "neurons": 5000}, "nvidia": {"calls": 400}, "groq": {"calls": 500},
        "openrouter": {"calls": 500},
        "openrouter_paid": {"calls": 100, "usd": 0.25}}   # owner 2026-11-20: paid fallbacks allowed, cost-effective, under a
# daily dollar cap; the spec's third field is the estimate per call in thousandths of a dollar (openrouter_paid:2:5 = 2 calls
# at $0.005); actual cost comes from OpenRouter's usage.cost, or the estimate when a reply carries none   # free models: the account has 1,000 a day (GET /api/v1/key, run or-key 2026-11-20:
# is_free_tier false, free_model_daily_requests limit 1000; $10 credit, $0 spent). The old cap of 30 assumed 50 a day.
# Half, as for the other pools, leaves the rest for the owner's chats and probes
ROLLING_NEURONS = 9000   # Cloudflare enforces its 10,000 on a rolling 24 h window, not at 00:00 UTC as its docs say:
# refused at 06:09Z on 2026-11-20 with 2,405 neurons used that UTC day (10,528 in the 24 h before), served again at
# 07:26Z (about 9,725 in the 24 h before) and at 09:16Z (account analytics, aiInferenceAdaptiveGroups); 1,000 left for
# calls the ledger does not see
EST_NEURONS = 1000   # default per Cloudflare call when the caller gives none (long reviews measured 560-780)


def parse(spec):
    """POOL:CALLS[:NEURONS_EACH],... -> {pool: (calls, neurons each)}"""
    out = {}
    for x in spec.split(","):
        f = x.split(":")
        out[f[0]] = (int(f[1]), int(f[2]) if len(f) > 2 else EST_NEURONS)
    return out
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


def rolling_neurons(wt, now=None):
    """Cloudflare neurons recorded in ledger/calls.jsonl in the 24 h before now."""
    now = now or dt.datetime.now(dt.timezone.utc)
    since, total = (now - dt.timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ"), 0.0
    try:
        with open(os.path.join(wt, "ledger", "calls.jsonl")) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("engine") == "cloudflare" and str(r.get("t", "")) >= since:
                    total += float((r.get("usage") or {}).get("neurons") or 0)
    except FileNotFoundError:
        pass
    return total


def headroom(led, want, rolling=0.0):
    for pool, (n, est) in want.items():
        if pool not in CAPS:
            return f"pool {pool} is not a free pool the worker may use"
        e, cap = led.get(pool, {}), CAPS[pool]
        pending = sum((e.get("pending") or {}).values())   # projections of runs not yet recorded
        # calls made plus calls reserved by runs not yet recorded: a recorded run's unused reservation (a fallback pool
        # it never called) is released, as neurons are (2026-11-19: 20 Cloudflare calls reserved, 1 made, and the cap
        # stopped the worker). A ledger written before pending_calls existed counts only its recorded calls.
        used = e.get("calls", 0) + sum((e.get("pending_calls") or {}).values())
        if used + n > cap["calls"] or ("neurons" in cap and e.get("neurons", 0) + pending + n * est > cap["neurons"]):
            return f"{pool}: cap reached ({used} calls made or held, {e.get('neurons', 0):.0f} neurons used; cap {cap})"
        if "usd" in cap and e.get("usd", 0) + sum((e.get("pending_usd") or {}).values()) + n * est / 1000 > cap["usd"]:
            return f"{pool}: daily dollar cap reached (${e.get('usd', 0):.4f} spent; cap ${cap['usd']})"
        if pool == "cloudflare" and rolling + sum((e.get("pending") or {}).values()) + n * est > ROLLING_NEURONS:
            return f"cloudflare: {rolling:.0f} neurons in the last 24 h; the account's rolling allowance would be passed"
    return None


def check(wt, spec):
    sync(wt)
    why = headroom(load(day_path(wt)), parse(spec), rolling_neurons(wt))
    print(why or "headroom ok")
    return 2 if why else 0


def reserve(wt, run, spec):
    want = parse(spec)
    for _ in range(TRIES):
        sync(wt)
        p = day_path(wt)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        led = load(p)
        why = headroom(led, want, rolling_neurons(wt))
        if why:
            print(why)
            return 2
        for pool, (n, est) in want.items():
            if pool not in CAPS:
                print(f"pool {pool} is not a free pool the worker may use", file=sys.stderr)
                return 2
            e = led.setdefault(pool, {"reserved": 0, "calls": 0, "tokens": 0, "neurons": 0.0, "runs": []})
            cap = CAPS[pool]
            e.setdefault("neurons_reserved", 0)
            e["reserved"] += n   # cumulative, for the record; the cap uses calls made plus pending_calls
            e.setdefault("pending_calls", {})[run] = e.get("pending_calls", {}).get(run, 0) + n
            if "neurons" in cap:
                e["neurons_reserved"] += n * est
                e.setdefault("pending", {})[run] = n * est   # replaced by the actual use when the run is recorded
            if "usd" in cap:
                e.setdefault("pending_usd", {})[run] = n * est / 1000
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
            if c["engine"] in CAPS and "usd" in CAPS[c["engine"]]:
                est = (e.get("pending_usd") or {}).get(run, 0) / max(1, (e.get("pending_calls") or {}).get(run, 1))
                e["usd"] = round(e.get("usd", 0) + (float(u["cost"]) if isinstance(u.get("cost"), (int, float)) else est), 6)
            (e.get("pending") or {}).pop(run, None)   # the projection gives way to the actual (release test, R2)
        for e in led.values():   # a reserved pool the run never called (a fallback not needed) releases its projection
            if isinstance(e, dict):
                (e.get("pending") or {}).pop(run, None)
                (e.get("pending_calls") or {}).pop(run, None)   # its reserved calls likewise give way to the calls made
                (e.get("pending_usd") or {}).pop(run, None)
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
