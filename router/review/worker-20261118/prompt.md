Adversarial review of an automation where free models work a task queue without a human in the loop. No praise. For each problem: file and function, a concrete failure or attack (model replies are untrusted and may be prompt-injected via repo docs, paper abstracts or queue text), whether it succeeds against THIS code (trace it), and a fix. Grade critical / major / minor; say "not traceable" when you can't trace it. End with: unpause / fix then unpause / do not unpause.
Focus: (1) model text reaching a shell or being executed; (2) changes to main beyond the declared Markdown targets; (3) two workers holding one item, or items stuck in progress; (4) budget caps bypassed; (5) the measures in digest.py being wrong (double claims missed, wrong counts).

=== worker/README.md
# worker — spec card (queue P8, first version; P7 and P9 wired in)

**What it is.** Free models working the queue without a chat. A scheduled job claims the top ready item, a free model
does it, another family checks it, a script checks both, and the result lands on main or as a PR. A daily digest
writes the handoff. Claude designs and spot-checks; it is not in the loop.

**How it decides (hypotheses, measured by the digest):**
- **Claims:** git is the lock. A claim is a commit to the queue branch, and git refuses the second of two concurrent
  pushes. Leases last 2 h. A chat's claim always wins. An item whose files a live claim touches is skipped.
- **Job types**, by what an item carries:
  - `targets` (exact file and line ranges): a doc fix (`doc_job.py`);
  - `research` (a question and queries): a literature card (`research_job.py`, P7).
- **Who does what:**
  - search: API scripts (OpenAlex, arXiv, Semantic Scholar);
  - drafting and first reads: gpt-oss-120b on Cloudflare;
  - the check: Nemotron 3 Ultra on NIM, another family;
  - verification by script: edits only on target lines with exact old text, Markdown only; quotes verbatim in the
    source abstract, numbers in the quote.
- **Lands on main** only if every script check passes and the second model approves; else a PR, with the item waiting
  on the owner.
- **Budget:** `ledger.py` reserves calls before any call, by commit, against caps of half each free pool's daily
  allowance. Paid pools can't be reserved.
- **Jobs:** the key and the write token are never in one job: claim (write, no keys), draft (keys, read-only, runs
  nothing a model writes), land (write, no keys).
- **Pause:** a file `worker/PAUSE` on main stops every run before it claims.

**Measured** (`digest.py`, P9's first measures): claims, double claims (target 0), releases, expired leases, results
by kind, budget used per pool. Spot-check acceptance is recorded per item by Claude or the owner.

**Files.** `queue_ops.py` (claims), `ledger.py` (budget), `doc_job.py`, `research_job.py`, `digest.py`,
`test_worker.py`; workflows `worker.yml` (hourly) and `digest.yml` (daily, 06:23 Brisbane). Records: the queue's
items (`runs`), and the `worker-runs` branch (`ledger/`, `digest/`).

**Not yet:** code items (they need P5's jailed proposer and Q56), the P10 auditor, process proposals (P9's weekly
free-model step).
=== worker/queue_ops.py
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
=== worker/ledger.py
"""Budget ledger for the free workers (queue P8; logs usage per call for the pools it uses, part of Q22).

python3 worker/ledger.py reserve WT RUN_ID POOL:CALLS[,POOL:CALLS...]   # exit 2 if a cap would be passed
python3 worker/ledger.py record WT RUN_ID CALLS_JSONL                     # actual usage of each call, appended

WT is a worktree of the "worker-runs" branch. ledger/YYYY-MM-DD.json holds, per pool, calls reserved and calls,
tokens and neurons used. A run reserves before it calls (a commit; git refuses a second concurrent push, so two runs
can't both take the last share). Caps are half of each pool's measured daily allowance, so the other half stays for
chats, reviews and probes (P8, "coexists with chat sessions"):
  cloudflare 10,000 neurons/day (measured: 560-780 per long review, router-runs reviews/44253058538, 44257135013)
  -> cap 5,000 neurons and 20 calls; nvidia (NIM) shape unmeasured (bench-5: about 30/min, then refusals) -> cap
  40 calls; groq 1,000 requests/day -> cap 500. Paid pools are not listed, so the worker can't reserve them.
"""
import datetime as dt, json, os, subprocess, sys

BRANCH = "worker-runs"
CAPS = {"cloudflare": {"calls": 20, "neurons": 5000}, "nvidia": {"calls": 40}, "groq": {"calls": 500}}
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
            if e["reserved"] + n > cap["calls"] or ("neurons" in cap and e["neurons"] >= cap["neurons"]):
                print(f"{pool}: cap reached ({e['reserved']} reserved, {e['neurons']:.0f} neurons; cap {cap})")
                return 2
            e["reserved"] += n
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
        sys.exit(reserve(a[1], a[2], a[3]) if a[0] == "reserve" else record(a[1], a[2], a[3]))
    except subprocess.CalledProcessError as e:
        print("FAILURE", e, e.stderr, file=sys.stderr)
        sys.exit(3)
=== worker/doc_job.py
"""Doc job for the queue worker (queue P8, first job type): fix a doc item whose exact targets are given.

python3 worker/doc_job.py context ITEM_JSON REPO OUT_DIR     # no keys: the prompt bundle (item + numbered target lines)
python3 worker/doc_job.py draft OUT_DIR                      # keys: drafter proposes edits, a reviewer of another family judges
python3 worker/doc_job.py apply REPO OUT_DIR                  # no keys: validate, apply, decide merge or PR

Nothing a model writes is executed: its reply is parsed as JSON and applied as text edits by this script.
An edit is accepted only if its file is one of the item's targets, its line is inside a target range, its "old" text
equals that line exactly, and the file is Markdown. The result is "merge" (self-merge, as P8's spec allows for doc
items) only if every edit is accepted, there is at least one, and the reviewer approves; else "pr" (a human or Claude
decides) with the reasons. Drafter: gpt-oss-120b (Cloudflare). Reviewer: Nemotron 3 Ultra (NVIDIA NIM), another family.
"""
import difflib, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DRAFTER = ("cloudflare", "@cf/openai/gpt-oss-120b")
REVIEWER = ("nvidia", "nvidia/nemotron-3-ultra-550b-a55b")
MARGIN = 3   # context lines shown around each target range

DRAFT_PROMPT = """You fix one documentation item in a repository. Change only the target lines shown, and only as the item asks.
Keep the file's style. Change as little as possible. Do not invent facts or numbers: use only what the item and the lines say.

Reply with JSON only, no prose, in this form:
{{"edits": [{{"path": "<file>", "line": <line number>, "old": "<the line exactly as shown>", "new": "<the replacement line>"}}],
 "note": "<one sentence: what you changed and why>"}}
Each edit replaces one whole line. "old" must equal the shown line exactly (without the number prefix).
If the item cannot be done within these lines, reply {{"edits": [], "note": "<why>"}}.

ITEM {id}: {title}
WHAT TO DO: {next}

TARGET LINES (number | text):
{excerpts}
"""

REVIEW_PROMPT = """You review a proposed documentation fix. Approve only if the change does what the item asks, changes nothing else,
adds no fact or number that the item or the original lines do not support, and keeps the meaning correct.

Reply with JSON only: {{"verdict": "approve" or "reject", "reason": "<one or two sentences>"}}

ITEM {id}: {title}
WHAT TO DO: {next}

PROPOSED CHANGE (unified diff):
{diff}
"""


def excerpts(repo, item):
    out = []
    for t in item["targets"]:
        lines = open(os.path.join(repo, t["path"]), encoding="utf-8").read().split("\n")
        a, b = t["lines"]
        lo, hi = max(1, a - MARGIN), min(len(lines), b + MARGIN)
        out.append(f"--- {t['path']} (targets: lines {a}-{b}; context {lo}-{hi}) ---")
        out += [f"{n} | {lines[n - 1]}" for n in range(lo, hi + 1)]
    return "\n".join(out)


def context(item_path, repo, out):
    item = json.load(open(item_path))["item"]
    os.makedirs(out, exist_ok=True)
    files = {t["path"]: open(os.path.join(repo, t["path"]), encoding="utf-8").read() for t in item["targets"]}
    json.dump({"item": item, "files": files,
               "prompt": DRAFT_PROMPT.format(id=item["id"], title=item["title"], next=item["next"],
                                              excerpts=excerpts(repo, item))},
              open(os.path.join(out, "bundle.json"), "w"), indent=1)
    return 0


def parse_json(text):
    """The first JSON object in a model's reply (models sometimes wrap it in a code fence)."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def check_edits(item, files, edits):
    """(accepted edits, reasons for any rejection). Pure: no model output is executed."""
    reasons, ok = [], []
    targets = {}
    for t in item["targets"]:
        targets.setdefault(t["path"], []).append(tuple(t["lines"]))
    seen = set()
    if not isinstance(edits, list):
        return [], ["edits is not a list"]
    for e in edits:
        if not isinstance(e, dict) or not {"path", "line", "old", "new"} <= set(e):
            reasons.append(f"malformed edit {str(e)[:120]}"); continue
        p, n = e["path"], e["line"]
        if p not in targets or not p.endswith(".md"):
            reasons.append(f"{p}: not a Markdown target"); continue
        if not isinstance(n, int) or not any(a <= n <= b for a, b in targets[p]):
            reasons.append(f"{p}:{n}: outside the target lines"); continue
        lines = files[p].split("\n")
        if n > len(lines) or lines[n - 1] != e["old"]:
            reasons.append(f"{p}:{n}: 'old' does not match the line"); continue
        if not isinstance(e["new"], str) or "\n" in e["new"] or e["new"] == e["old"]:
            reasons.append(f"{p}:{n}: 'new' is not one changed line"); continue
        if (p, n) in seen:
            reasons.append(f"{p}:{n}: edited twice"); continue
        seen.add((p, n))
        ok.append(e)
    return ok, reasons


def apply_edits(files, edits):
    new = {p: s.split("\n") for p, s in files.items()}
    for e in edits:
        new[e["path"]][e["line"] - 1] = e["new"]
    return {p: "\n".join(v) for p, v in new.items()}


def diff(files, new):
    return "".join("".join(difflib.unified_diff(files[p].splitlines(True), new[p].splitlines(True), "a/" + p, "b/" + p))
                   for p in sorted(files) if files[p] != new[p])


def draft(out):
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "router"))
    import engines
    b = json.load(open(os.path.join(out, "bundle.json")))
    item, files, calls = b["item"], b["files"], []

    def ask(engine, model, prompt):
        r = (engines.call(engine, model, prompt, max_tokens=4000) if engine in ("cloudflare", "openrouter", "gemini")
             else engines.compat_call(engine, model, prompt, max_tokens=4000))
        calls.append(r)
        return r

    d = ask(*DRAFTER, b["prompt"])
    proposal = parse_json(d["text"]) or {"edits": [], "note": "drafter reply was not JSON"}
    edits, reasons = check_edits(item, files, proposal.get("edits"))
    review = None
    if edits:
        dd = diff(files, apply_edits(files, edits))
        r = ask(*REVIEWER, REVIEW_PROMPT.format(id=item["id"], title=item["title"], next=item["next"], diff=dd))
        review = parse_json(r["text"]) or {"verdict": "none", "reason": "reviewer reply was not JSON: " + str(r["error"])[:200]}
    json.dump({"proposal": proposal, "review": review, "pre_check": reasons}, open(os.path.join(out, "draft.json"), "w"), indent=1)
    with open(os.path.join(out, "calls.jsonl"), "w") as f:
        for c in calls:
            f.write(json.dumps({k: c.get(k) for k in ("t", "engine", "model", "ok", "status", "latency_s", "usage", "error")}) + "\n")
    print(f"drafter ok={d['ok']} edits={len(edits)} rejected={len(reasons)} review={review and review.get('verdict')}")
    return 0


def apply(repo, out):
    b = json.load(open(os.path.join(out, "bundle.json")))
    item = b["item"]
    files = {p: open(os.path.join(repo, p), encoding="utf-8").read() for p in b["files"]}
    if files != b["files"]:
        res = {"result": "pr", "why": ["a target file changed since the bundle was made"], "edits": 0}
    else:
        d = json.load(open(os.path.join(out, "draft.json")))
        edits, reasons = check_edits(item, files, d["proposal"].get("edits"))   # checked again here, with no keys
        verdict = (d.get("review") or {}).get("verdict")
        why = reasons + ([] if edits else ["no accepted edit: " + str(d["proposal"].get("note", ""))[:300]]) \
            + ([] if verdict == "approve" else [f"reviewer: {verdict}: {(d.get('review') or {}).get('reason', '')}"[:400]])
        new = apply_edits(files, edits)
        for p, s in new.items():
            if s != files[p]:
                open(os.path.join(repo, p), "w", encoding="utf-8").write(s)
        open(os.path.join(out, "change.diff"), "w").write(diff(files, new))
        res = {"result": "merge" if not why else "pr", "why": why, "edits": len(edits),
               "note": d["proposal"].get("note"), "review": d.get("review")}
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)
    print(json.dumps(res)[:600])
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit({"context": lambda: context(a[1], a[2], a[3]), "draft": lambda: draft(a[1]),
              "apply": lambda: apply(a[1], a[2])}[a[0]]())
=== worker/research_job.py
"""Research job for the queue worker (queue P7 wired into P8): a literature card for one question.

python3 worker/research_job.py search ITEM_JSON OUT_DIR        # no keys: OpenAlex, arXiv, Semantic Scholar APIs (logged)
python3 worker/research_job.py read OUT_DIR                    # keys: a free model extracts claims with exact quotes;
                                                               # a second, of another family, checks each kept claim
python3 worker/research_job.py card REPO OUT_DIR               # no keys: verify quotes by script, write research/cards/<ID>.md

Routing by step, as P7 asks: search by script; first read by a free model; verify by script (every quote must occur
verbatim in the source's abstract, and every number in a claim must occur in its quote) plus a second free model;
Claude judges last (the card says so). Nothing a model writes is executed. An item opts in with
"research": {"question": "...", "queries": ["...", ...]} and "touches": ["research/cards/<ID>.md"].
"""
import json, os, re, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
READER = ("cloudflare", "@cf/openai/gpt-oss-120b")
CHECKER = ("nvidia", "nvidia/nemotron-3-ultra-550b-a55b")
PER_QUERY = 6
MAX_SOURCES = 18
UA = {"User-Agent": "exagium-research/1.0 (+https://github.com/5kasgard/exagium)"}

READ_PROMPT = """You read abstracts to answer a research question for a project that measures small AI models.
Extract only claims that bear on the question. For each, copy an exact quote from the abstract (verbatim, 8 to 40 words)
that states it; a claim without such a quote is not allowed. Prefer measured results, numbers to beat, known failure modes
and prior art. Do not add anything the abstracts don't say.

Reply with JSON only:
{{"claims": [{{"source": "<source id>", "claim": "<one sentence, in your words>", "quote": "<exact words from that abstract>",
  "kind": "result" | "frontier" | "failure mode" | "prior art" | "method"}}],
 "open_questions": ["<what these sources leave open about the question>"]}}

QUESTION: {question}

SOURCES:
{sources}
"""

CHECK_PROMPT = """Check each claim against its quote. Keep a claim only if the quote supports it, the claim adds nothing the quote
doesn't say, and it bears on the question. Reply with JSON only: {{"keep": [<indexes of claims to keep>], "notes": "<one sentence>"}}

QUESTION: {question}

CLAIMS:
{claims}
"""


def get(url, as_json=True):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    return json.loads(data) if as_json else data


def openalex(q):
    j = get("https://api.openalex.org/works?per-page=%d&search=%s" % (PER_QUERY, urllib.parse.quote(q)))
    out = []
    for w in j.get("results", []):
        inv = w.get("abstract_inverted_index") or {}
        pos = sorted((p, word) for word, ps in inv.items() for p in ps)
        out.append({"id": w.get("id", "").rsplit("/", 1)[-1], "title": w.get("title") or "", "year": w.get("publication_year"),
                    "url": w.get("doi") or w.get("id"), "cited_by": w.get("cited_by_count"),
                    "abstract": " ".join(word for _, word in pos), "via": "openalex"})
    return out


def arxiv(q):
    x = get("http://export.arxiv.org/api/query?max_results=%d&search_query=%s" % (PER_QUERY, urllib.parse.quote("all:" + q)), False)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in ET.fromstring(x).findall("a:entry", ns):
        aid = (e.findtext("a:id", "", ns) or "").rsplit("/", 1)[-1]
        out.append({"id": "arXiv:" + aid, "title": " ".join((e.findtext("a:title", "", ns) or "").split()),
                    "year": (e.findtext("a:published", "", ns) or "")[:4], "url": e.findtext("a:id", "", ns),
                    "abstract": " ".join((e.findtext("a:summary", "", ns) or "").split()), "via": "arxiv"})
    return out


def semantic_scholar(q):
    j = get("https://api.semanticscholar.org/graph/v1/paper/search?limit=%d&fields=title,abstract,year,url,citationCount&query=%s"
            % (PER_QUERY, urllib.parse.quote(q)))
    return [{"id": "S2:" + p["paperId"], "title": p.get("title") or "", "year": p.get("year"), "url": p.get("url"),
             "cited_by": p.get("citationCount"), "abstract": p.get("abstract") or "", "via": "semanticscholar"}
            for p in j.get("data", [])]


def search(item_path, out):
    item = json.load(open(item_path))["item"]
    r = item["research"]
    os.makedirs(out, exist_ok=True)
    log, seen, sources = [], set(), []
    for q in r["queries"]:
        for name, f in (("openalex", openalex), ("arxiv", arxiv), ("semanticscholar", semantic_scholar)):
            t0 = time.time()
            try:
                got = f(q)
                log.append({"api": name, "query": q, "n": len(got), "s": round(time.time() - t0, 1)})
            except Exception as e:  # an API down or rate-limited is logged, not fatal
                got = []
                log.append({"api": name, "query": q, "error": f"{type(e).__name__}: {e}"[:200]})
            for s in got:
                key = re.sub(r"\W+", "", s["title"].lower())[:80]
                if s["abstract"] and key and key not in seen:
                    seen.add(key)
                    sources.append(s)
    sources.sort(key=lambda s: -(s.get("cited_by") or 0))
    sources = sources[:MAX_SOURCES]
    for i, s in enumerate(sources):
        s["sid"] = f"S{i + 1}"
    json.dump({"item": item, "sources": sources, "log": log}, open(os.path.join(out, "search.json"), "w"), indent=1)
    print(f"{len(sources)} sources with abstracts; " + ", ".join(f"{x['api']}:{x.get('n', 'ERR')}" for x in log))
    return 0


def parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def verified(claims, sources):
    """Script check: the quote occurs verbatim (whitespace-normalised) in the source's abstract, and every number in
    the claim occurs in the quote. Returns (kept, dropped with reasons)."""
    by = {s["sid"]: s for s in sources}
    norm = lambda s: " ".join(str(s).split()).lower()
    kept, dropped = [], []
    for c in claims if isinstance(claims, list) else []:
        if not isinstance(c, dict) or not {"source", "claim", "quote"} <= set(c):
            dropped.append((c, "malformed")); continue
        s = by.get(c["source"])
        if not s:
            dropped.append((c, "unknown source")); continue
        if len(str(c["quote"]).split()) < 5 or norm(c["quote"]) not in norm(s["abstract"]):
            dropped.append((c, "quote not found verbatim in the abstract")); continue
        nums = re.findall(r"\d+(?:\.\d+)?", str(c["claim"]))
        if any(n not in str(c["quote"]) for n in nums):
            dropped.append((c, "a number in the claim is not in the quote")); continue
        kept.append(c)
    return kept, dropped


def read(out):
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "router"))
    import engines
    d = json.load(open(os.path.join(out, "search.json")))
    q = d["item"]["research"]["question"]
    calls = []

    def ask(engine, model, prompt):
        r = (engines.call(engine, model, prompt, max_tokens=6000) if engine == "cloudflare"
             else engines.compat_call(engine, model, prompt, max_tokens=4000))
        calls.append(r)
        return r

    src = "\n\n".join(f"[{s['sid']}] {s['title']} ({s['year']}). {s['abstract'][:1800]}" for s in d["sources"])
    r = ask(*READER, READ_PROMPT.format(question=q, sources=src))
    reading = parse_json(r["text"]) or {"claims": [], "open_questions": [], "error": "reader reply was not JSON"}
    kept, dropped = verified(reading.get("claims"), d["sources"])
    check = None
    if kept:
        lines = "\n".join(f"{i}. [{c['source']}] CLAIM: {c['claim']} QUOTE: \"{c['quote']}\"" for i, c in enumerate(kept))
        c2 = ask(*CHECKER, CHECK_PROMPT.format(question=q, claims=lines))
        check = parse_json(c2["text"]) or {"keep": None, "notes": "checker reply was not JSON: " + str(c2["error"])[:200]}
    json.dump({"reading": reading, "script_kept": kept, "script_dropped": [[c, why] for c, why in dropped], "check": check},
              open(os.path.join(out, "read.json"), "w"), indent=1)
    with open(os.path.join(out, "calls.jsonl"), "w") as f:
        for c in calls:
            f.write(json.dumps({k: c.get(k) for k in ("t", "engine", "model", "ok", "status", "latency_s", "usage", "error")}) + "\n")
    print(f"reader ok={r['ok']} claims={len(reading.get('claims') or [])} script-verified={len(kept)} "
          f"checker kept={None if not check else check.get('keep')}")
    return 0


def card(repo, out):
    d = json.load(open(os.path.join(out, "search.json")))
    rd = json.load(open(os.path.join(out, "read.json")))
    item, by = d["item"], {s["sid"]: s for s in d["sources"]}
    kept, _ = verified(rd["script_kept"], d["sources"])   # re-checked here, with no keys
    keep = (rd.get("check") or {}).get("keep")
    final = [c for i, c in enumerate(kept) if isinstance(keep, list) and i in keep]
    L = [f"# Literature card: {item['id']} — {item['research']['question']}", "",
         "Made by the queue worker (P7 in P8): search by script (OpenAlex, arXiv, Semantic Scholar), first read by "
         f"{READER[1]}, quotes checked verbatim against each abstract by script, claims checked by {CHECKER[1]}. "
         "**Not yet judged by Claude**: abstracts only, so read the source before relying on a claim.", "",
         "Search log: " + "; ".join(f"{x['api']} {x.get('n', 'error')}" for x in d["log"]), ""]
    for kind in ("frontier", "result", "failure mode", "prior art", "method"):
        cs = [c for c in final if c.get("kind") == kind]
        if cs:
            L.append(f"## {kind.capitalize()}")
            for c in cs:
                s = by[c["source"]]
                L.append(f"- {c['claim']} — \"{c['quote']}\" ({s['title']}, {s['year']}; {s['url']})")
            L.append("")
    other = [c for c in final if c.get("kind") not in ("frontier", "result", "failure mode", "prior art", "method")]
    if other:
        L += ["## Other"] + [f"- {c['claim']} — \"{c['quote']}\" ({by[c['source']]['url']})" for c in other] + [""]
    oq = (rd["reading"] or {}).get("open_questions") or []
    if oq:
        L += ["## Open questions (reader's, unverified)"] + [f"- {x}" for x in oq[:8]] + [""]
    L += [f"Dropped by the script check: {len(rd['script_dropped'])}; by the second model: {len(kept) - len(final)}.",
          f"Sources read: {len(d['sources'])}."]
    path = os.path.join(repo, "research", "cards", f"{item['id']}.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("\n".join(L) + "\n")
    res = {"result": "merge" if final else "pr", "why": [] if final else ["no claim survived both checks"],
           "claims": len(final), "path": f"research/cards/{item['id']}.md"}
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit({"search": lambda: search(a[1], a[2]), "read": lambda: read(a[1]),
              "card": lambda: card(a[1], a[2])}[a[0]]())
=== worker/digest.py
"""Daily digest: the automatic handoff (queue P8). Facts by script; a short summary by a free model, checked by script.

python3 worker/digest.py facts MAIN_WT QUEUE_WT RUNS_WT OUT_DIR [--hours 24]   # no keys
python3 worker/digest.py summary OUT_DIR                                         # key: one free-model call
python3 worker/digest.py write RUNS_WT OUT_DIR                                   # no keys: commits digest/DATE.md

The facts section is the record: queue items that changed status, worker runs with their results, commits on main,
budget used per pool, and the open list in the queue's own order. The summary is labelled as model-drafted; any
sentence in it with a number but no evidence pointer (a path, run ID, commit or item ID) is flagged, as P8 asks.
"""
import datetime as dt, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import queue_ops as Q  # noqa: E402

SUMMARY_MODEL = ("cloudflare", "@cf/openai/gpt-oss-120b")
POINTER = re.compile(r"(\b[QP]\d+\b|\brun \d{6,}|\b\d{8,}\b|\b[0-9a-f]{7,40}\b|[\w./-]+/[\w.-]+|\w+\.(md|py|json|c|yml)\b)")


def git(wt, *a):
    return subprocess.run(["git", "-C", wt, *a], capture_output=True, text=True).stdout


def queue_at(wt, since):
    """The queue as it was at the last commit before `since` (None if none is in the fetched history)."""
    sha = git(wt, "rev-list", "-1", f"--before={since}", "HEAD").strip()
    if not sha:
        return None
    s = git(wt, "show", f"{sha}:{Q.QUEUE}")
    return json.loads(s) if s else None


def facts(main_wt, queue_wt, runs_wt, out, hours=24):
    t = Q.now()
    since = Q.iso(t - dt.timedelta(hours=hours))
    q = Q.load(queue_wt)
    old = queue_at(queue_wt, since)
    old_status = {i["id"]: i.get("status") for i in (old or {}).get("items", [])}
    changed = [(i["id"], old_status.get(i["id"], "new"), i.get("status"), i["title"][:110]) for i in q["items"]
               if old_status.get(i["id"], "new") != i.get("status")]
    runs = [(i["id"], r) for i in q["items"] for r in i.get("runs", []) if r.get("at", "") >= since]
    commits = git(main_wt, "log", f"--since={since}", "--format=%h %s").strip().splitlines()
    ledgers = {}
    for d in sorted({(t - dt.timedelta(hours=h)).strftime("%Y-%m-%d") for h in (0, hours)}):
        p = os.path.join(runs_wt, "ledger", d + ".json")
        if os.path.exists(p):
            ledgers[d] = {k: {x: v[x] for x in ("reserved", "calls", "tokens", "neurons")} for k, v in json.load(open(p)).items()}
    by = {i["id"]: i for i in q["items"]}
    open_list = [i for i in q["items"] if i.get("status") in ("open", "in progress")]
    lines = [f"# Digest {t.strftime('%Y-%m-%d')} (facts by script, {hours} h to {Q.iso(t)})", "",
             f"## Queue: status changes ({'baseline missing: history too shallow' if old is None else len(changed)})"]
    lines += [f"- {a}: {b} -> {c}. {d}" for a, b, c, d in changed] or ["- none"]
    lines += ["", f"## Worker runs ({len(runs)})"]
    lines += [f"- {a}: {r.get('result')} by {r.get('worker')} at {r.get('at')}; {r.get('where', '')} {('(lease lost)' if r.get('lost_lease') else '')}".rstrip()
              for a, r in runs] or ["- none"]
    lines += ["", f"## Commits on main ({len(commits)})"] + ([f"- {c}" for c in commits] or ["- none"])
    lines += ["", "## Budget used (ledger on worker-runs)"]
    lines += [f"- {d} {k}: {v}" for d, l in ledgers.items() for k, v in l.items()] or ["- no worker spend recorded"]
    lines += ["", "## Open list, in the queue's order (first 12)"]
    for i in open_list[:12]:
        waits = [b for b in i.get("blocked_by", []) if by.get(b, {}).get("status") != "done"]
        lines.append(f"- {i['id']} [{i.get('status')}{', claimed by ' + i['claimed_by'] if i.get('claimed_by') else ''}"
                     f"{', waits on ' + ', '.join(waits) if waits else ''}{', worker-ready' if i.get('targets') and not waits else ''}]: {i['title'][:120]}")
    lines += ["", "## Worker measures (queue P9: measured by script every day)"] + metrics(queue_wt, since, q)
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "facts.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))
    return 0


def metrics(queue_wt, since, q):
    """From the queue branch's own history: claims, double claims (an item claimed while another worker holds it),
    releases, expired leases; and from the items' run notes: results by kind and job type."""
    subj = git(queue_wt, "log", "--reverse", f"--since={since}", "--format=%s").splitlines()
    holder, claims, double, releases, expired = {}, 0, [], 0, 0
    for s_ in subj:
        m = re.match(r"worker: (worker-\d+) claims (\S+)", s_)
        if m:
            claims += 1
            if holder.get(m.group(2)) not in (None, m.group(1)):
                double.append(f"{m.group(2)} by {m.group(1)} while {holder[m.group(2)]} held it")
            holder[m.group(2)] = m.group(1)
            continue
        m = re.match(r"worker: (worker-\d+) (releases|records \(lease lost\)) (\S+)", s_)
        if m:
            releases += 1
            if holder.get(m.group(3)) == m.group(1):
                holder.pop(m.group(3))
            continue
        if s_.startswith("worker: leases expired"):
            expired += 1
    results = {}
    for i in q["items"]:
        for r in i.get("runs", []):
            if r.get("at", "") >= since:
                results[str(r.get("result"))] = results.get(str(r.get("result")), 0) + 1
    return [f"- claims {claims}, releases {releases}, double claims {len(double)}{': ' + '; '.join(double) if double else ''}, "
            f"lease-expiry commits {expired}",
            f"- results: {', '.join(f'{k} {v}' for k, v in sorted(results.items())) or 'none'}",
            "- spot-check acceptance (Claude or the owner): recorded per item when checked; not measured by script"]


def flag(summary):
    out = []
    for s in re.split(r"(?<=[.!?])\s+", summary or ""):
        if re.search(r"\d", s) and not POINTER.search(s):
            out.append(s.strip())
    return out


def summary(out):
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "router"))
    import engines
    f = open(os.path.join(out, "facts.md")).read()
    prompt = ("Write a short handoff (at most 8 sentences) from these facts only: what changed, what is blocked or stuck, "
              "and what the next three items are. Cite the item ID, run ID, commit or file for every number. Add nothing "
              "the facts don't say.\n\n" + f)
    r = engines.call(*SUMMARY_MODEL, prompt, max_tokens=3000)
    json.dump({k: r.get(k) for k in ("t", "engine", "model", "ok", "status", "latency_s", "usage", "error", "text")},
              open(os.path.join(out, "summary.json"), "w"), indent=1)
    with open(os.path.join(out, "calls.jsonl"), "w") as fh:
        fh.write(json.dumps({k: r.get(k) for k in ("t", "engine", "model", "ok", "status", "latency_s", "usage")}) + "\n")
    print("summary ok", r["ok"], r["status"])
    return 0


def write(runs_wt, out):
    f = open(os.path.join(out, "facts.md")).read()
    s = {}
    if os.path.exists(os.path.join(out, "summary.json")):
        s = json.load(open(os.path.join(out, "summary.json")))
    text = (s.get("text") or "").strip()
    flagged = flag(text)
    body = f + "\n## Summary (drafted by " + (s.get("model") or "no model") + "; not checked by a person)\n\n" + \
        (text or "_no summary: " + str(s.get("error") or "the summary job did not run") + "_") + "\n"
    if flagged:
        body += "\n**Flagged: a number with no evidence pointer:**\n" + "\n".join(f"- {x}" for x in flagged) + "\n"
    day = Q.now().strftime("%Y-%m-%d")
    for _ in range(Q.TRIES):
        subprocess.run(["git", "-C", runs_wt, "fetch", "-q", "--depth", "5", "origin", "worker-runs"], capture_output=True)
        subprocess.run(["git", "-C", runs_wt, "reset", "-q", "--hard", "FETCH_HEAD"], capture_output=True)
        p = os.path.join(runs_wt, "digest", day + ".md")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(body)
        subprocess.run(["git", "-C", runs_wt, "add", p], check=True)
        subprocess.run(["git", "-C", runs_wt, "-c", "user.name=exagium-worker", "-c", "user.email=worker@exagium.invalid",
                        "commit", "-qm", f"digest {day}"], check=True)
        if subprocess.run(["git", "-C", runs_wt, "push", "-q", "origin", "HEAD:worker-runs"]).returncode == 0:
            print(f"digest/{day}.md written; {len(flagged)} sentence(s) flagged")
            return 0
    return 1


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "facts":
        h = int(a[a.index("--hours") + 1]) if "--hours" in a else 24
        sys.exit(facts(a[1], a[2], a[3], a[4], h))
    sys.exit({"summary": lambda: summary(a[1]), "write": lambda: write(a[1], a[2])}[a[0]]())
=== .github/workflows/worker.yml
# Queue worker (queue P8, first version): free models work queue items without a chat. Job types (worker/queue_ops.py):
# doc fixes with exact targets (doc_job.py) and literature cards (research_job.py, queue P7).
# Three jobs, as P5 and P8 require: `claim` (writes the queue, no keys), `draft` (keys, read-only token, runs nothing a
# model writes), `land` (writes, no keys: checks, applies, merges or opens a PR, records usage, releases the claim).
# Pause switch: a file worker/PAUSE on main stops every run before it claims anything.
name: worker
on:
  schedule:
    - cron: "17 * * * *"   # hourly; an idle run costs nothing on this public repo
  workflow_dispatch:
permissions: {}
concurrency:
  group: worker-${{ github.run_id }}   # runs may overlap on purpose: the queue's git CAS is the lock (P8's race test)
defaults:
  run:
    shell: bash
env:
  WORKER: worker-${{ github.run_id }}
jobs:
  claim:
    runs-on: ubuntu-24.04
    timeout-minutes: 10
    permissions:
      contents: write
    outputs:
      item: ${{ steps.claim.outputs.item }}
      type: ${{ steps.claim.outputs.type }}
    steps:
      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0
      - name: pause switch
        run: |
          if [ -f worker/PAUSE ]; then echo "paused by worker/PAUSE"; echo "PAUSED=1" >> "$GITHUB_ENV"; fi
      - name: tests
        if: env.PAUSED != '1'
        run: python3 -m unittest worker/test_worker.py
      - name: claim the top ready item
        id: claim
        if: env.PAUSED != '1'
        run: |
          git config --global user.name exagium-worker; git config --global user.email worker@exagium.invalid
          git worktree add -q --detach "$RUNNER_TEMP/q" HEAD
          rc=0; python3 -I worker/queue_ops.py claim "$RUNNER_TEMP/q" "$WORKER" "$RUNNER_TEMP/claim.json" || rc=$?; echo "claim exit $rc"
          cat "$RUNNER_TEMP/claim.json"
          id=$(python3 -c "import json;i=json.load(open('$RUNNER_TEMP/claim.json')).get('item');print(i['id'] if i else '')")
          typ=$(python3 -c "import json,sys;sys.path.insert(0,'worker');import queue_ops as Q;i=json.load(open('$RUNNER_TEMP/claim.json')).get('item');print(Q.job_type(i) if i else '')")
          echo "item=$id" >> "$GITHUB_OUTPUT"; echo "ITEM=$id" >> "$GITHUB_ENV"; echo "type=$typ" >> "$GITHUB_OUTPUT"; echo "TYPE=$typ" >> "$GITHUB_ENV"
      - name: reserve budget (1 drafter call, 1 reviewer call)
        if: env.ITEM != ''
        run: |
          if git ls-remote --exit-code origin worker-runs >/dev/null; then
            git worktree add -q --detach "$RUNNER_TEMP/r" origin/worker-runs 2>/dev/null || { git fetch -q origin worker-runs && git worktree add -q --detach "$RUNNER_TEMP/r" FETCH_HEAD; }
          else
            git worktree add -q --orphan -b worker-runs "$RUNNER_TEMP/r" && git -C "$RUNNER_TEMP/r" commit -q --allow-empty -m "worker-runs: ledger and digests" \
              && git -C "$RUNNER_TEMP/r" push -q origin worker-runs
          fi
          rc=0; python3 -I worker/ledger.py reserve "$RUNNER_TEMP/r" "$WORKER" cloudflare:1,nvidia:1 || rc=$?
          if [ "$rc" != 0 ]; then
            echo '{"result": "not started: budget cap or ledger push failed", "where": ""}' > "$RUNNER_TEMP/note.json"
            python3 -I worker/queue_ops.py release "$RUNNER_TEMP/q" "$WORKER" "$ITEM" open "$RUNNER_TEMP/note.json"
            echo "ITEM=" >> "$GITHUB_ENV"; echo "item=" >> "$GITHUB_OUTPUT"
          fi
          echo "reserve exit $rc"
      - name: prompt bundle (doc) or literature search by API scripts (research, no keys)
        if: env.ITEM != ''
        run: |
          if [ "$TYPE" = doc ]; then python3 -I worker/doc_job.py context "$RUNNER_TEMP/claim.json" . "$RUNNER_TEMP/job"
          else python3 -I worker/research_job.py search "$RUNNER_TEMP/claim.json" "$RUNNER_TEMP/job"; fi
          cp "$RUNNER_TEMP/claim.json" "$RUNNER_TEMP/job/"
      - uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02  # v4.6.2
        if: env.ITEM != ''
        with:
          name: bundle
          path: ${{ runner.temp }}/job/
  draft:
    needs: claim
    if: needs.claim.outputs.item != ''
    runs-on: ubuntu-24.04
    timeout-minutes: 20
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0
        with:
          persist-credentials: false
      - uses: actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093  # v4.3.0
        with:
          name: bundle
          path: job
      - name: drafter and reviewer (free models; their text is parsed as JSON, never run)
        env:
          CLOUDFLARE_API_TOKEN: ${{ secrets.CLOUDFLARE_API_TOKEN }}
          CLOUDFLARE_ACCOUNT_ID: ${{ secrets.CLOUDFLARE_ACCOUNT_ID }}
          NVIDIA_API_KEY: ${{ secrets.NVIDIA_API_KEY }}
          TYPE: ${{ needs.claim.outputs.type }}
        run: |
          if [ "$TYPE" = doc ]; then python3 -I worker/doc_job.py draft job; else python3 -I worker/research_job.py read job; fi
      - uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02  # v4.6.2
        if: ${{ !cancelled() }}
        with:
          name: draft
          path: job/
  land:
    needs: [claim, draft]
    if: ${{ always() && needs.claim.outputs.item != '' }}
    runs-on: ubuntu-24.04
    timeout-minutes: 15
    permissions:
      contents: write
      pull-requests: write
    env:
      ITEM: ${{ needs.claim.outputs.item }}
      TYPE: ${{ needs.claim.outputs.type }}
      GH_TOKEN: ${{ github.token }}
    steps:
      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0
        with:
          fetch-depth: 5
      - uses: actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093  # v4.3.0
        continue-on-error: true
        with:
          name: draft
          path: job
      - name: check, apply, merge or open a PR
        run: |
          git config --global user.name exagium-worker; git config --global user.email worker@exagium.invalid
          RUN_URL="$GITHUB_SERVER_URL/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID"
          if [ ! -f job/draft.json ] && [ ! -f job/read.json ]; then
            echo "{\"result\": \"failed: no draft (draft job did not finish)\", \"where\": \"$RUN_URL\"}" > note.json
            echo "STATUS=open" >> "$GITHUB_ENV"; exit 0
          fi
          if [ "$TYPE" = doc ]; then python3 -I worker/doc_job.py apply . job; else python3 -I worker/research_job.py card . job; fi
          res=$(python3 -c "import json;print(json.load(open('job/result.json'))['result'])")
          msg="worker: $ITEM ($(python3 -c "import json;print((json.load(open('job/result.json')).get('note') or '')[:150])"))"
          git add -A -- '*.md'   # stage first: a research card is a new file, which plain git diff would not see
          if [ "$res" = merge ] && ! git diff --cached --quiet; then
            git commit -qm "$msg" -m "$TYPE job: drafted by gpt-oss-120b (Cloudflare), checked by Nemotron 3 Ultra (NIM) and by worker/${TYPE}_job.py. Run: $RUN_URL"
            for i in 1 2 3; do git pull -q --rebase origin main && git push -q origin HEAD:main && break; sleep 5; done
            sha=$(git rev-parse --short HEAD)
            echo "{\"result\": \"merged\", \"where\": \"main $sha\", \"run\": \"$RUN_URL\", \"edits\": $(python3 -c "import json;print(json.load(open('job/result.json'))['edits'])")}" > note.json
            echo "STATUS=done" >> "$GITHUB_ENV"
          elif ! git diff --cached --quiet; then
            b="worker/$ITEM-$GITHUB_RUN_ID"; git switch -q -c "$b"; git commit -qm "$msg"; git push -q origin "$b"
            why=$(python3 -c "import json;print('; '.join(json.load(open('job/result.json'))['why'])[:900])")
            url=$(gh pr create --base main --head "$b" --title "$msg" --body "Queue item $ITEM, drafted by the worker; not merged because: $why. Run: $RUN_URL")
            echo "{\"result\": \"pr\", \"where\": \"$url\", \"why\": $(python3 -c "import json;print(json.dumps(json.load(open('job/result.json'))['why']))")}" > note.json
            echo "STATUS=waiting on the owner" >> "$GITHUB_ENV"
          else
            echo "{\"result\": \"no accepted edit\", \"where\": \"$RUN_URL\", \"why\": $(python3 -c "import json;print(json.dumps(json.load(open('job/result.json'))['why']))")}" > note.json
            echo "STATUS=waiting on the owner" >> "$GITHUB_ENV"
          fi
          cat note.json
      - name: record usage and release the claim
        if: always()
        run: |
          git config --global user.name exagium-worker; git config --global user.email worker@exagium.invalid
          [ -f note.json ] || echo '{"result": "failed before a note was written", "where": ""}' > note.json
          git fetch -q origin worker-runs && git worktree add -q --detach "$RUNNER_TEMP/r" FETCH_HEAD
          if [ -f job/calls.jsonl ]; then python3 -I worker/ledger.py record "$RUNNER_TEMP/r" "$WORKER" job/calls.jsonl || echo "ledger record exit $?"; fi
          git worktree add -q --detach "$RUNNER_TEMP/q" HEAD
          rc=0; python3 -I worker/queue_ops.py release "$RUNNER_TEMP/q" "$WORKER" "$ITEM" "${STATUS:-open}" note.json || rc=$?
          echo "release exit $rc"; exit $rc
      - uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02  # v4.6.2
        if: always()
        with:
          name: result
          path: |
            job/
            note.json
