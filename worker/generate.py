"""Queue P27, kind 1: when no worker item is ready, file research questions for open items that cite no outside evidence.

python3 -I worker/generate.py QUEUE_WT OUT_DIR   (needs OPENROUTER_API_KEY; writes OUT_DIR/calls.jsonl for the ledger)

Data that opened the gate (P27's own rule: 2 days with idle hours while budget remained and no item was ready):
2026-11-19, 2 runs found no item; 2026-11-20, 6 runs found no item (07:30-10:22Z), every research item done or held.

Pipeline (worker/transitions.py): a generated item is born "proposed", which the worker never claims. A gate then moves
it: "accept" -> open, "reject" -> parked. Steps:
1. pick (script): open build/test/fix items with no research field, not used as a source before, highest value first.
2. draft (free model, PROPOSER): one research question and 3-5 search queries per source item, as JSON.
3. check (script): shape, length, no URLs, not a near-duplicate of any existing research question.
4. gate (decision model, another family, DECIDER): "would published research answering this inform the item's next
   step?", with two known-answer controls in the same request (one useful pair from the record, one off-topic pair).
   If either control comes out wrong, nothing is accepted: the drafts stay "proposed" for the Claude shift.
Caps: MAX_PER_DAY generated items per UTC day. Nothing here runs model text; the queue changes go through
queue_ops' compare-and-push and render.py's transition check.
"""
import datetime as dt, json, os, re, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "router"))
import queue_ops as Q
import transitions as T

PROPOSER = "particle-ai/particle-1:free"      # P31: 30/30 twice on the known-answer grading set
DECIDER = "perplexity/pplx-decider-v1.3:free"  # P31: checker trial 6/6 twice; another family than the proposer
MAX_PER_DAY = 3
KINDS = ("build", "test", "fix")
# Known-answer controls, frozen text (first live gate, 2026-11-20 10:52Z: a control built from live queue text scored
# 0.42 because P5's next step had moved on since R5 was filed; the drafts scored 0.96-0.99, the off-topic pair 0.005)
CONTROL_ITEM = ("Rank model families on tabular data by accuracy and cost",
                "Compare tuned gradient-boosted trees with small MLPs on tabular benchmarks before adding new families.")
CONTROL_GOOD_Q = "How accurate are tuned gradient-boosted decision trees compared with small MLPs on tabular benchmarks?"
CONTROL_BAD_Q = "What is the history of bagpipe making in the Scottish Highlands?"

DRAFT = """You write one research question for a literature search. The search reads paper abstracts, Wikipedia, Stack
Exchange, Hacker News, Reddit and GitHub. The question must be answerable from published work and must inform the
item's next step below. Do not ask about this project itself.

ITEM {id}: {title}
NEXT STEP: {next}

Reply with JSON only: {{"question": "...?", "queries": ["3 to 5 short search queries"], "why": "one sentence"}}"""


def words(s):
    return set(re.findall(r"[a-z0-9]+", s.lower())) - {"the", "a", "of", "and", "or", "to", "in", "for", "how", "what", "is", "are", "do", "does", "which", "on", "with"}


def near_dup(q, others, limit=0.6):
    a = words(q)
    return any(len(a & words(o)) / max(1, len(a | words(o))) >= limit for o in others)


def check(d):
    """Script check of a draft. Returns the cleaned draft or a reason."""
    if not isinstance(d, dict) or not isinstance(d.get("question"), str) or not isinstance(d.get("queries"), list):
        return None, "not the asked shape"
    q = " ".join(d["question"].split())
    qs = [" ".join(str(x).split()) for x in d["queries"] if isinstance(x, str)]
    if not (20 <= len(q) <= 300 and q.endswith("?")):
        return None, "question length or form"
    if not 3 <= len(qs) <= 5 or any(not 3 <= len(x) <= 80 for x in qs):
        return None, "queries count or length"
    if re.search(r"https?://|www\.", q + " ".join(qs)):
        return None, "contains a URL"
    if re.search(r"\b(exagium|forge|this project|our (queue|worker|repo))\b", q, re.I):
        return None, "asks about the project itself"
    return {"question": q, "queries": qs, "why": str(d.get("why", ""))[:200]}, None


def candidates(q):
    used = {i.get("generated_from") for i in q["items"] if i.get("generated_from")}
    out = [i for i in q["items"] if i.get("status") == "open" and i.get("kind") in KINDS and not i.get("research")
           and i["id"] not in used and i.get("needs") != "the owner"]
    return sorted(out, key=lambda i: ({"high": 0, "medium": 1, "low": 2}.get(i.get("value"), 3), i["id"]))


def today_count(q, day):
    return sum(1 for i in q["items"] if i.get("origin") == "generated" and str(i.get("generated_at", "")).startswith(day))


def next_rid(q):
    n = [int(m.group(1)) for i in q["items"] for m in [re.fullmatch(r"R(\d+)", str(i["id"]))] if m]
    return max(n, default=0) + 1


ASK_RESEARCH = ("Pair {k}: would published research that answers its QUESTION directly inform or change its ITEM's NEXT STEP? "
                "Off-topic, too vague to search, or about something the item does not need: no.")


def decide(pairs, calls, ask=ASK_RESEARCH, label="QUESTION"):
    """One Decisions API request: one yes/no question per (candidate, item title, item next) pair. Returns p per pair."""
    state = "\n\n".join(f"PAIR {k}:\n{label}: {qq}\nITEM: {t}\nNEXT STEP: {nx[:600]}" for k, (qq, t, nx) in enumerate(pairs))
    body = {"model": DECIDER, "state": state,
            "questions": {f"p{k}": {"type": "noul", "instructions": ask.format(k=k)} for k in range(len(pairs))}}
    req = urllib.request.Request("https://openrouter.ai/api/alpha/decisions", data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            resp, st = json.loads(r.read()), r.status
    except Exception as e:
        resp, st = {"error": str(e)[:200]}, getattr(e, "code", 0)
    calls.append({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "engine": "openrouter", "model": DECIDER,
                  "ok": st == 200, "status": st, "latency_s": round(time.time() - t0, 2), "usage": resp.get("usage") or {}})
    ps = [((resp.get("answers") or {}).get(f"p{k}") or {}).get("noul") for k in range(len(pairs))] if st == 200 else []
    if st != 200 or any(not isinstance(p, (int, float)) for p in ps):
        return None, f"status {st}: {str(resp.get('error', 'missing answers'))[:150]}"
    return ps, None


def controls():
    return [(CONTROL_GOOD_Q, *CONTROL_ITEM), (CONTROL_BAD_Q, *CONTROL_ITEM)]


def regate(wt, decide_fn, calls):
    """Items still 'proposed' (a withheld gate) get the gate again, before any new draft."""
    Q.sync(wt)
    q = Q.load(wt)
    by = {i["id"]: i for i in q["items"]}
    pend = [i for i in q["items"] if i.get("status") == "proposed" and i.get("generated_from") in by][:MAX_PER_DAY]
    if not pend:
        return []
    pairs = controls() + [(i["research"]["question"], by[i["generated_from"]]["title"], by[i["generated_from"]].get("next", ""))
                          for i in pend]
    ps, err = decide_fn(pairs)
    outcomes = gate_outcomes(ps, len(pairs) - 2)
    print(f"regate: {[i['id'] for i in pend]} p={ps} err={err}")
    return move_all(wt, [i["id"] for i in pend], outcomes, ps) if outcomes else []


def move_all(wt, ids, outcomes, ps):
    moved = []
    for k, rid in enumerate(ids):
        for _ in range(Q.TRIES):
            Q.sync(wt)
            q = Q.load(wt)
            i = next(x for x in q["items"] if x["id"] == rid)
            if i["status"] != "proposed":
                break
            i["status"] = T.next_status(i["status"], outcomes[k])
            i.setdefault("runs", []).append({"at": Q.iso(Q.now()), "worker": "generator-gate", "outcome": outcomes[k],
                                             "result": f"decision model p={ps[k + 2]:.3f}; controls {ps[0]:.3f}/{ps[1]:.3f}"})
            Q.save(wt, q)
            if Q.commit_push(wt, f"generator gate: {rid} {outcomes[k]} -> {i['status']}"):
                moved.append((rid, outcomes[k]))
                break
    return moved


def gate_outcomes(ps, n):
    """Controls first: the useful pair must pass and the off-topic pair must fail, or nothing is decided (None)."""
    if ps is None or len(ps) != n + 2 or not (ps[0] >= 0.5 and ps[1] < 0.5):
        return None
    return ["accept" if p >= 0.5 else "reject" for p in ps[2:]]


def run(wt, out, draft_fn=None, decide_fn=None):
    """Every exit writes the calls made, so the ledger records them (GLM F021: a push failure or crash skipped it)."""
    os.makedirs(out, exist_ok=True)
    calls = []
    try:
        return _run(wt, out, calls, draft_fn, decide_fn)
    finally:
        _write_calls(out, calls)


def _run(wt, out, calls, draft_fn=None, decide_fn=None):
    log = []
    Q.sync(wt)
    q = Q.load(wt)
    day = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    regated = regate(wt, decide_fn or (lambda p: decide(p, calls)), calls)   # a withheld gate first, inside the cap
    if regated:
        print(f"generate: regated {regated}")
    q = Q.load(wt)
    room = MAX_PER_DAY - today_count(q, day)
    if room <= 0:
        print(f"generate: cap reached ({MAX_PER_DAY} today)")
        _write_calls(out, calls)
        json.dump({"regated": regated}, open(os.path.join(out, "generate.json"), "w"))
        return 0
    if draft_fn is None:
        import engines
        engines.TIMEOUT = 300

        def draft_fn(prompt):
            r = engines.call("openrouter", PROPOSER, prompt, max_tokens=1500)
            calls.append({k: r.get(k) for k in ("t", "engine", "model", "ok", "status", "latency_s", "usage", "error")})
            m = re.search(r"\{.*\}", r.get("text") or "", re.S)
            try:
                return json.loads(m.group(0)) if m else None
            except json.JSONDecodeError:
                return None
    existing = [i["research"]["question"] for i in q["items"] if i.get("research")]
    drafts = []
    for src in candidates(q):
        if len(drafts) >= room:
            break
        d, why = check(draft_fn(DRAFT.format(id=src["id"], title=src["title"], next=src.get("next", "")[:1500])))
        if d and near_dup(d["question"], existing + [x[1]["question"] for x in drafts]):
            d, why = None, "near-duplicate of an existing research question"
        log.append({"source": src["id"], "ok": d is not None, "why": why, "draft": d})
        if d:
            drafts.append((src, d))
    if not drafts:
        json.dump({"log": log}, open(os.path.join(out, "generate.json"), "w"), indent=1)
        _write_calls(out, calls)
        print("generate: no draft passed the script check")
        return 0
    pairs = controls() + [(d["question"], s["title"], s.get("next", "")) for s, d in drafts]
    ps, err = (decide_fn or (lambda p: decide(p, calls)))(pairs)
    outcomes = gate_outcomes(ps, len(drafts))
    controls_ok = outcomes is not None
    gate = {"controls_ok": controls_ok, "p": ps, "error": err}
    # write: born proposed, then moved by the gate through the transition table (two commits keep render's check exact)
    for _ in range(Q.TRIES):
        Q.sync(wt)
        q = Q.load(wt)
        if today_count(q, day) + len(drafts) > MAX_PER_DAY:
            break
        n, made = next_rid(q), []
        for s, d in drafts:
            rid = f"R{n}"; n += 1
            q["items"].append({"id": rid, "title": d["question"][:160], "kind": "test", "status": "proposed",
                               "origin": "generated", "generated_from": s["id"], "generated_at": Q.iso(Q.now()),
                               "research": {"question": d["question"], "queries": d["queries"]},
                               "touches": [f"research/cards/{rid}.md"], "refs": [f"queue {s['id']}", "queue P27 (generator)"],
                               "next": f"Worker job (research): literature card for {s['id']}. Why: {d['why']}",
                               "severity": "minor", "needs": "free", "cost": 0, "value": s.get("value", "medium"),
                               "blocked_by": [], "batch": s.get("batch", "B3")})
            made.append(rid)
        Q.save(wt, q)
        if Q.commit_push(wt, f"generator: {', '.join(made)} proposed (P27)"):
            break
    else:
        print("generate: could not push"); return 1
    moved = move_all(wt, made, outcomes, ps) if controls_ok else []
    json.dump({"log": log, "gate": gate, "made": made, "moved": moved}, open(os.path.join(out, "generate.json"), "w"), indent=1)
    _write_calls(out, calls)
    print(f"generate: proposed {made}; gate {'ran' if controls_ok else 'withheld (controls: ' + str(ps) + ', ' + str(err) + ')'}; moved {moved}")
    return 0


def _write_calls(out, calls):
    with open(os.path.join(out, "calls.jsonl"), "w") as f:
        for c in calls:
            f.write(json.dumps(c) + "\n")


if __name__ == "__main__":
    sys.exit(run(sys.argv[1], sys.argv[2]))
