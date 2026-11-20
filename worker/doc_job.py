"""Doc job for the queue worker (queue P8, first job type): fix a doc item whose exact targets are given.

python3 worker/doc_job.py context ITEM_JSON REPO OUT_DIR     # no keys: the prompt bundle (item + numbered target lines)
python3 worker/doc_job.py draft OUT_DIR                      # keys: drafter proposes edits, a reviewer of another family judges
python3 worker/doc_job.py apply REPO OUT_DIR                  # no keys: validate, apply, decide merge or PR

Nothing a model writes is executed: its reply is parsed as JSON and applied as text edits by this script.
An edit is accepted only if its file is one of the item's targets, its line is inside a target range, its "old" text
equals that line exactly, and the file is Markdown. The result is "merge" (self-merge, as P8's spec allows for doc
items) only if every edit is accepted, there is at least one, and the reviewer approves; else "pr" (a human or Claude
decides) with the reasons. Drafter: gpt-oss-120b (Groq; Cloudflare if Groq fails). Reviewer: Nemotron 3 Ultra (NVIDIA NIM),
another family.
"""
import difflib, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DRAFTER = ("groq", "openai/gpt-oss-120b")             # same model on Groq's free tier, so Cloudflare's neurons go to research (P8r)
DRAFTER_FALLBACK = ("cloudflare", "@cf/openai/gpt-oss-120b")
REVIEWER = ("nvidia", "nvidia/nemotron-3-ultra-550b-a55b")
MARGIN = 3   # context lines shown around each target range

DRAFT_PROMPT = """You fix one documentation item in a repository. Change only the target lines shown, and only as the item asks.
Keep the file's style. Change as little as possible. Do not invent facts or numbers: use only what the item and the lines say.

Reply with JSON only, no prose, in this form:
{{"edits": [{{"path": "<file>", "line": <line number>, "old": "<the line exactly as shown>", "new": "<the replacement line>"}}],
 "note": "<one sentence: what you changed and why>"}}
Each edit replaces one whole line. "old" must equal the shown line exactly (without the number prefix).
If the item cannot be done within these lines, reply {{"edits": [], "note": "<why>"}}.
If the item asks you to check a line and change it only if it is wrong, a line that is true stays as it is, even when
its wording differs from the source: reply with no edits and say it is accurate (release test 2, D21).

ITEM {id}: {title}
WHAT TO DO: {next}

TARGET LINES (number | text):
{excerpts}
"""

REVIEW_PROMPT = """You review a proposed documentation fix. Approve only if the change does what the item asks, changes nothing else,
adds no fact or number that the item or the original lines do not support, and keeps the meaning correct. Also reject
wording that is redundant (e.g. "currently ... today"), vague, or weaker than the item asks: a careful maintainer must be
able to merge the line as it is (release test 2026-11-18, Q38). Reject any change that softens a guarantee or a rule
(for example "can never" -> "are expected to"). If the item says to change a line only if it is wrong, approve an edit only
if the original line states something false; rewording a true line to match the source is not a fix (release test 2, D21).

Reply with JSON only: {{"verdict": "approve" or "reject", "reason": "<one or two sentences>"}}

ITEM {id}: {title}
WHAT TO DO: {next}

PROPOSED CHANGE (unified diff):
{diff}
"""


# Never written by the worker: workflows, the worker and its records, git internals, registered plans.
REFUSE = re.compile(r"^\.github/|^worker/|^ledger/|^job/|(^|/)\.git(/|$)|PREREG|REGISTERED", re.I)
# Written only through a PR (a person decides): anything an agent reads as instructions (review N3: an allow-list for merge).
# the keyword may sit anywhere in the file name: tests/canary-BRIEF.md was committed directly in release test 2 (DB, Q59)
PR_ONLY = re.compile(r"(^|/)[^/]*(AGENTS|CLAUDE|GEMINI|VERIFY|BRIEF|SKILL|copilot-instructions)[^/]*\.md$|HANDOFF|(^|/)\.(claude|cursor)/", re.I)
TOKENS = re.compile(r"https?://\S+|www\.\S+|<[^>]*>|@[A-Za-z0-9][\w-]*|\]\([^)]*\)|javascript:\S*", re.I)


def safe_path(p):
    """A relative Markdown path inside the repo that the worker may write at all."""
    return (isinstance(p, str) and p.endswith(".md") and not p.startswith("/") and ".." not in p.split("/")
            and not REFUSE.search(p))


def merge_ok(p):
    return safe_path(p) and not PR_ONLY.search(p)


def safe_new(old, new):
    """Every link, HTML tag, mention or Markdown link target in the new line occurs in the old line, as often, and the
    line stays short (review N2: whole tokens, not their prefixes)."""
    have = TOKENS.findall(old)
    for t in TOKENS.findall(new):
        if t in have:
            have.remove(t)
        else:
            return False
    return len(new) <= max(2 * len(old), len(old) + 300)


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
    bad = [t.get("path") for t in item["targets"] if not safe_path(t.get("path"))
           or not os.path.isfile(os.path.join(repo, t["path"])) or t["path"] not in item.get("touches", [])]
    if bad:
        raise SystemExit(f"targets refused (must be existing Markdown, listed in touches, not a workflow, the worker or a registered plan): {bad}")
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


GUARANTEE = re.compile(r"\b(never|cannot|can't|anyone|no one|always|must|every|only)\b", re.I)


def lost_guarantee(old, new):
    """Words that carry a guarantee's strength or scope, present in the old line and gone from the new one. A script, not the
    reviewer, holds this line: the reviewer approved 'can never ... by anyone at all' -> 'cannot ... once recorded' (Q59 replay
    E3). Replay on release test 2's 30 items: catches Q38 and E3, blocks none of the 26 acceptable."""
    return sorted({w.lower() for w in GUARANTEE.findall(old)} - {w.lower() for w in GUARANTEE.findall(new)})


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
        if not isinstance(e, dict) or not {"path", "line", "old", "new"} <= set(e) \
                or not all(isinstance(e[k], str) for k in ("path", "old", "new")):
            reasons.append(f"malformed edit {str(e)[:120]}"); continue
        p, n = e["path"], e["line"]
        if p not in targets or not safe_path(p):
            reasons.append(f"{p}: not a Markdown target"); continue
        if not isinstance(n, int) or not any(a <= n <= b for a, b in targets[p]):
            reasons.append(f"{p}:{n}: outside the target lines"); continue
        lines = files[p].split("\n")
        if n > len(lines) or lines[n - 1] != e["old"]:
            reasons.append(f"{p}:{n}: 'old' does not match the line"); continue
        if "\n" in e["new"] or "\r" in e["new"] or e["new"] == e["old"]:
            reasons.append(f"{p}:{n}: 'new' is not one changed line"); continue
        if lost_guarantee(e["old"], e["new"]):
            reasons.append(f"{p}:{n}: drops a guarantee word ({', '.join(lost_guarantee(e['old'], e['new']))})"); continue
        if not safe_new(e["old"], e["new"]):
            reasons.append(f"{p}:{n}: 'new' adds a link, HTML or mention, or is too long"); continue
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
    engines.TIMEOUT = 600   # Nemotron with 16,000 tokens took over the router's 180 s (release test, R2)
    b = json.load(open(os.path.join(out, "bundle.json")))
    item, files, calls = b["item"], b["files"], []

    import slots

    def ask(engine, model, prompt):
        r = slots.call(engines, engine, model, prompt)
        if not r["ok"] and "timed out" in str(r.get("error")):   # one retry on a timeout
            calls.append(r)
            r = slots.call(engines, engine, model, prompt)
        calls.append(r)
        return r

    try:
        return _draft(out, b, item, files, calls, ask)
    finally:
        write_calls(out, calls)


def write_calls(out, calls):
    with open(os.path.join(out, "calls.jsonl"), "w") as f:
        for c in calls:
            f.write(json.dumps({k: c.get(k) for k in ("t", "engine", "model", "ok", "status", "latency_s", "usage", "error")}) + "\n")


def _draft(out, b, item, files, calls, ask):
    import slots
    obj = lambda t: (lambda j: j if isinstance(j, dict) else None)(parse_json(t))
    # the drafter slot's fallback chain (router/slots.py): Groq, Cloudflare, NIM, then paid when reserved
    proposal, d, dtries = slots.run("drafter", b["prompt"], ask, obj)
    model_fail = []   # a failed model call is an infrastructure failure, never a judgement on the item (review of Q60, 1)
    if proposal is None:
        quota = all(t["status"] == 429 for t in dtries)   # every filler's allowance spent: a budget wait, not a failure
        model_fail.append(("quota: " if quota else "") + ("drafter call failed" if not (d or {}).get("ok") else "drafter reply was not JSON"))
        proposal = {"edits": [], "note": model_fail[-1]}
    d = d or {"ok": False}
    edits, reasons = check_edits(item, files, proposal.get("edits"))
    review = None
    if edits:
        dd = diff(files, apply_edits(files, edits))
        review, r, _ = slots.run("reviewer", REVIEW_PROMPT.format(id=item["id"], title=item["title"], next=item["next"], diff=dd), ask, obj)
        if review is None:
            r = r or {"ok": False, "error": "no reviewer"}
            model_fail.append("reviewer call failed" if not r["ok"] else "reviewer reply was not JSON")
            review = {"verdict": "none", "reason": model_fail[-1] + ": " + str(r.get("error"))[:200]}
    json.dump({"proposal": proposal, "review": review, "pre_check": reasons, "model_fail": model_fail},
              open(os.path.join(out, "draft.json"), "w"), indent=1)
    print(f"drafter ok={d['ok']} edits={len(edits)} rejected={len(reasons)} review={review and review.get('verdict')}")
    return 0


def apply(repo, out):
    b = json.load(open(os.path.join(out, "bundle.json")))
    item = b["item"]
    files = {p: open(os.path.join(repo, p), encoding="utf-8").read() for p in b["files"]}
    d = json.load(open(os.path.join(out, "draft.json")))
    proposed = d["proposal"].get("edits")
    proposed = len(proposed) if isinstance(proposed, list) else 0
    if files != b["files"]:
        res = {"result": "stale", "why": ["a target file changed since the bundle was made"], "edits": 0}
    elif d.get("model_fail"):
        quota = all(str(m).startswith("quota: ") for m in d["model_fail"])
        res = {"result": "quota" if quota else "failed_call", "why": d["model_fail"], "edits": 0}
    elif d.get("review") and d["review"].get("verdict") not in ("approve", "reject"):   # GLM F025: no review happened
        res = {"result": "failed_call", "why": [f"reviewer gave no verdict: {d['review'].get('verdict')!r}"[:300]], "edits": 0}
    else:
        edits, reasons = check_edits(item, files, d["proposal"].get("edits"))   # checked again here, with no keys
        verdict = (d.get("review") or {}).get("verdict")
        why = reasons + ([] if edits else ["no accepted edit: " + str(d["proposal"].get("note", ""))[:300]]) \
            + ([] if verdict == "approve" else [f"reviewer: {verdict}: {(d.get('review') or {}).get('reason', '')}"[:400]])
        new = apply_edits(files, edits)
        for p, s in new.items():
            if s != files[p]:
                open(os.path.join(repo, p), "w", encoding="utf-8").write(s)
        open(os.path.join(out, "change.diff"), "w").write(diff(files, new))
        pr_only = sorted({e["path"] for e in edits if not merge_ok(e["path"])})
        if pr_only:
            why.append(f"agent-facing file(s), merged only by a person: {', '.join(pr_only)}")
        res = {"result": "merge" if not why else "pr", "why": why, "edits": len(edits), "proposed": proposed,
               "accepted": len(edits), "refused": len(reasons), "note": d["proposal"].get("note"), "review": d.get("review")}
        if edits and verdict == "reject":   # a change the reviewer rejected is not offered to a person as a PR (release test 2, Q38; Q59)
            for p in files:
                open(os.path.join(repo, p), "w", encoding="utf-8").write(files[p])
            res.update(result="rejected", edits=0)
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)
    print(json.dumps(res)[:600])
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit({"context": lambda: context(a[1], a[2], a[3]), "draft": lambda: draft(a[1]),
              "apply": lambda: apply(a[1], a[2])}[a[0]]())
