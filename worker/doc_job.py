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
