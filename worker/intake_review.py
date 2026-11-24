#!/usr/bin/env python3
"""Turn an outside review package (audit/reviews/<name>/findings.jsonl + fixes.jsonl + RESPONSE.md) into worker-sized
code items, by script (owner 2026-11-21: the queue said "no work" while a 111-finding review sat unused).

    python3 -I worker/intake_review.py QUEUE_WORKTREE REVIEW_DIR [--dry]

An item is filed only if the code job could finish it AND the tier could merge it without a person:
  - the finding is still open (not listed with a commit, or as not changed, in RESPONSE.md);
  - its file passes code_job.allowed() and tier.MERGE_REFUSE (gates, money paths and agent internals stay with a person);
  - it is not in a trust or scoring path (record/, counter/, tasks/*/check.py: a person merges those);
  - it is not a design decision (kind "design" with no small fix): those go to the owner, listed in RESPONSE.md;
  - a test file for its directory exists (TESTS below).
Each item carries the finding's claim, its verbatim evidence, the reviewer's proposed fix and the repro, so the coder
works from evidence, and `review` (REVIEW:F0xx) for de-duplication: filing twice files nothing new.
Items are filed open: each finding was quote-checked by its reviewer at a named commit, and every change still passes
the item's test, the CI suite, a second-family code review and the tier's merge gate before it lands."""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import code_job as CJ  # noqa: E402
import queue_ops as Q  # noqa: E402
import tier  # noqa: E402  merge_allowed: file only what the tier may merge (audit 2026-11-21)

OWNER = {"F101"}   # changes what counts as a valid proposal on the board: decided with F054 by the owner, not merged by a tier
PERSON = re.compile(r"^(record|counter|ledger|probe)/|^tasks/[^/]+/check\.py$|^worker/(tier|generate|research_job|doc_job)\.py$"
                    r"|^router/(slots|engines|pick)\.py$|^facts/")   # tier.MERGE_REFUSE plus the trust and scoring paths
TESTS = [  # first match wins: (path prefix, test file)
    ("runner/proposals/confirm/", "runner/proposals/confirm/test_decide.py"),
    ("runner/proposals/", "runner/proposals/test_proposals.py"),
    ("runner/", "runner/test_runner.py"),
    ("router/", "router/test_router.py"),
    ("worker/", "worker/test_worker.py"),
    ("transfer/", "transfer/test_fetch.py"),
    ("experiments/", "experiments/test_experiments.py"),
    ("README.md", "worker/test_worker.py"),
]
ROOT = os.path.dirname(HERE)


def test_for(path):
    if CJ.TEST.fullmatch(path):
        return path if os.path.exists(os.path.join(ROOT, path)) else None
    for pre, t in TESTS:
        if path == pre or path.startswith(pre):
            return t if os.path.exists(os.path.join(ROOT, t)) else None
    return None


def closed_ids(response):
    """Findings RESPONSE.md lists as fixed (with a commit) or not changed (with a reason)."""
    fixed = set(re.findall(r"^\| (F\d+) \| [^|]*\| [0-9a-f]{7}", response, re.M))
    kept = set(re.findall(r"^\| (F\d+) \| [^|]*\| (?![0-9a-f]{7})", response.split("## Not changed", 1)[-1].split("## Open", 1)[0], re.M))
    return fixed | kept


def candidates(review_dir):
    load = lambda n: {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(review_dir, n)) if l.strip()}
    F, X, R = load("findings.jsonl"), load("fixes.jsonl"), load("rankings.jsonl")
    closed = closed_ids(open(os.path.join(review_dir, "RESPONSE.md")).read())
    name = os.path.basename(review_dir.rstrip("/"))
    out, skipped = [], {}
    for fid, f in sorted(F.items(), key=lambda kv: (R.get(kv[0], {}).get("body", {}).get("order") or 999, kv[0])):
        path, b = f["file"], f["body"]
        rk = R.get(fid, {}).get("body", {})
        why = ("closed" if fid in closed else "owner decision" if fid in OWNER else "workflow or refused path" if path.startswith("wf:") or not CJ.allowed(path)
               else "a person merges this path" if PERSON.search(path) or not tier.merge_allowed(path)
               else "design decision" if b.get("kind") == "design" and rk.get("effort") != "small"
               else "no test file for this path" if not test_for(path)
               else "a person merges its test file" if not tier.merge_allowed(test_for(path)) else None)
        if why:
            skipped[fid] = why
            continue
        fx = X.get(fid, {}).get("body", {})
        test = test_for(path)
        nxt = (f"Fix review finding {name}:{fid} ({b.get('kind')}, {rk.get('severity')}). Claim: {b['claim']} "
               f"Evidence at {f['file']} lines {f.get('lines')}: {str(b.get('evidence'))[:600]} "
               f"Proposed fix: {fx.get('approach', '')} Add a test in {test} that fails on the current code and passes "
               f"after the fix. Change nothing else.")
        out.append({"review": f"{name}:{fid}", "title": f"{fid}: {b['claim'][:110]}", "files": sorted({path} if path != test else {path}),
                    "test": test, "next": nxt, "severity": rk.get("severity") or "minor"})
    return out, skipped


def file_items(wt, items):
    for _ in range(Q.TRIES):
        Q.sync(wt)
        q = Q.load(wt)
        have = {i.get("review") for i in q["items"] if i.get("review")}
        new = [x for x in items if x["review"] not in have]
        if not new:
            return []
        n = max([int(m.group(1)) for i in q["items"] for m in [re.fullmatch(r"G(\d+)", str(i["id"]))] if m] or [0]) + 1
        made = []
        for x in new:
            gid = f"G{n}"; n += 1
            q["items"].append({"id": gid, "title": x["title"], "kind": "fix", "status": "open", "origin": "review",
                               "review": x["review"], "filed_at": Q.iso(Q.now()),
                               "code": {"files": x["files"], "test": x["test"]},
                               "touches": sorted(set(x["files"]) | {x["test"]}), "next": x["next"],
                               "refs": [f"audit/reviews/{x['review'].split(':')[0]}/findings.jsonl", "worker/intake_review.py"],
                               "severity": "major" if x["severity"] == "high" else "minor", "needs": "free", "cost": 0,
                               "value": "medium", "blocked_by": [], "batch": "B3"})
            made.append(gid)
        Q.save(wt, q)
        if Q.commit_push(wt, f"intake: {len(made)} code items from review {items[0]['review'].split(':')[0]} ({made[0]}-{made[-1]})"):
            return made
    raise RuntimeError("could not push the queue")


def main(a):
    items, skipped = candidates(a[1])
    print(f"intake: {len(items)} worker-sized, {len(skipped)} kept for a person or closed")
    for x in items:
        print(f"  {x['review']}: {x['files'][0]} (test {x['test']})")
    for fid, why in skipped.items():
        print(f"  skip {fid}: {why}")
    if "--dry" in a or not items:
        return 0
    made = file_items(a[0], items)
    print(f"filed {len(made)}: {', '.join(made)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
