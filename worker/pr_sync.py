#!/usr/bin/env python3
"""PR sync (queue P38, 2026-11-25): the queue learns when a code item's pull request was merged or closed by someone else.

    python3 -I worker/pr_sync.py QUEUE_WORKTREE [--dry-run]     # needs GH_TOKEN (read access to pull requests)

Why: on 2026-11-25, 4 of 9 code items "waiting on the owner" had PRs already merged (#40, #66, #80, #118) and nothing
moved them, so they sat as open work for days. It reads GitHub only; no model call. --dry-run reads the worktree as it
is and writes nothing (no fetch, no reset).

Rule, per item whose status is "waiting on the owner" or "escalated" and whose LATEST worker event is that PR (a later
claim, redo or reopen means the PR is history: tier.py closes rejected PRs itself and sends the item back):
- PR merged          -> decide_done ("merged to main as <sha>")
- PR closed unmerged -> park (the PR holds the reason; a PR closed because another replaced it is parked too, and the
                        replacing PR's own item records the work)
- PR open or unknown -> nothing (an open PR is the owner's or the tier's to judge; a failed fetch changes nothing)
Items in "review" are left to worker/tier.py. Each move re-checks the item at write time (status and PR unchanged).
"""
import json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import queue_ops as Q  # noqa: E402
import transitions as T  # noqa: E402

ACTOR = "pr-sync"
WATCH = ("waiting on the owner", "escalated")
PR_URL = re.compile(r"/pull/(\d+)")
# events after which an earlier PR no longer speaks for the item (worker outcomes, a new claim, a tier or owner reopen)
LATER = set(T.WORKER) | {"claim", "redo", "decide_open", "resplit"}


def pr_number(item):
    """The PR number if the item's latest worker-side event is a PR, else None."""
    run = next((r for r in reversed(item.get("runs") or []) if r.get("outcome") in LATER), None)
    if not run or run.get("outcome") != "pr":
        return None
    m = PR_URL.search(str(run.get("where", "")))
    return int(m.group(1)) if m else None


def decide(item, pr):
    """(outcome, reason) for one item given its PR's state {"number", "state", "merged", "sha"}, or None. Pure."""
    if item.get("status") not in WATCH or not isinstance(pr, dict) or pr.get("number") != pr_number(item):
        return None
    if pr.get("merged") is True:
        return "decide_done", f"PR #{pr['number']} merged to main as {str(pr.get('sha') or '')[:7]} (found by {ACTOR})"
    if pr.get("state") == "closed" and pr.get("merged") is False:
        return "park", f"PR #{pr['number']} closed without merging (found by {ACTOR}); the PR holds the reason"
    return None


def fetch(n, repo=None):
    repo = repo or os.environ.get("GITHUB_REPOSITORY", "5kasgard/exagium")
    try:
        r = subprocess.run(["gh", "api", f"repos/{repo}/pulls/{n}", "-q", "{state, merged, sha: .merge_commit_sha}"],
                           capture_output=True, text=True, timeout=60)
        d = json.loads(r.stdout) if r.returncode == 0 else None
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return None
    if isinstance(d, dict):
        d["number"] = n
        return d
    return None


def plan(q, get=fetch):
    out = []
    for i in q["items"]:
        n = pr_number(i) if i.get("status") in WATCH else None
        pr = get(n) if n else None
        d = decide(i, pr) if pr else None
        if d:
            out.append((i["id"], n, *d))
    return out


def apply(wt, item_id, n, outcome, reason):
    """One move with the rule re-checked on the freshest queue (review 2026-11-25: plan-to-write gap). True if written,
    None if the item changed meanwhile (not a failure), False if the push kept losing."""
    for _ in range(Q.TRIES):
        Q.sync(wt)
        q = Q.load(wt)
        i = next((x for x in q["items"] if x["id"] == item_id), None)
        if i is None or i.get("status") not in WATCH or pr_number(i) != n:
            print(f"{item_id}: changed since the plan, left alone")
            return None
        to = T.next_status(i["status"], outcome)
        Q.event(i, ACTOR, outcome, i["status"], to, reason)
        i["status"] = to
        Q.save(wt, q)
        if Q.commit_push(wt, f"{ACTOR}: {item_id} {outcome} -> {to} (PR #{n})"):
            return True
    return False


def main(wt, dry=False, get=fetch, write=apply):
    if not dry:
        Q.sync(wt)
    todo = plan(Q.load(wt), get)
    failed = 0
    for item, n, outcome, reason in todo:
        print(f"{item}: {outcome}: {reason}" + (" (dry run)" if dry else ""))
        if not dry:
            try:
                ok = write(wt, item, n, outcome, reason)
            except (T.NotAllowed, subprocess.CalledProcessError) as e:
                print(f"{item}: not moved: {e}")
                ok = False
            failed += ok is False
    print(f"pr-sync: {len(todo)} item(s), {failed} not moved")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], "--dry-run" in sys.argv[2:]))
