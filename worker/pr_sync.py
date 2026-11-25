#!/usr/bin/env python3
"""PR sync (queue P38, 2026-11-25): the queue learns when a code item's pull request was merged or closed by someone else.

    python3 -I worker/pr_sync.py QUEUE_WORKTREE [--dry-run]     # needs GH_TOKEN (read access to pull requests)

Why: on 2026-11-25, 4 of 9 code items "waiting on the owner" had PRs already merged (#40, #66, #80, #118) and nothing
moved them, so they sat as open work for days. Only reads GitHub; no model call, no secret but the read token.

Rule, per item whose status is "waiting on the owner" or "escalated" and whose latest worker run is a PR:
- PR merged          -> decide_done ("merged to main as <sha>")
- PR closed unmerged -> park (the reason is the PR's own record; the splitter may cut the parent again)
- PR open or unknown -> nothing (an open PR is the owner's or the tier's to judge; a failed fetch changes nothing)
Items in "review" are left to worker/tier.py, which merges or closes their PRs itself.
"""
import json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import queue_ops as Q  # noqa: E402

ACTOR = "pr-sync"
WATCH = ("waiting on the owner", "escalated")
PR_URL = re.compile(r"/pull/(\d+)")


def pr_number(item):
    """The PR of the item's latest worker run, if that run made one (the same reading as tier.queue_authorises_pr)."""
    run = next((r for r in reversed(item.get("runs") or []) if r.get("outcome") == "pr"), None)
    m = PR_URL.search(str((run or {}).get("where", "")))
    return int(m.group(1)) if m else None


def decide(item, pr):
    """(outcome, reason) for one item given its PR's state {"state", "merged", "sha"}, or None. Pure."""
    if item.get("status") not in WATCH or not isinstance(pr, dict):
        return None
    if pr.get("merged"):
        return "decide_done", f"PR #{pr['number']} merged to main as {str(pr.get('sha') or '')[:7]} (found by {ACTOR})"
    if pr.get("state") == "closed":
        return "park", f"PR #{pr['number']} closed without merging (found by {ACTOR}); the PR holds the reason"
    return None


def fetch(n, repo=None):
    repo = repo or os.environ.get("GITHUB_REPOSITORY", "5kasgard/exagium")
    r = subprocess.run(["gh", "api", f"repos/{repo}/pulls/{n}", "-q", "{state, merged, sha: .merge_commit_sha}"],
                       capture_output=True, text=True)
    if r.returncode:
        return None
    try:
        d = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    d["number"] = n
    return d


def plan(q, get=fetch):
    out = []
    for i in q["items"]:
        if i.get("status") not in WATCH:
            continue
        n = pr_number(i)
        d = decide(i, get(n)) if n else None
        if d:
            out.append((i["id"], *d))
    return out


def main(wt, dry=False, get=fetch, move=Q.move):
    Q.sync(wt)
    todo = plan(Q.load(wt), get)
    for item, outcome, reason in todo:
        print(f"{item}: {outcome}: {reason}" + (" (dry run)" if dry else ""))
        if not dry:
            move(wt, ACTOR, item, outcome, reason)
    print(f"pr-sync: {len(todo)} item(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], "--dry-run" in sys.argv[2:]))
