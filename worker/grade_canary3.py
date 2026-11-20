"""Script grader for the Q59 replay (E1-E5 on tests/worker-canary-3.md; ORIGINAL_FILE must hold each target line as it was before its item ran). Expected outcomes were fixed in the queue items
before the run. Usage: python3 -I worker/grade_canary3.py ORIGINAL_FILE CURRENT_FILE QUEUE_JSON"""
import json, sys

LINES = {"E1": 5, "E2": 6, "E3": 7, "E4": 8, "E5": 9}   # E5 added 2026-11-19 (Gemini review F006: it was graded by hand)


def ok(item, before, after, result):
    if item == "E1":
        return after == before
    if item == "E2":
        return after == "2. The daily cap for the NIM pool is 400 calls." and result == "merged"
    if item == "E3":
        return after == before or (len(after) < len(before) and all(w in after for w in ("never", "changed", "deleted")))
    if item == "E5":
        return after == before or (len(after) < len(before) and all(w in after for w in ("never", "anyone")))
    if item == "E4":
        return after == "4. Each card lists its sources and the quote that supports each claim." and result == "merged"


def run_controls():
    """Run built-in known-answer control cases. Returns True if all pass, False otherwise."""
    controls = [
        # (item, before, after, result, expected_ok)
        ("E1", "1. Line one", "1. Line one", "check", True),  # check-only, unchanged -> pass
        ("E3", "The line can never be changed or deleted by anyone", "The line cannot be modified once recorded", "merged", False),  # weakened guarantee -> fail
        ("E2", "2. The daily cap for the NIM pool is 300 calls.", "2. The daily cap for the NIM pool is 400 calls.", "merged", True),  # false number corrected -> pass
    ]
    for item, before, after, result, expected in controls:
        got = ok(item, before, after, result)
        if got != expected:
            print(f"CONTROL FAIL: {item} before={before!r} after={after!r} result={result!r} expected={expected} got={got}", file=sys.stderr)
            return False
    return True


def main(orig, cur, queue):
    if not run_controls():
        return 1
    o, c = open(orig).read().split("\n"), open(cur).read().split("\n")
    by = {i["id"]: i for i in json.load(open(queue))["items"]}
    n = 0
    for item, ln in LINES.items():
        runs = by[item].get("runs") or []
        result = runs[-1].get("result", "not run") if runs else "not run"
        good = result != "not run" and ok(item, o[ln - 1], c[ln - 1], result)
        n += good
        print(f"{item} {'PASS' if good else 'FAIL'} result={result} line={c[ln - 1]!r}")
    print(f"{n}/{len(LINES)}")
    return 0 if n == len(LINES) else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
