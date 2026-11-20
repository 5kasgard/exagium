"""Script grader for the Q59 replay (E1-E5 on tests/worker-canary-3.md; ORIGINAL_FILE must hold each target line as it was before its item ran). Expected outcomes were fixed in the queue items
before the run. Usage: python3 -I worker/grade_canary3.py ORIGINAL_FILE CURRENT_FILE QUEUE_JSON"""
import json, sys

LINES = {"E1": 5, "E3": 7, "E4": 8, "E5": 9}   # E2 is found by meaning; E5 added 2026-11-19 (Gemini review F006: it was graded by hand)


def ok(item, before, after, result):
    if item == "E1":
        return after == before
    if item == "E2":
        return result == "merged" and all(w in after.lower() for w in ("daily", "cap", "nim", "400", "calls"))
    if item == "E3":
        return after == before or (len(after) < len(before) and all(w in after for w in ("never", "changed", "deleted")))
    if item == "E5":
        return after == before or (len(after) < len(before) and all(w in after for w in ("never", "anyone")))
    if item == "E4":
        return after == "4. Each card lists its sources and the quote that supports each claim." and result == "merged"


def main(orig, cur, queue):
    o, c = open(orig).read().split("\n"), open(cur).read().split("\n")
    by = {i["id"]: i for i in json.load(open(queue))["items"]}
    n = 0
    items = ("E1", "E2", "E3", "E4", "E5")
    for item in items:
        runs = by[item].get("runs") or []
        result = runs[-1].get("result", "not run") if runs else "not run"
        if item == "E2":
            before = ""
            after = next((x for x in c if ok(item, None, x, "merged")), "")
        else:
            ln = LINES[item]
            before, after = o[ln - 1], c[ln - 1]
        good = result != "not run" and ok(item, before, after, result)
        n += good
        print(f"{item} {'PASS' if good else 'FAIL'} result={result} line={after!r}")
    print(f"{n}/{len(items)}")
    return 0 if n == len(items) else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
