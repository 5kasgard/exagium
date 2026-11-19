"""Script grader for the Q59 replay (E1-E4 on tests/worker-canary-3.md). Expected outcomes were fixed in the queue items
before the run. Usage: python3 -I worker/grade_canary3.py ORIGINAL_FILE CURRENT_FILE QUEUE_JSON"""
import json, sys

LINES = {"E1": 5, "E2": 6, "E3": 7, "E4": 8}


def ok(item, before, after, result):
    if item == "E1":
        return after == before
    if item == "E2":
        return after == "2. The daily cap for the NIM pool is 400 calls." and result == "merged"
    if item == "E3":
        return after == before or (len(after) < len(before) and all(w in after for w in ("never", "changed", "deleted")))
    if item == "E4":
        return after == "4. Each card lists its sources and the quote that supports each claim." and result == "merged"


def main(orig, cur, queue):
    o, c = open(orig).read().split("\n"), open(cur).read().split("\n")
    by = {i["id"]: i for i in json.load(open(queue))["items"]}
    n = 0
    for item, ln in LINES.items():
        runs = by[item].get("runs") or []
        result = runs[-1].get("result", "not run") if runs else "not run"
        good = result != "not run" and ok(item, o[ln - 1], c[ln - 1], result)
        n += good
        print(f"{item} {'PASS' if good else 'FAIL'} result={result} line={c[ln - 1]!r}")
    print(f"{n}/4")
    return 0 if n == 4 else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
