"""P28 step 3: freeze the issue text (problem_statement) of step 1's usable tasks, from the same dataset rows the
candidates came from (illusion.FRESH). candidates.json stays as frozen on 2026-11-20; this adds statements.json beside it.
python3 -I statements.py OUT.json"""
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "illusion"))
import envcheck  # noqa: E402
import illusion  # noqa: E402


def main(out):
    want = set(envcheck.USABLE)
    cands = {c["instance_id"]: c for c in json.load(open(os.path.join(HERE, "candidates.json")))["candidates"]}
    ds, splits = illusion.FRESH
    got = {}
    for s in splits:
        for r in illusion.rows(ds, s):
            if r["instance_id"] in want:
                assert r["base_commit"] == cands[r["instance_id"]]["base_commit"], r["instance_id"]   # the same row
                got[r["instance_id"]] = r["problem_statement"]
    missing = sorted(want - set(got))
    if missing:
        raise SystemExit(f"not found: {missing}")
    json.dump({"source": [ds, splits], "statements": {i: {"text": got[i], "sha256": hashlib.sha256(got[i].encode()).hexdigest()}
                                                      for i in sorted(got)}}, open(out, "w"), indent=1)
    print(f"{len(got)} statements written")


if __name__ == "__main__":
    main(sys.argv[1])
