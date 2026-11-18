"""SAT checker: python3 -I check.py IN KEY OUT  (runner/CONTRACT.md). KEY is unused: the checker verifies.

Prints one JSON line per item: {"score": 1 if the output assigns every variable 0 or 1 and satisfies every clause,
else 0, "satisfied": fraction of clauses satisfied (0 when the output is malformed), "malformed": bool}.
Exit 1 if OUT is not exactly one record per input; the output is untrusted bytes.
"""
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runner"))
import records  # noqa: E402


def main():
    ins = records.read(sys.argv[1])
    try:
        outs = records.read(sys.argv[3], max_len=255, count=len(ins))
    except ValueError as e:
        print(f"malformed output: {e}", file=sys.stderr)
        sys.exit(1)
    for x, o in zip(ins, outs):
        n, m = x[0], x[1] | (x[2] << 8)
        lits = [v - 256 if v > 127 else v for v in x[3:3 + 3 * m]]
        if len(o) != n or any(b > 1 for b in o):
            print(json.dumps({"score": 0, "satisfied": 0.0, "malformed": True}))
            continue
        sat = sum(any((o[abs(l) - 1] == 1) == (l > 0) for l in lits[3 * c:3 * c + 3]) for c in range(m))
        print(json.dumps({"score": int(sat == m), "satisfied": sat / m, "malformed": False}))


if __name__ == "__main__":
    main()
