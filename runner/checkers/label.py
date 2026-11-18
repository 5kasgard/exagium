"""The built-in "label" checker: python3 -I label.py IN KEY OUT  (runner/CONTRACT.md).

One JSON line per item: {"score": 1 if the output record equals the key record byte for byte, else 0}.
Exit 1 if OUT is not exactly one record per key, or a record is longer than the longest key.
"""
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import records  # noqa: E402


def main():
    keys = records.read(sys.argv[2])
    try:
        outs = records.read(sys.argv[3], max_len=max(map(len, keys), default=0), count=len(keys))
    except ValueError as e:
        print(f"malformed output: {e}", file=sys.stderr)
        sys.exit(1)
    for k, o in zip(keys, outs):
        print(json.dumps({"score": int(k == o)}))


if __name__ == "__main__":
    main()
