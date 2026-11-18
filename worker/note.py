"""Build the worker's run note and commit text from result.json, safely (review M1, m4).

python3 worker/note.py OUT_NOTE_JSON RESULT_KIND WHERE RUN_URL [RESULT_JSON]
python3 worker/note.py msg ITEM TYPE RESULT_JSON     # prints a one-line commit or PR title

Model text (the drafter's note, the reasons) never reaches a shell unquoted: it is read here, cleaned (no newlines,
no #N issue references, no @mentions) and written as JSON or printed as one line.
"""
import json, os, re, sys


def clean(t, n=200):
    t = re.sub(r"https?://\S+|www\.\S+|[@#](?=\w)|[\r\n`$\\]", " ", str(t or ""))   # review N5: no URLs either
    return " ".join(t.split())[:n]


def load(p):
    try:
        r = json.load(open(p))
        return r if isinstance(r, dict) else {}
    except (OSError, ValueError):
        return {}


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "msg":
        r = load(a[3])
        print(clean(f"worker: {a[1]} ({a[2]}): {r.get('note') or r.get('path') or r.get('result')}", 160))
    else:
        r = load(a[4]) if len(a) > 4 else {}
        note = {"result": a[1], "where": clean(a[2], 300), "run": a[3],
                "edits": int(r.get("edits") or 0) if str(r.get("edits", "0")).isdigit() else 0,
                "why": [clean(x, 300) for x in (r.get("why") or []) if isinstance(x, str)][:6]}
        json.dump(note, open(a[0], "w"), indent=1)
        print(json.dumps(note))
