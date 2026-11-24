"""List model call sites per file (K62)."""
import json, os, sys

PATTERNS = ("engines.call(", "slots.call(", "anthropic_call(")


def _skip(rel):
    parts = rel.replace(os.sep, "/").split("/")
    return os.path.basename(rel).startswith("test_") or "test" in parts[:-1]


def sites(root):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__")]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if _skip(rel):
                continue
            try:
                with open(full, encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        for call in PATTERNS:
                            if call in line:
                                out.append({"file": rel, "line": i, "call": call[:-1]})
            except OSError:
                pass
    return sorted(out, key=lambda r: (r["file"], r["line"], r["call"]))


def counts(records):
    c = {}
    for r in records:
        c[r["file"]] = c.get(r["file"], 0) + 1
    return c


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    recs = sites(root)
    print(json.dumps(recs, indent=2))
    print(json.dumps(counts(recs), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
