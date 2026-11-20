#!/usr/bin/env python3
"""The repository by its parts, by script: every Python module's purpose line, its functions and classes with their
signatures and first docstring lines, its imports of other repo modules, and the tests that import it.
Retrieval picks the parts an item's text is about, so a job loads the slice it needs instead of a bare file list
(owner 2026-11-21: z-axis partitioning for context; measured that day: worker prompts had a median of 512 tokens and
the splitter saw only file names).

    python3 -I worker/repo_index.py REPO "item text" [K]   -> the K most relevant parts, as the prompt would show them
Nothing is executed: modules are read with ast."""
import ast, math, os, re, sys

DIRS = ("worker", "router", "runner", "facts", "transfer", "record", "counter", "probe", "loop", "audit", "tasks", "experiments")
WORD = re.compile(r"[a-z][a-z0-9]{2,}")
STOP = set("the and for with that this from not are was were has have its into when then than each only any all one two "
           "new add use used make run runs def self return none true false import file files item items test tests".split())


def words(text):
    return [w for w in WORD.findall(text.lower().replace("_", " ")) if w not in STOP]


def module(repo, path):
    src = open(os.path.join(repo, path), encoding="utf-8", errors="replace").read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    doc = (ast.get_docstring(tree) or "").strip().split("\n")[0][:200]
    parts, imports = [], set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if isinstance(n, ast.ClassDef):
                sig = f"class {n.name}"
            else:
                a = n.args
                names = [x.arg for x in a.posonlyargs + a.args] + ([f"*{a.vararg.arg}"] if a.vararg else []) \
                    + [x.arg for x in a.kwonlyargs] + ([f"**{a.kwarg.arg}"] if a.kwarg else [])
                sig = f"def {n.name}({', '.join(names)})"
            d = (ast.get_docstring(n) or "").strip().split("\n")[0][:160]
            parts.append({"sig": sig, "doc": d, "line": n.lineno})
        elif isinstance(n, ast.Import):
            imports.update(x.name.split(".")[0] for x in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            imports.add(n.module.split(".")[0])
    for n in ast.walk(tree):   # imports inside functions count too (the worker imports lazily)
        if isinstance(n, ast.Import):
            imports.update(x.name.split(".")[0] for x in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            imports.add(n.module.split(".")[0])
    return {"path": path, "doc": doc, "lines": src.count("\n"), "parts": parts, "imports": sorted(imports)}


def build(repo):
    mods = []
    for d in DIRS:
        for root, _, fs in os.walk(os.path.join(repo, d)):
            for f in sorted(fs):
                if f.endswith(".py"):
                    m = module(repo, os.path.relpath(os.path.join(root, f), repo))
                    if m:
                        mods.append(m)
    stem = {}
    for m in mods:
        stem.setdefault(os.path.splitext(os.path.basename(m["path"]))[0], []).append(m["path"])
    for m in mods:   # which tests import this module (by module name, as the tests do)
        name = os.path.splitext(os.path.basename(m["path"]))[0]
        m["tested_by"] = sorted(t["path"] for t in mods if os.path.basename(t["path"]).startswith("test_") and name in t["imports"]
                                and os.path.dirname(t["path"]).split("/")[0] == os.path.dirname(m["path"]).split("/")[0])
    return mods


def rank(mods, text, k=8):
    """The k modules whose names, purpose lines and part names share the most rare words with the text (tf-idf)."""
    docs = {m["path"]: words(m["path"] + " " + m["doc"] + " " + " ".join(p["sig"] + " " + p["doc"] for p in m["parts"]))
            for m in mods}
    df = {}
    for ws in docs.values():
        for w in set(ws):
            df[w] = df.get(w, 0) + 1
    n, q = len(docs), set(words(text))
    score = {p: sum(math.log(n / df[w]) * (1 + math.log(ws.count(w))) for w in q if w in ws) for p, ws in docs.items()}
    named = {m["path"] for m in mods if m["path"] in text or os.path.basename(m["path"]) in text}   # a path the item names comes first
    return sorted((m for m in mods if score[m["path"]] > 0 or m["path"] in named),
                  key=lambda m: (m["path"] not in named, -score[m["path"]]))[:k]


def rank_bm25(mods, text, k=8, k1=1.5, b=0.75):
    """Okapi BM25 over the same per-module words (ported from RabbitHole rabbithole/search.py LocalCorpusProvider, owner's
    repo; P26). Kept beside rank() until experiments/retrieval_bm25.py decides which one the jobs use."""
    docs = {m["path"]: words(m["path"] + " " + m["doc"] + " " + " ".join(p["sig"] + " " + p["doc"] for p in m["parts"]))
            for m in mods}
    df = {}
    for ws in docs.values():
        for w in set(ws):
            df[w] = df.get(w, 0) + 1
    n, avg = len(docs), (sum(len(w) for w in docs.values()) / len(docs)) if docs else 0.0
    q = words(text)
    score = {}
    for p, ws in docs.items():
        s, dl = 0.0, len(ws)
        for t in q:
            if t in df:
                tf = ws.count(t)
                idf = math.log(1.0 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * (tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / max(1.0, avg))))
        score[p] = s
    named = {m["path"] for m in mods if m["path"] in text or os.path.basename(m["path"]) in text}
    return sorted((m for m in mods if score[m["path"]] > 0 or m["path"] in named),
                  key=lambda m: (m["path"] not in named, -score[m["path"]]))[:k]


def show(ms, max_parts=25):
    out = []
    for m in ms:
        head = f"{m['path']} ({m['lines']} lines){' - ' + m['doc'] if m['doc'] else ''}"
        tb = f"  tested by: {', '.join(m['tested_by'])}" if m["tested_by"] else "  tested by: (none found)"
        body = [f"  {p['sig']}{'  # ' + p['doc'] if p['doc'] else ''}" for p in m["parts"][:max_parts]]
        more = [f"  ... {len(m['parts']) - max_parts} more"] if len(m["parts"]) > max_parts else []
        out.append("\n".join([head, tb] + body + more))
    return "\n\n".join(out)


if __name__ == "__main__":
    a = sys.argv[1:]
    print(show(rank(build(a[0]), a[1], int(a[2]) if len(a) > 2 else 8)))
