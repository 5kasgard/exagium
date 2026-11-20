"""Does BM25 (ported from RabbitHole, P26) find the right module more often than the tf-idf ranker the jobs use now?"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAN = ("Labelled data already on main: GLM's 111 review findings (audit/reviews/glm-20261120/findings.jsonl), each "
        "naming the one file it is about. For every finding whose file is a Python module in worker/repo_index.py's "
        "index, rank modules from the finding's claim text with every path and file name removed, by tf-idf (rank, used "
        "now) and by BM25 (rank_bm25). Measure hit@3: the true file among the top 3. Rule, fixed before the run: holds "
        "(switch the jobs to BM25) if BM25's hit@3 is at least tf-idf's plus 0.05; fails otherwise; invalid if fewer "
        "than 30 findings qualify.")


def run():
    sys.path.insert(0, os.path.join(ROOT, "worker"))
    import repo_index as RI
    mods = RI.build(ROOT)
    paths = {m["path"] for m in mods}
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "audit/reviews/glm-20261120/findings.jsonl")) if l.strip()]
    hits = {"tfidf": 0, "bm25": 0}
    used = []
    for r in rows:
        if r["file"] not in paths:
            continue
        text = re.sub(r"\S*[/\\]\S*|\S+\.(py|md|yml|json|c|h)\b", " ", r["body"]["claim"])   # no path or file name
        used.append(r["id"])
        for name, fn in (("tfidf", RI.rank), ("bm25", RI.rank_bm25)):
            if r["file"] in [m["path"] for m in fn(mods, text, 3)]:
                hits[name] += 1
    n = len(used)
    return {"n": n, "hit3": {k: round(v / n, 3) if n else None for k, v in hits.items()}, "findings": used}


def rule(r):
    if r["n"] < 30:
        return {"verdict": "invalid", "why": f"only {r['n']} findings qualify (need 30)"}
    t, b = r["hit3"]["tfidf"], r["hit3"]["bm25"]
    return {"verdict": "holds" if b >= t + 0.05 else "fails", "why": f"hit@3 tf-idf {t}, BM25 {b} over {r['n']} findings"}
