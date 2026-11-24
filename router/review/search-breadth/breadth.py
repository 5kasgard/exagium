"""Search breadth test (PLAN.md). Needs SearXNG on SEARX_URL (default http://127.0.0.1:8080).
python3 -I router/review/search-breadth/breadth.py OUT_JSON [--dry]   (needs OPENROUTER_API_KEY unless --dry)"""
import json, os, sys, time, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "worker"))
import research_job as RJ  # noqa: E402  the production filter

SEARX = os.environ.get("SEARX_URL", "http://127.0.0.1:8080")
CAP = 76


def searx(q):
    url = SEARX + "/search?" + urllib.parse.urlencode({"q": q, "format": "json", "categories": "general,science"})
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            j = json.loads(r.read())
    except Exception as e:  # recorded, never guessed around
        return [], {"error": str(e)[:200]}
    res = [{"title": x.get("title") or "", "abstract": (x.get("content") or "")[:600], "url": x.get("url"),
            "via": ",".join(x.get("engines") or [])} for x in j.get("results", [])]
    return res, {"n": len(res), "unresponsive": j.get("unresponsive_engines", [])}


def round_robin(lists, cap):
    """Interleave result lists, dropping repeated URLs, up to cap. Pure."""
    out, seen = [], set()
    for i in range(max((len(l) for l in lists), default=0)):
        for l in lists:
            if i < len(l) and l[i]["url"] and l[i]["url"] not in seen:
                seen.add(l[i]["url"]); out.append(l[i])
                if len(out) == cap:
                    return out
    return out


def kept(q, pool, calls):
    ps = RJ.relevance(q, pool, calls)
    if ps is None:
        return None, RJ.REL_FAIL[:]
    return [dict(pool[i], p=round(p, 3)) for i, p in enumerate(ps) if p >= RJ.REL_KEEP], None


def main(out, dry=False):
    rows = []
    for q in json.load(open(os.path.join(HERE, "questions.json")))["questions"]:
        lists, logs = [], []
        for query in q["queries"]:
            res, log = searx(query); lists.append(res); logs.append(dict(log, query=query)); time.sleep(1)
        b = round_robin(lists, CAP)
        row = {"item": q["item"], "question": q["question"], "searx_log": logs, "b_found": len(b)}
        if not dry:
            calls = []
            for arm, pool in (("A", q["current"]), ("B", b)):
                k, err = kept(q["question"], pool, calls) if pool else ([], None)
                row[arm] = {"pool": len(pool), "kept": k, "error": err}
        else:
            row["B_sample"] = [x["title"][:80] for x in b[:5]]
        rows.append(row)
        print(json.dumps({"item": q["item"], "b_found": len(b), "A_kept": None if dry or row["A"]["kept"] is None else len(row["A"]["kept"]),
                          "B_kept": None if dry or row["B"]["kept"] is None else len(row["B"]["kept"])}))
    json.dump({"dry": dry, "rows": rows}, open(out, "w"), indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], "--dry" in sys.argv))
