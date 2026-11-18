"""Research job for the queue worker (queue P7 wired into P8): a literature card for one question.

python3 worker/research_job.py search ITEM_JSON OUT_DIR        # no keys: OpenAlex, arXiv, Semantic Scholar APIs (logged)
python3 worker/research_job.py read OUT_DIR                    # keys: a free model extracts claims with exact quotes;
                                                               # a second, of another family, checks each kept claim
python3 worker/research_job.py card REPO OUT_DIR               # no keys: verify quotes by script, write research/cards/<ID>.md

Routing by step, as P7 asks: search by script; first read by a free model; verify by script (every quote must occur
verbatim in the source's abstract, and every number in a claim must occur in its quote) plus a second free model;
Claude judges last (the card says so). Nothing a model writes is executed. An item opts in with
"research": {"question": "...", "queries": ["...", ...]} and "touches": ["research/cards/<ID>.md"].
"""
import json, os, re, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
READER = ("cloudflare", "@cf/openai/gpt-oss-120b")
CHECKER = ("nvidia", "nvidia/nemotron-3-ultra-550b-a55b")
PER_QUERY = 6
MAX_SOURCES = 18
UA = {"User-Agent": "exagium-research/1.0 (+https://github.com/5kasgard/exagium)"}

READ_PROMPT = """You read abstracts to answer a research question for a project that measures small AI models.
Extract only claims that bear on the question. For each, copy an exact quote from the abstract (verbatim, 8 to 40 words)
that states it; a claim without such a quote is not allowed. Prefer measured results, numbers to beat, known failure modes
and prior art. Do not add anything the abstracts don't say.

Reply with JSON only:
{{"claims": [{{"source": "<source id>", "claim": "<one sentence, in your words>", "quote": "<exact words from that abstract>",
  "kind": "result" | "frontier" | "failure mode" | "prior art" | "method"}}],
 "open_questions": ["<what these sources leave open about the question>"]}}

QUESTION: {question}

SOURCES:
{sources}
"""

CHECK_PROMPT = """Check each claim against its quote. Keep a claim only if the quote supports it, the claim adds nothing the quote
doesn't say, and it bears on the question. Reply with JSON only: {{"keep": [<indexes of claims to keep>], "notes": "<one sentence>"}}

QUESTION: {question}

CLAIMS:
{claims}
"""


def get(url, as_json=True):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    return json.loads(data) if as_json else data


def openalex(q):
    j = get("https://api.openalex.org/works?per-page=%d&search=%s" % (PER_QUERY, urllib.parse.quote(q)))
    out = []
    for w in j.get("results", []):
        inv = w.get("abstract_inverted_index") or {}
        pos = sorted((p, word) for word, ps in inv.items() for p in ps)
        out.append({"id": w.get("id", "").rsplit("/", 1)[-1], "title": w.get("title") or "", "year": w.get("publication_year"),
                    "url": w.get("doi") or w.get("id"), "cited_by": w.get("cited_by_count"),
                    "abstract": " ".join(word for _, word in pos), "via": "openalex"})
    return out


def arxiv(q):
    x = get("http://export.arxiv.org/api/query?max_results=%d&search_query=%s" % (PER_QUERY, urllib.parse.quote("all:" + q)), False)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in ET.fromstring(x).findall("a:entry", ns):
        aid = (e.findtext("a:id", "", ns) or "").rsplit("/", 1)[-1]
        out.append({"id": "arXiv:" + aid, "title": " ".join((e.findtext("a:title", "", ns) or "").split()),
                    "year": (e.findtext("a:published", "", ns) or "")[:4], "url": e.findtext("a:id", "", ns),
                    "abstract": " ".join((e.findtext("a:summary", "", ns) or "").split()), "via": "arxiv"})
    return out


def semantic_scholar(q):
    j = get("https://api.semanticscholar.org/graph/v1/paper/search?limit=%d&fields=title,abstract,year,url,citationCount&query=%s"
            % (PER_QUERY, urllib.parse.quote(q)))
    return [{"id": "S2:" + p["paperId"], "title": p.get("title") or "", "year": p.get("year"), "url": p.get("url"),
             "cited_by": p.get("citationCount"), "abstract": p.get("abstract") or "", "via": "semanticscholar"}
            for p in j.get("data", [])]


def search(item_path, out):
    item = json.load(open(item_path))["item"]
    r = item["research"]
    os.makedirs(out, exist_ok=True)
    log, seen, sources = [], set(), []
    for q in r["queries"]:
        for name, f in (("openalex", openalex), ("arxiv", arxiv), ("semanticscholar", semantic_scholar)):
            t0 = time.time()
            try:
                got = f(q)
                log.append({"api": name, "query": q, "n": len(got), "s": round(time.time() - t0, 1)})
            except Exception as e:  # an API down or rate-limited is logged, not fatal
                got = []
                log.append({"api": name, "query": q, "error": f"{type(e).__name__}: {e}"[:200]})
            for s in got:
                key = re.sub(r"\W+", "", s["title"].lower())[:80]
                if s["abstract"] and key and key not in seen:
                    seen.add(key)
                    sources.append(s)
    sources.sort(key=lambda s: -(s.get("cited_by") or 0))
    sources = sources[:MAX_SOURCES]
    for i, s in enumerate(sources):
        s["sid"] = f"S{i + 1}"
    json.dump({"item": item, "sources": sources, "log": log}, open(os.path.join(out, "search.json"), "w"), indent=1)
    print(f"{len(sources)} sources with abstracts; " + ", ".join(f"{x['api']}:{x.get('n', 'ERR')}" for x in log))
    return 0


def parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def verified(claims, sources):
    """Script check: the quote occurs verbatim (whitespace-normalised) in the source's abstract, and every number in
    the claim occurs in the quote. Returns (kept, dropped with reasons)."""
    by = {s["sid"]: s for s in sources}
    norm = lambda s: " ".join(str(s).split()).lower()
    kept, dropped = [], []
    for c in claims if isinstance(claims, list) else []:
        if not isinstance(c, dict) or not {"source", "claim", "quote"} <= set(c):
            dropped.append((c, "malformed")); continue
        s = by.get(c["source"])
        if not s:
            dropped.append((c, "unknown source")); continue
        if len(str(c["quote"]).split()) < 5 or norm(c["quote"]) not in norm(s["abstract"]):
            dropped.append((c, "quote not found verbatim in the abstract")); continue
        nums = re.findall(r"\d+(?:\.\d+)?", str(c["claim"]))
        if any(n not in str(c["quote"]) for n in nums):
            dropped.append((c, "a number in the claim is not in the quote")); continue
        kept.append(c)
    return kept, dropped


def read(out):
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "router"))
    import engines
    d = json.load(open(os.path.join(out, "search.json")))
    q = d["item"]["research"]["question"]
    calls = []

    def ask(engine, model, prompt):
        r = (engines.call(engine, model, prompt, max_tokens=6000) if engine == "cloudflare"
             else engines.compat_call(engine, model, prompt, max_tokens=4000))
        calls.append(r)
        return r

    src = "\n\n".join(f"[{s['sid']}] {s['title']} ({s['year']}). {s['abstract'][:1800]}" for s in d["sources"])
    r = ask(*READER, READ_PROMPT.format(question=q, sources=src))
    reading = parse_json(r["text"]) or {"claims": [], "open_questions": [], "error": "reader reply was not JSON"}
    kept, dropped = verified(reading.get("claims"), d["sources"])
    check = None
    if kept:
        lines = "\n".join(f"{i}. [{c['source']}] CLAIM: {c['claim']} QUOTE: \"{c['quote']}\"" for i, c in enumerate(kept))
        c2 = ask(*CHECKER, CHECK_PROMPT.format(question=q, claims=lines))
        check = parse_json(c2["text"]) or {"keep": None, "notes": "checker reply was not JSON: " + str(c2["error"])[:200]}
    json.dump({"reading": reading, "script_kept": kept, "script_dropped": [[c, why] for c, why in dropped], "check": check},
              open(os.path.join(out, "read.json"), "w"), indent=1)
    with open(os.path.join(out, "calls.jsonl"), "w") as f:
        for c in calls:
            f.write(json.dumps({k: c.get(k) for k in ("t", "engine", "model", "ok", "status", "latency_s", "usage", "error")}) + "\n")
    print(f"reader ok={r['ok']} claims={len(reading.get('claims') or [])} script-verified={len(kept)} "
          f"checker kept={None if not check else check.get('keep')}")
    return 0


def card(repo, out):
    d = json.load(open(os.path.join(out, "search.json")))
    rd = json.load(open(os.path.join(out, "read.json")))
    item, by = d["item"], {s["sid"]: s for s in d["sources"]}
    kept, _ = verified(rd["script_kept"], d["sources"])   # re-checked here, with no keys
    keep = (rd.get("check") or {}).get("keep")
    final = [c for i, c in enumerate(kept) if isinstance(keep, list) and i in keep]
    L = [f"# Literature card: {item['id']} — {item['research']['question']}", "",
         "Made by the queue worker (P7 in P8): search by script (OpenAlex, arXiv, Semantic Scholar), first read by "
         f"{READER[1]}, quotes checked verbatim against each abstract by script, claims checked by {CHECKER[1]}. "
         "**Not yet judged by Claude**: abstracts only, so read the source before relying on a claim.", "",
         "Search log: " + "; ".join(f"{x['api']} {x.get('n', 'error')}" for x in d["log"]), ""]
    for kind in ("frontier", "result", "failure mode", "prior art", "method"):
        cs = [c for c in final if c.get("kind") == kind]
        if cs:
            L.append(f"## {kind.capitalize()}")
            for c in cs:
                s = by[c["source"]]
                L.append(f"- {c['claim']} — \"{c['quote']}\" ({s['title']}, {s['year']}; {s['url']})")
            L.append("")
    other = [c for c in final if c.get("kind") not in ("frontier", "result", "failure mode", "prior art", "method")]
    if other:
        L += ["## Other"] + [f"- {c['claim']} — \"{c['quote']}\" ({by[c['source']]['url']})" for c in other] + [""]
    oq = (rd["reading"] or {}).get("open_questions") or []
    if oq:
        L += ["## Open questions (reader's, unverified)"] + [f"- {x}" for x in oq[:8]] + [""]
    L += [f"Dropped by the script check: {len(rd['script_dropped'])}; by the second model: {len(kept) - len(final)}.",
          f"Sources read: {len(d['sources'])}."]
    path = os.path.join(repo, "research", "cards", f"{item['id']}.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("\n".join(L) + "\n")
    res = {"result": "merge" if final else "pr", "why": [] if final else ["no claim survived both checks"],
           "claims": len(final), "path": f"research/cards/{item['id']}.md"}
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit({"search": lambda: search(a[1], a[2]), "read": lambda: read(a[1]),
              "card": lambda: card(a[1], a[2])}[a[0]]())
