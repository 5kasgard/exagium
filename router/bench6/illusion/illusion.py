"""Queue P28: SWE-Bench Illusion probe (arXiv 2506.12286). Is the fresh SWE-rebench set less memorised than SWE-bench Verified?
The free model sees only the repository name and the issue text and names the file(s) to change; a script grades exact
path matches against each task's own patch. Rules: PLAN.md (fixed before any real call).

python3 -I illusion.py sample OUT.json            # fetch both pools (HF datasets-server), seeded sample of 40 + 40
python3 -I illusion.py ask SAMPLE.json OUT_DIR [--dry]   # one call per task (Nemotron 3 Ultra, NIM); --dry: known answers
python3 -I illusion.py grade SAMPLE.json OUT_DIR  # RESULT.json + RESULT.md; exit 0 always (the verdict is in the files)
--dry makes no call: even-numbered tasks get their first gold path, odd ones a wrong path, so grading must read exactly
50% in each set (a known-answer control for the path before calls are spent)."""
import concurrent.futures as cf, json, math, os, random, re, sys, time, urllib.parse, urllib.request

FRESH = ("nebius/SWE-rebench-leaderboard", ["2025_10", "2025_11", "2025_12", "2026_01", "2026_02", "2026_03"])
VERIFIED = ("princeton-nlp/SWE-bench_Verified", ["test"])
N, SEED, MAX_TOKENS, WORKERS = 40, 20261120, 16000, 4
ISSUE_CHARS = 8000   # PLAN.md: longest Verified issue in the sample is 4,361 chars; 2 fresh ones run past 8,000
ENGINE, MODEL = "nvidia", "nvidia/nemotron-3-ultra-550b-a55b"
API = "https://datasets-server.huggingface.co/rows"


def rows(dataset, split):
    out, off = [], 0
    while True:
        q = urllib.parse.urlencode({"dataset": dataset, "config": "default", "split": split, "offset": off, "length": 100})
        for t in range(4):
            try:
                j = json.load(urllib.request.urlopen(API + "?" + q, timeout=60)); break
            except Exception:
                if t == 3: raise
                time.sleep(5 * (t + 1))
        got = [r["row"] for r in j["rows"]]
        out += got; off += len(got)
        if not got or off >= j.get("num_rows_total", off):
            return out


def gold(patch):
    files = []
    for a, b in re.findall(r"^diff --git a/(\S+) b/(\S+)", patch, re.M):
        files.append(b if b != "/dev/null" else a)
    return sorted(set(files))


def pick(pool, n, seed):
    pool = sorted(pool, key=lambda r: r["instance_id"])
    return random.Random(seed).sample(pool, n)


def sample(out):
    res = {}
    for name, (ds, splits) in (("fresh", FRESH), ("verified", VERIFIED)):
        pool = [dict(r, _split=s) for s in splits for r in rows(ds, s)]
        chosen = pick(pool, N, SEED)
        res[name] = {"dataset": ds, "splits": splits, "pool": len(pool),
                     "tasks": [{"id": r["instance_id"], "repo": r["repo"], "split": r["_split"], "created_at": r.get("created_at"),
                                "issue": r["problem_statement"], "gold": gold(r["patch"])} for r in chosen]}
        print(f"{name}: pool {len(pool)}, sampled {len(chosen)}")
    frozen = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen-ids.json")
    if os.path.exists(frozen):   # the sample fixed in PLAN.md; a changed dataset must not change it silently
        want = json.load(open(frozen))
        for name in res:
            got = [t["id"] for t in res[name]["tasks"]]
            if got != want[name]:
                sys.exit(f"{name}: sample differs from frozen-ids.json (dataset changed?); refusing")
    json.dump(res, open(out, "w"), indent=1)


PROMPT = """Below is a GitHub issue from the repository {repo}. You cannot see the repository.
Name the file that most likely has to be changed to fix this issue, as a path from the repository root.
Give up to 3 paths, most likely first, one per line, inside a block that starts with ```paths and ends with ```.

Issue:
{issue}
"""


def norm(p):
    p = p.strip().strip("`'\"*,;: ").strip()
    p = re.sub(r"^(?:[-*]|\d+[.)])\s+", "", p).strip("`'\" ")
    for pre in ("./", "a/", "b/", "/"):
        if p.startswith(pre): p = p[len(pre):]
    return p


def parse(text):
    """Pre-registered (PLAN.md): lines of the last ```paths block; if there is none, path-like tokens anywhere, in order."""
    text = text or ""
    blocks = re.findall(r"```paths\s*\n(.*?)```", text, re.S)
    if blocks:
        cand = [norm(l) for l in blocks[-1].splitlines()]
    else:
        cand = [norm(m) for m in re.findall(r"[\w./-]+\.[A-Za-z0-9]{1,6}\b", text) if "/" in m]
    out = []
    for c in cand:
        if c and " " not in c and c not in out: out.append(c)
    return out[:3]


def ask(sample_path, out, dry=False):
    s = json.load(open(sample_path))
    os.makedirs(out, exist_ok=True)
    jobs = [(name, k, t) for name in ("fresh", "verified") for k, t in enumerate(s[name]["tasks"])]

    def one(job):
        name, k, t = job
        if dry:
            text = "```paths\n" + (t["gold"][0] if k % 2 == 0 else "no/such_file.py") + "\n```"
            return name, t["id"], {"ok": True, "status": 200, "text": text, "finish": "dry"}
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
        import engines
        engines.TIMEOUT = 600   # Nemotron reasons before answering (worker release test 1)
        return name, t["id"], engines.compat_call(ENGINE, MODEL, PROMPT.format(repo=t["repo"], issue=t["issue"][:ISSUE_CHARS]), max_tokens=MAX_TOKENS)

    replies = {}
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        for name, tid, r in ex.map(one, jobs):
            replies.setdefault(name, {})[tid] = {"text": r.get("text"), "ok": r.get("ok"), "status": r.get("status"), "finish": r.get("finish")}
            if not dry:
                with open(os.path.join(out, "calls.jsonl"), "a") as f:
                    f.write(json.dumps({k: r.get(k) for k in ("t", "engine", "model", "ok", "status", "latency_s", "usage")}) + "\n")
            print(f"{name} {tid}: ok {r.get('ok')} status {r.get('status')} finish {r.get('finish')}")
    json.dump(replies, open(os.path.join(out, "replies.json"), "w"), indent=1)


def grade(sample_path, out):
    s, rep = json.load(open(sample_path)), json.load(open(os.path.join(out, "replies.json")))
    res = {}
    for name in ("fresh", "verified"):
        rows_ = []
        for t in s[name]["tasks"]:
            r = rep.get(name, {}).get(t["id"], {})
            pred = parse(r.get("text"))
            rows_.append({"id": t["id"], "gold": t["gold"], "pred": pred, "answered": bool(pred), "finish": r.get("finish"),
                          "top1": bool(pred) and pred[0] in t["gold"], "any3": any(p in t["gold"] for p in pred)})
        n = len(rows_)
        single = [x for x in rows_ if len(x["gold"]) == 1]
        res[name] = {"single_file_n": len(single), "single_file_top1": (sum(x["top1"] for x in single) / len(single)) if single else None,
                     "n": n, "top1": sum(x["top1"] for x in rows_) / n, "any3": sum(x["any3"] for x in rows_) / n,
                     "answered": sum(x["answered"] for x in rows_), "rows": rows_}
    f, v = res["fresh"]["top1"], res["verified"]["top1"]
    se = math.sqrt(f * (1 - f) / res["fresh"]["n"] + v * (1 - v) / res["verified"]["n"])
    gap = v - f
    verdict = "fresh set not shown to be less memorised: treat as contaminated (P28 builds its own tasks)" if gap < 0.15 \
        else "fresh set reads at least 15 points lower: use it"
    res["rule"] = {"gap_top1_points": round(100 * gap, 1), "threshold_points": 15, "se_points": round(100 * se, 1),
                   "ci95_points": [round(100 * (gap - 1.96 * se), 1), round(100 * (gap + 1.96 * se), 1)], "verdict": verdict}
    json.dump(res, open(os.path.join(out, "RESULT.json"), "w"), indent=1)
    md = [f"# P28 Illusion probe", "", "| set | n | top-1 | any of 3 | answered |", "|---|---|---|---|---|"]
    for name in ("fresh", "verified"):
        r = res[name]; md.append(f"| {name} | {r['n']} | {r['top1']:.2f} | {r['any3']:.2f} | {r['answered']} |")
    md += ["", "Secondary (not decisive): single-file tasks only: " + ", ".join(
        f"{n2} {res[n2]['single_file_top1']:.2f} of {res[n2]['single_file_n']}" for n2 in ("fresh", "verified") if res[n2]["single_file_n"])]
    md += ["", f"Verified minus fresh, top-1: {res['rule']['gap_top1_points']} points (SE {res['rule']['se_points']}, 95% "
           f"interval {res['rule']['ci95_points']}); rule: at least 15. Verdict: {verdict}."]
    open(os.path.join(out, "RESULT.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sample": sample(sys.argv[2])
    elif cmd == "ask": ask(sys.argv[2], sys.argv[3], "--dry" in sys.argv[4:])
    elif cmd == "grade": grade(sys.argv[2], sys.argv[3])
    else: sys.exit("usage: sample | ask | grade")
