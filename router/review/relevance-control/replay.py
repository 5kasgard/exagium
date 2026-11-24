"""Replay the research relevance filter's request on frozen pools with old, new and unrelated controls (PLAN.md).
python3 -I router/review/relevance-control/replay.py POOLS_JSON OUT_JSON [--dry] [--prod]   (needs OPENROUTER_API_KEY unless --dry)"""
import json, os, sys, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "worker"))
import research_job as RJ  # noqa: E402  the production model, instruction and unrelated control

ASK = "Is source {i} about the QUESTION itself (not merely the same broad field), so that it could answer part of it?"
NEW = "{q} An empirical study that answers this question directly: we measure it and report the answer, with numbers and methods."


def body(q, batch, prod=False):
    """prod: exactly production's layout [YES, unrelated, sources] (review: the first replay added the bare question first)."""
    head = [NEW.format(q=q)] if prod else [q, NEW.format(q=q)]
    docs = head + list(RJ.REL_CONTROL_NO) + [f"{x['title']}. {x['abstract']}" for x in batch]
    return {"model": RJ.DECIDER, "state": f"QUESTION: {q}\n\nSOURCES:\n" + "\n".join(f"{i}. {t}" for i, t in enumerate(docs)),
            "questions": {f"s{i}": {"type": "noul", "instructions": ASK.format(i=i)} for i in range(len(docs))}}, len(docs)


def judge(rows):
    """PLAN.md rule, pure: every row has new >= 0.5 and unrelated < 0.5 (a failed request is a miss)."""
    return bool(rows) and all(r.get("new") is not None and r["new"] >= 0.5 and r["unrelated"] < 0.5 for r in rows)


def main(pools, out, dry=False, prod=False):
    rows = []
    for p in json.load(open(pools))["pools"]:
        b, n = body(p["question"], p["batch1"], prod)
        if prod:   # the production request: compare with research_job's own builder
            assert b["state"].split("SOURCES:\n")[1].splitlines()[0] == "0. " + RJ.REL_CONTROL_YES.format(q=p["question"])
        if dry:
            rows.append({"item": p["item"], "sources": n, "state_chars": len(b["state"])}); continue
        req = urllib.request.Request("https://openrouter.ai/api/alpha/decisions", data=json.dumps(b).encode(), method="POST",
                                     headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp, st = json.loads(r.read()), r.status
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            resp, st = {"error": str(e)[:200]}, getattr(e, "code", 0)
        ps = [((resp.get("answers") or {}).get(f"s{i}") or {}).get("noul") for i in range(n)] if st == 200 else []
        row = {"item": p["item"], "filter_failed": p["filter_failed"], "status": st}
        if st != 200 or any(not isinstance(x, (int, float)) for x in ps):
            row.update(new=None, error=str(resp.get("error", "missing answers"))[:200])
        else:
            o = 0 if prod else 1
            row.update(old=None if prod else round(ps[0], 3), new=round(ps[o], 3), unrelated=round(ps[o + 1], 3),
                       kept=sum(x >= 0.5 for x in ps[o + 2:]), of=len(ps) - o - 2, usage=resp.get("usage"))
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("item", "filter_failed", "status", "old", "new", "unrelated", "kept", "error")}))
        time.sleep(2)
    res = {"dry": dry, "prod_layout": prod, "rows": rows, "rule_met": None if dry else judge(rows)}
    json.dump(res, open(out, "w"), indent=1)
    if not dry:
        print(f"rule (new >= 0.5 and unrelated < 0.5 on all {len(rows)}): {'MET' if res['rule_met'] else 'NOT MET'}")
    return 0


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x not in ("--dry", "--prod")]
    sys.exit(main(a[0], a[1], "--dry" in sys.argv, "--prod" in sys.argv))
