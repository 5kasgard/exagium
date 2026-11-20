"""P32, first test: does a reranker choosing the reader's sources raise surviving claims? A control arm on identical items.

python3 worker/rerank_arm.py QUEUE_JSON OUT_DIR ID [ID ...]

Per item: one search (one candidate pool), then two arms read from that same pool and differ only in how the reader's
sources are picked: "cite" (live pipeline: most-cited per kind) and "rerank" (reranker relevance to the question, same
counts per kind). Both arms use the same reader, checker and decision model. Writes OUT_DIR/<ID>/{cite,rerank}/ and
OUT_DIR/summary.json; all calls go to OUT_DIR/usage.jsonl for the shared ledger.

Adoption rule fixed before the run (queue P32): the rerank arm has at least 3 more surviving claims (kept by the
decision model) over 6 cards, and none of its extra claims is rejected in the next tier's check (a later blind review).
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import research_job as rj


def summarise(arm_dir):
    r = json.load(open(os.path.join(arm_dir, "read.json")))
    chk = r.get("check") or {}
    shadow = chk.get("shadow_nemotron") or {}
    return {"reader_claims": len((r.get("reading") or {}).get("claims") or []), "script_kept": len(r.get("script_kept") or []),
            "decider_kept": None if chk.get("keep") is None else len(chk["keep"]),
            "nemotron_kept": None if shadow.get("keep") is None else len(shadow["keep"]),
            "model_fail": r.get("model_fail") or []}


def main(queue_path, out, ids):
    q = json.load(open(queue_path))
    items = {i["id"]: i for i in (q["items"] if isinstance(q, dict) else q)}
    os.makedirs(out, exist_ok=True)
    rows = []
    for iid in ids:
        it = items.get(iid)
        if not it or not it.get("research"):
            rows.append({"id": iid, "error": "no such research item"}); continue
        d0 = os.path.join(out, iid)
        os.makedirs(d0, exist_ok=True)
        json.dump({"item": it}, open(os.path.join(d0, "item.json"), "w"))
        cite = os.path.join(d0, "cite")
        rj.search(os.path.join(d0, "item.json"), cite)
        s = json.load(open(os.path.join(cite, "search.json")))
        kinds = [k for k in it["research"].get("kinds", list(rj.APIS)) if k in rj.APIS]
        calls = []
        scores, err = rj.rerank_scores(it["research"]["question"], s["pool"], calls)
        with open(os.path.join(out, "usage.jsonl"), "a") as f:
            for c in calls:
                f.write(json.dumps(c) + "\n")
        row = {"id": iid, "pool": len(s["pool"])}
        if scores is None:
            row["error"] = "rerank failed: " + err
            rows.append(row); continue
        rr = os.path.join(d0, "rerank")
        os.makedirs(rr, exist_ok=True)
        picked = rj.pick(s["pool"], kinds, scores)
        json.dump(dict(s, sources=picked), open(os.path.join(rr, "search.json"), "w"), indent=1)
        key = lambda x: x["title"] + x["abstract"][:80]
        row["sources"] = len(s["sources"])
        row["overlap"] = len({key(x) for x in s["sources"]} & {key(x) for x in picked})
        for arm, path in (("cite", cite), ("rerank", rr)):
            rj.read(path)
            row[arm] = summarise(path)
            with open(os.path.join(path, "calls.jsonl")) as f, open(os.path.join(out, "usage.jsonl"), "a") as g:
                g.write(f.read())
        rows.append(row)
        print(json.dumps(row))
    ok = [r for r in rows if "cite" in r and "rerank" in r]
    tot = lambda arm, k: sum((r[arm][k] or 0) for r in ok)
    summary = {"items": rows, "cards_compared": len(ok),
               "decider_kept": {"cite": tot("cite", "decider_kept"), "rerank": tot("rerank", "decider_kept")},
               "nemotron_kept": {"cite": tot("cite", "nemotron_kept"), "rerank": tot("rerank", "nemotron_kept")},
               "script_kept": {"cite": tot("cite", "script_kept"), "rerank": tot("rerank", "script_kept")},
               "rule": "adopt only if rerank keeps >= 3 more decider-kept claims over 6 cards and a blind review rejects none of the extra"}
    summary["first_gate_passed"] = len(ok) >= 6 and summary["decider_kept"]["rerank"] - summary["decider_kept"]["cite"] >= 3
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "items"}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3:]))
