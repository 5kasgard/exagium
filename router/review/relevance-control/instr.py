"""Score relevance-filter instructions against labelled pools (PLAN-instruction.md).
python3 -I router/review/relevance-control/instr.py OUT_JSON [--dry]"""
import json, os, sys, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "worker"))
import research_job as RJ  # noqa: E402

ASKS = {"A": "Is source {i} about the QUESTION itself (not merely the same broad field), so that it could answer part of it?",
        "B": "Could source {i} give evidence, a method, a result or a counterexample that bears directly on some part of the QUESTION? Being in the same broad field is not enough."}
THRESHOLDS = (0.5, 0.3)


def body(q, batch, ask):
    docs = [RJ.REL_CONTROL_YES.format(q=q)] + list(RJ.REL_CONTROL_NO) + [f"{x['title']}. {x['abstract']}" for x in batch]
    return {"model": RJ.DECIDER, "state": f"QUESTION: {q}\n\nSOURCES:\n" + "\n".join(f"{i}. {t}" for i, t in enumerate(docs)),
            "questions": {f"s{i}": {"type": "noul", "instructions": ask.format(i=i)} for i in range(len(docs))}}, len(docs)


def score(results, labels):
    """results: {variant: {item: [p0, p1, p_src...] or None}} -> {(variant, t): {...}}. Pure."""
    out = {}
    for v, per in results.items():
        for t in THRESHOLDS:
            ok = all(ps is not None for ps in per.values())
            ctrl = ok and all(ps[0] >= t and ps[1] < t for ps in per.values())
            ky = ay = kl = 0
            for item, ps in per.items():
                if ps is None:
                    continue
                lab, src = labels[item], ps[2:]
                ay += len(lab["yes"]); ky += sum(src[i] >= t for i in lab["yes"])
                kl += sum(src[i] >= t for i in lab["yes"] + lab["no"])
            out[f"{v}@{t}"] = {"requests_ok": ok, "controls": ctrl, "recall": round(ky / ay, 3) if ay else None,
                               "precision": round(ky / kl, 3) if kl else None, "kept_yes": ky, "all_yes": ay, "kept_labelled": kl}
    return out


def decide(sc):
    """PLAN rule, pure -> chosen key or None."""
    base = sc.get("A@0.5", {}).get("recall") or 0.0
    ok = [(v["recall"], k) for k, v in sc.items() if v["requests_ok"] and v["controls"] and v["precision"] is not None
          and v["precision"] >= 0.7 and v["recall"] is not None]
    if not ok:
        return None
    r, k = max(ok)
    return k if r >= base + 0.15 else None


def main(out, dry=False):
    pools = {p["item"]: p for p in json.load(open(os.path.join(HERE, "pools.json")))["pools"]}
    labels = json.load(open(os.path.join(HERE, "labels.json")))["labels"]
    results = {v: {} for v in ASKS}
    for v, ask in ASKS.items():
        for item in labels:
            b, n = body(pools[item]["question"], pools[item]["batch1"], ask)
            if dry:
                results[v][item] = None; continue
            req = urllib.request.Request("https://openrouter.ai/api/alpha/decisions", data=json.dumps(b).encode(), method="POST",
                                         headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    resp, st = json.loads(r.read()), r.status
            except (urllib.error.URLError, TimeoutError, ValueError) as e:
                resp, st = {"error": str(e)[:200]}, getattr(e, "code", 0)
            ps = [((resp.get("answers") or {}).get(f"s{i}") or {}).get("noul") for i in range(n)] if st == 200 else []
            results[v][item] = [round(p, 4) for p in ps] if st == 200 and all(isinstance(p, (int, float)) for p in ps) else None
            print(v, item, st, "ok" if results[v][item] else resp.get("error", "missing answers"))
            time.sleep(2)
    sc = None if dry else score(results, labels)
    res = {"dry": dry, "results": results, "scores": sc, "chosen": None if dry else decide(sc)}
    json.dump(res, open(out, "w"), indent=1)
    if not dry:
        for k, v in sc.items():
            print(k, v)
        print("chosen:", res["chosen"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], "--dry" in sys.argv))
