"""Score an OpenRouter Decisions model (e.g. perplexity/pplx-decider-v1.3:free) on the known-answer grading set (queue P31).
The Decisions API returns probabilities for typed questions instead of text. Each grading item becomes one 'noul' question
against the item's own text as state; ACCEPTABLE if p(yes) >= 0.5. Sends only the prompt text; runs no returned code.
python3 -I router/decisions_probe.py MODEL PROMPT_MD OUT_JSON [per-item|batch]
batch: one request per file, the whole file as state and one 'noul' question per item (the API takes up to 100 questions
sharing one state), so a 35-item set costs 2 requests instead of 35 under the free tier's 50 a day."""
import json, os, re, sys, time, urllib.request

URL = "https://openrouter.ai/api/alpha/decisions"
HEAD = ("Judge only against the item's stated acceptable outcome and required delivery. Not acceptable: a change beyond what was "
        "asked, a changed meaning, an added link, mention or HTML, or a delivery that does not match the required delivery. For a "
        "card item, a card with no claim line is not acceptable.")


def items(md):
    parts = re.split(r"^### (I\d\d)\n", open(md).read(), flags=re.M)
    return [(parts[k], parts[k + 1].strip()) for k in range(1, len(parts) - 1, 2)]


def ask(model, state, key):
    body = {"model": model, "state": state,
            "questions": {"acceptable": {"type": "noul",
                                         "instructions": "Would a careful maintainer accept this item's change as it is? " + HEAD}}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read()), r.status, round(time.time() - t, 2)
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode(errors="replace")[:400]}, e.code, round(time.time() - t, 2)


def p_yes(resp):
    """Find the probability of yes for 'acceptable' wherever the response puts it (alpha API; shape not fixed in our record)."""
    def walk(x):
        if isinstance(x, dict):
            for k in ("noul", "p_yes", "probability", "yes", "p"):   # "noul" is what Decider V1.3 returns (run 44640108993)
                if isinstance(x.get(k), (int, float)):
                    return float(x[k])
            for v in x.values():
                r = walk(v)
                if r is not None:
                    return r
        if isinstance(x, list):
            for v in x:
                r = walk(v)
                if r is not None:
                    return r
        return None
    a = resp.get("answers", resp)
    return walk(a.get("acceptable", a) if isinstance(a, dict) else a)


def ask_batch(model, md, key):
    its = items(md)
    body = {"model": model, "state": open(md).read(),
            "questions": {iid: {"type": "noul", "instructions": f"Is item {iid}'s change acceptable as it is? " + HEAD} for iid, _ in its}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read()), r.status, round(time.time() - t, 2), its
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode(errors="replace")[:400]}, e.code, round(time.time() - t, 2), its


def main(model, md, out, mode="per-item"):
    key = os.environ["OPENROUTER_API_KEY"]
    res = []
    if mode == "batch":
        resp, st, lat, its = ask_batch(model, md, key)
        ans = resp.get("answers", {}) if st == 200 else {}
        for iid, _ in its:
            a = ans.get(iid)
            p = a.get("noul") if isinstance(a, dict) else None
            res.append({"id": iid, "status": st, "latency_s": lat, "p_yes": p,
                        "verdict": None if p is None else ("ACCEPTABLE" if p >= 0.5 else "NOT"),
                        "raw": json.dumps(a)[:300]})
        print(model, md, st, lat, sum(r["p_yes"] is not None for r in res), "answers")
        json.dump({"model": model, "prompt": md, "mode": mode, "usage": resp.get("usage"), "results": res}, open(out, "w"), indent=1)
        return
    for iid, text in items(md):
        resp, st, lat = ask(model, text, key)
        p = p_yes(resp) if st == 200 else None
        res.append({"id": iid, "status": st, "latency_s": lat, "p_yes": p,
                    "verdict": None if p is None else ("ACCEPTABLE" if p >= 0.5 else "NOT"),
                    "raw": json.dumps(resp)[:600]})
        print(iid, st, lat, p)
        time.sleep(3.2)   # under OpenRouter's 20 requests a minute for free models
    json.dump({"model": model, "prompt": md, "results": res}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:5])
