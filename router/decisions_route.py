"""Route known cases with a Decisions API 'choice' question (router/review/tier-routing/PLAN.md). Prompt text only.
python3 -I router/decisions_route.py MODEL CASES_JSON OUT_JSON"""
import json, os, sys, time, urllib.request, urllib.error

URL = "https://openrouter.ai/api/alpha/decisions"
ASK = "Which move should the tier above the free workers make for this item?"
# 'choice' takes criteria: a record of option -> description (the API refused a plain options list, run 44666907315)
CRITERIA = {"confirm": "the worker's outcome is right as it stands: no change was needed, or a refused edit was rightly refused",
            "redo": "send it back to the free workers, because a different approach could succeed (for example, research that found no surviving claim)",
            "escalate": "only the owner can decide: the wording of a guarantee, keys, settings, money"}


def main(model, cases, out):
    key = os.environ["OPENROUTER_API_KEY"]
    res = []
    for c in json.load(open(cases)):
        body = {"model": model, "state": c["state"],
                "questions": {"move": {"type": "choice", "instructions": ASK, "criteria": CRITERIA}}}
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        t = time.time()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp, st = json.loads(r.read()), r.status
        except urllib.error.HTTPError as e:
            resp, st = {"error": e.read().decode(errors="replace")[:400]}, e.code
        res.append({"id": c["id"], "expect": c["expect"], "status": st, "latency_s": round(time.time() - t, 2), "raw": resp})
        print(c["id"], c["expect"], st, json.dumps(resp)[:300])
        time.sleep(3.2)
    json.dump({"model": model, "results": res}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
