"""Can a decision model replace the noisy research checker? (queue P31; the Nemotron checker kept [], [0,1,2], [0,1] on the
same claims.) One request per file: the file (question and numbered claims with quotes) is the state, and each claim gets
one 'noul' question; keep if p >= 0.5. Known answers fixed in router/review/checker-relevance/PLAN.md by the Claude shift.
python3 -I router/decisions_checker_probe.py MODEL OUT_JSON FILE:EXPECT [FILE:EXPECT ...]   (EXPECT like KKD)"""
import json, os, re, sys, time, urllib.request

URL = "https://openrouter.ai/api/alpha/decisions"
Q = ("Keep claim {i}? Keep only if its quote supports it, the claim says exactly what the quote says and no more (no added "
     "interpretation, comparison or conclusion), and it directly answers the QUESTION, not merely the same field.")


def main(model, out, *specs):
    key = os.environ["OPENROUTER_API_KEY"]
    res = []
    for spec in specs:
        path, expect = spec.rsplit(":", 1)
        text = open(path).read()
        n = len(re.findall(r"^\d+\. \[", text, re.M))
        assert n == len(expect), (path, n, expect)
        body = {"model": model, "state": text.split("CLAIMS:", 1)[0].split("\n\n", 1)[-1] + "CLAIMS:" + text.split("CLAIMS:", 1)[1],
                "questions": {f"c{i}": {"type": "noul", "instructions": Q.format(i=i)} for i in range(n)}}
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        t = time.time()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp, st = json.loads(r.read()), r.status
        except urllib.error.HTTPError as e:
            resp, st = {"error": e.read().decode(errors="replace")[:300]}, e.code
        lat = round(time.time() - t, 2)
        ans = resp.get("answers", {}) if st == 200 else {}
        for i in range(n):
            p = (ans.get(f"c{i}") or {}).get("noul")
            got = None if p is None else ("K" if p >= 0.5 else "D")
            res.append({"file": os.path.basename(path), "claim": i, "expect": expect[i], "p_keep": p, "got": got, "status": st, "latency_s": lat})
            print(os.path.basename(path), i, expect[i], got, p, st, lat)
        time.sleep(3.2)
    right = sum(r["got"] == r["expect"] for r in res)
    print(f"{model}: {right}/{len(res)} as expected")
    json.dump({"model": model, "right": right, "n": len(res), "results": res}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
