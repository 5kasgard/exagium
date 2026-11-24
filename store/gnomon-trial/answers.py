"""P25 step 1: the 10 fixed questions' answers, computed from the record before Gnomon or grep is run on them.
Usage (repo root): python3 -I store/gnomon-trial/answers.py > store/gnomon-trial/answers.json
Reads only committed result files and cards; no network, no model. Deterministic."""
import glob, json, math, os, re, subprocess, sys

P3 = "runner/results/2026-11-19-p3-step1/ci/dev-{}.json"
P2 = "runner/results/2026-11-18/ci/dev-sat.json"
TR = "transfer/results/2026-11-18/main/decision.json"
RUN = "44486290814"


def cands(task):
    return json.load(open(P3.format(task)))["candidates"]


def acc(c):
    return float(c["mean"])


def paired(a, b):   # same formula as runner/results/2026-11-19-p3-step1/decide.py paired()
    d = [x["score"] - y["score"] for x, y in zip(a["items"], b["items"])]
    n = len(d); m = sum(d) / n
    return m, math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)


def r4(x):
    return round(x, 4)


def main():
    a = {}
    c = cands("text"); base = c["label-linear"]
    a["Q1"] = sorted([[k, r4(paired(v, base)[0]), r4(paired(v, base)[1])] for k, v in c.items()
                      if k != "label-linear" and acc(v) > acc(base)])
    a["Q2"] = sorted(subprocess.run(["git", "grep", "-l", RUN], capture_output=True, text=True).stdout.split())
    q3, q9 = {}, {}
    for t in ("optdigits", "text", "adult"):
        c = cands(t)
        best = max(("label-linear", "label-mlp16"), key=lambda k: acc(c[k]))
        m, se = paired(c["label-wisard"], c[best])
        q3[t] = {"vs": best, "diff": r4(m), "se": r4(se), "matched": abs(m) <= 2 * se,
                 "ir_wisard": round(float(c["label-wisard"]["ir_per_item"])), "ir_vs": round(float(c[best]["ir_per_item"]))}
        g, gse = paired(c["label-gbdt"], c[best])
        q9[t] = g > 2 * gse
    a["Q3"] = q3
    tr = json.load(open(TR))
    flips = tr["flips"]
    a["Q4"] = sorted({tuple(f["pair"]) for f in flips if f["measure"] in ("acc", "accuracy")})
    a["Q5"] = {m: sum(1 for f in flips if f["measure"] == m) for m in sorted({f["measure"] for f in flips})}
    sat = json.load(open(P2))["candidates"]
    a["Q6"] = {k: sum(int(i["score"]) for i in v["items"]) for k, v in sorted(sat.items())}
    q7 = {}
    for t in ("optdigits", "text", "adult"):
        c = {k: v for k, v in cands(t).items() if k != "majority"}
        top = max(c, key=lambda k: acc(c[k]))
        near = [k for k in c if k == top or acc(c[top]) - acc(c[k]) <= float(c[top]["se"])]
        q7[t] = min(near, key=lambda k: float(c[k]["ir_per_item"]))
    a["Q7"] = q7
    cards = {}
    for n in (1, 2, 3, 4, 5, 6, 7, 8):
        p = f"research/cards/R{n}.md"
        if os.path.exists(p):
            cards[f"R{n}"] = {h.lower(): int(k) for h, k in re.findall(r"^## ([A-Za-z ]+) \((\d+)\)", open(p).read(), re.M)}
        else:
            cards[f"R{n}"] = None
    a["Q8"] = cards
    a["Q9"] = sorted(t for t, v in q9.items() if v)
    a["Q10"] = sorted(k for k, v in cards.items() if v is not None and not v.get("measured"))
    json.dump(a, sys.stdout, indent=1, sort_keys=True, default=list); print()


if __name__ == "__main__":
    main()
