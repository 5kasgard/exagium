"""P5b decision by PLAN.md's fixed rules. python3 -I decide.py CONFIRM.json LEADERBOARD.md > DECISION.md"""
import json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from judge import paired  # noqa: E402

PROPS = ["prop-44492370928-2", "prop-44587812936-1", "prop-44610648477-1", "prop-44549931886-1", "prop-44601956368-1"]
BASE = ["label-gbdt", "label-mlp16", "label-linear"]


def dev_scores(board):
    out = {}
    for line in open(board):
        c = [x.strip() for x in line.split("|")]
        if len(c) > 7 and c[1].startswith("prop-"):
            try:
                out[c[1]] = float(c[6].split("±")[0])
            except ValueError:   # unscored rows are normal on the board; one of the five decided here is an error (GLM F063)
                if c[1] in PROPS:
                    raise SystemExit(f"{c[1]}: unreadable dev score on the board: {c[6]!r}")
    return out


def main(conf_path, board):
    r = json.load(open(conf_path))["candidates"]
    dev = dev_scores(board)
    DEV_BASE = {"label-gbdt": 0.9791, "label-mlp16": 0.9770, "label-linear": 0.9613}   # runner/results/2026-11-19-p3-step1
    L = ["# P5b decision (rules in PLAN.md, fixed before the run)", "",
         "| candidate | confirmation acc | SE | Ir/item | ns/item | vs gbdt | vs mlp16 | vs linear | rule 1 | rule 2 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    held, front = 0, 0
    for p in PROPS + BASE:
        c = r.get(p)
        if not c or "items" not in c:
            L.append(f"| {p} | missing | | | | | | | | |"); continue
        diffs = {b: paired(c["items"], r[b]["items"]) for b in BASE if b != p and b in r}
        cells = [f"{m:+.4f} ± {s:.4f}" for m, s in (diffs.get(b, (0, 0)) for b in BASE)]
        r1 = r2 = ""
        if p in PROPS and any(b not in diffs for b in BASE):   # GLM F061: all() over a partial set held vacuously
            r1 = r2 = "undecidable (a baseline is missing)"
        elif p in PROPS:
            ok1 = all(m > 2 * s for m, s in diffs.values())
            # diffs[b] = (proposal - baseline, SE). PLAN.md rule 2 as registered: dominated if a baseline is at least as
            # accurate within 1 paired SE and cheaper in Ir/item. (A second clause, "more accurate and no dearer", was not
            # registered and is gone, GLM F062; on the recorded run it changed no outcome.)
            dominated = any(diffs[b][0] <= diffs[b][1] and r[b]["ir_per_item"] < c["ir_per_item"] for b in BASE)
            ok2 = not dominated
            held += ok1; front += ok2
            r1, r2 = ("holds" if ok1 else "no"), ("advance" if ok2 else "dominated")
        L.append(f"| {p} | {c['mean']:.4f} | {c['se']:.4f} | {c.get('ir_per_item')} | {c.get('ns_per_item')} | " + " | ".join(cells) + f" | {r1} | {r2} |")
    drops_p = [dev[p] - r[p]["mean"] for p in PROPS if p in dev and p in r]
    drops_b = [DEV_BASE[b] - r[b]["mean"] for b in BASE if b in r]
    if not drops_p:   # GLM F063: no proposal with both a dev and a confirmation score makes rule 3 invalid, not NaN
        raise SystemExit("rule 3 invalid: no proposal has both a dev score and a confirmation score")
    mp = sum(drops_p) / len(drops_p)
    mb = sum(drops_b) / len(drops_b) if drops_b else float("nan")
    L += ["", f"Rule 1 (above every family): **{held} of {len(PROPS)}** hold.",
          f"Rule 2 (accuracy-cost front, no baseline as accurate within 1 SE and cheaper): **{front} of {len(PROPS)}** advance.",
          f"Rule 3 (winner's curse): mean drop dev -> confirmation, proposals {mp:+.4f}, baselines {mb:+.4f}; excess {mp - mb:+.4f}."]
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
