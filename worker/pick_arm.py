"""Does a paper-heavy source pick raise surviving claims? (2026-11-20, following the data.)

Data that raised it: over the 16 cards on main the reader read 385 sources and kept 34 claims: 32 from papers, 2 from
preprints, 0 from the 193 forum, Q&A, encyclopedia and code sources. The search pools hold 35-65 papers per question
(rerank arm 44712474121), but the live pick takes 12 papers and fills 18 places with the other kinds.

python3 -I worker/pick_arm.py SAVED_ARM_DIR OUT_DIR ID   (one item; reads its saved cite/search.json pool)

Two arms from the same saved pool, same reader, checker and decision model; only the per-kind counts differ:
  live   PER_KIND as on main (papers 12, encyclopedia 3, Q&A 5, forum 6, code 4)
  papers PAPERS (papers 24, encyclopedia 2, Q&A 2, forum 1, code 1): the same 30 places
Rule fixed before the run: adopt if, over the 6 items, the papers arm keeps at least 3 more decision-kept claims AND
ends in "no claims" on no more items than the live arm. Otherwise keep the live pick.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import research_job as rj
from rerank_arm import summarise

PAPERS = {"papers": 24, "encyclopedia": 2, "Q&A": 2, "forum": 1, "code": 1}


def main(saved, out, iid):
    s = json.load(open(os.path.join(saved, iid, "cite", "search.json")))
    kinds = [k for k in s["item"]["research"].get("kinds", list(rj.APIS)) if k in rj.APIS]
    row = {"id": iid, "pool": len(s["pool"])}
    for arm, per in (("live", None), ("papers", PAPERS)):
        d = os.path.join(out, iid, arm)
        os.makedirs(d, exist_ok=True)
        json.dump(dict(s, sources=rj.pick(s["pool"], kinds, per_kind=per)), open(os.path.join(d, "search.json"), "w"), indent=1)
        rj.read(d)
        row[arm] = summarise(d)
        row[arm]["sources"] = len(json.load(open(os.path.join(d, "search.json")))["sources"])
        with open(os.path.join(d, "calls.jsonl")) as f, open(os.path.join(out, "usage.jsonl"), "a") as g:
            g.write(f.read())
    json.dump(row, open(os.path.join(out, f"{iid}.json"), "w"), indent=1)
    print(json.dumps(row))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
