"""Which free model can read for the research worker when Cloudflare's window is full? (Owner 2026-11-20: free first;
spend only where no free model does the job.) Same saved bundles (search.json), same prompt, same script check and
decision model; only the reader differs.

python3 worker/reader_bake.py OUT_DIR BUNDLE_DIR[,BUNDLE_DIR...] POOL,MODEL [POOL,MODEL ...]

Rule fixed before the run (queue P32, reader fallback): a reader qualifies if, over the bundles, it fails no call,
its replies parse every time, and its decision-kept claims are at least 80% of the baseline's (the first POOL,MODEL
given). Among those that qualify, the free one with the most kept claims is the fallback; ties go to the cheaper pool
(the one with more headroom in the ledger)."""
import json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import research_job as rj


def main(out, bundles, readers):
    rj.CHECK_CALLS = 0   # no Nemotron shadow votes: the decision model decides, as live; saves 6 NIM calls per read
    os.makedirs(out, exist_ok=True)
    rows = {}
    for spec in readers:
        rj.READER = tuple(spec.split(",", 1))
        os.environ["READER_OVERRIDE"] = spec   # GLM F033: research_job honours rj.READER only with this set; without it every
                                               # row read through the default chain and was labelled with the wrong reader
        tot = {"bundles": 0, "reader_fail": 0, "claims": 0, "script_kept": 0, "decider_kept": 0, "model_fail": []}
        for b in bundles:
            name = os.path.basename(os.path.dirname(b.rstrip("/"))) + "-" + os.path.basename(b.rstrip("/"))
            d = os.path.join(out, spec.replace("/", "_").replace(",", "__").replace(":", "_"), name)
            os.makedirs(d, exist_ok=True)
            shutil.copy(os.path.join(b, "search.json"), d)
            rj.read(d)
            r = json.load(open(os.path.join(d, "read.json")))
            with open(os.path.join(d, "calls.jsonl")) as f, open(os.path.join(out, "usage.jsonl"), "a") as g:
                g.write(f.read())
            chk = r.get("check") or {}
            tot["bundles"] += 1
            tot["reader_fail"] += any("reader" in str(m) for m in r.get("model_fail") or [])
            tot["claims"] += len((r.get("reading") or {}).get("claims") or [])
            tot["script_kept"] += len(r.get("script_kept") or [])
            tot["decider_kept"] += len(chk.get("keep") or []) if chk.get("keep") is not None else 0
            tot["model_fail"] += [f"{name}: {m}" for m in r.get("model_fail") or []]
        rows[spec] = tot
        print(spec, json.dumps(tot))
    base = rows[readers[0]]["decider_kept"]
    for spec, t in rows.items():
        t["qualifies"] = t["reader_fail"] == 0 and t["decider_kept"] >= 0.8 * base
    json.dump({"baseline": readers[0], "rule": __doc__.split("Rule fixed before the run")[1].strip(), "readers": rows},
              open(os.path.join(out, "summary.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2].split(","), sys.argv[3:]))
