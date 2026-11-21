"""Judge proposals by PLAN.md's fixed measures, and write the leaderboard (queue P5). No secrets; reads JSON only.
python3 -I judge.py one NAME EXTRACT_STATUS REPRODUCE.json DEV.json OUT.json [TASK]   -> one proposal's verdict
python3 -I judge.py board DIR                                                 -> DIR/LEADERBOARD.md from DIR/*/result.json
Valid: extracted, trained twice to identical weights, and dev accuracy above the floor by at least 3 SE (paired with the
floor's items). Beats a baseline: above label-linear by more than 2 paired SE."""
import glob, hashlib, json, math, os, sys


def paired(a, b):
    """mean and SE of the per-item difference a - b; the two must cover the same items (review 44486820706, 2)."""
    if len(a) != len(b) or not a:
        raise ValueError(f"paired lists differ in length ({len(a)} vs {len(b)})")
    d = [x["score"] - y["score"] for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    return m, sd / math.sqrt(n)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def one(name, extract_status, rep_path, dev_path, task=""):
    v = {"name": name, "task": task, "extracted": extract_status == "extracted", "extract": extract_status,
         "reproduced": False, "valid": False, "beats_linear": False}
    if not v["extracted"]:
        return v
    if os.path.exists(rep_path):
        rep = [r for r in json.load(open(rep_path)) if r.get("candidate") == name]
        v["reproduced"] = bool(rep) and all(r.get("identical") for r in rep)
    if not v["reproduced"] or not os.path.exists(dev_path):
        return v
    dev = json.load(open(dev_path))
    c = dev["candidates"].get(name)
    if not c or "items" not in c:
        return v
    floor = dev["candidates"][dev["floor"]["baseline"]]
    lin = dev["candidates"].get("label-linear")
    try:
        mf, sf = paired(c["items"], floor["items"])
    except ValueError as e:
        v["judge_error"] = str(e)
        return v
    v.update(task=dev["task"], mean=c["mean"], se=c["se"], ir_per_item=c.get("ir_per_item"),
             ns_per_item=c.get("ns_per_item"), over_floor=mf, over_floor_se=sf)
    n = dev.get("n") or len(c["items"])
    if c.get("ir_n") and c.get("ns_n"):   # startup included (loop.py's decision measures; GLM F054): recorded for the decision
        v.update(ir_per_item_with_startup=c["ir_n"] / n, ns_per_item_with_startup=sum(c["ns_n"]) / len(c["ns_n"]) / n)
    v["valid"] = mf >= 3 * sf and mf > 0
    if lin:
        ml, sl = paired(c["items"], lin["items"])
        v.update(over_linear=ml, over_linear_se=sl, beats_linear=v["valid"] and ml > 2 * sl)
    return v


COST = "ir_per_item_with_startup"   # whole-run Ir/item ranks the front (decided by data 2026-11-22, audit F054 / R4):
# on all 165 valid proposals with both counts the front was the same under either and the order almost unchanged
# (Spearman 0.9996 optdigits, 0.99996 text, 1.0 adult; startup adds 143-177 Ir/item on optdigits, 21-28 on text and
# adult), and the marginal count can be gamed by moving work into the N=0 run (loop.py's schema: informational only).
# The marginal count stays on the board, unranked.


def _beaten(p, b, key):
    return b is not p and b["task"] == p["task"] and b["mean"] >= p["mean"] - p["se"] and b[key] < p[key]


def fronts(rs):
    """(front, provisional, rescore) over valid proposals, on their task's accuracy-cost front: no other valid proposal on
    the same task at least as accurate (within the candidate's 1 SE) and cheaper in whole-run Ir/item. confirm/PLAN.md
    rule 2, applied among proposals; beats-linear alone rewarded 30-266x instruction blowups (GLM F101).
    A result scored before the whole-run count was recorded competes with its marginal cost plus the task's measured
    startup floor (the smallest startup cost measured there; the do-nothing baseline's is about the same, so only a
    libc-free program could start cheaper) (review 2026-11-22: leaving such results out put 5 points on the front
    that they beat by more than any startup spread). Then:
      front: has the whole-run count and nothing beats it;
      provisional: has it, and only lower-bound results beat it (a rescore of those decides);
      rescore: has no whole-run count and nothing beats it at that floor."""
    v = [dict(r) for r in rs if r.get("valid") and r.get("ir_per_item") and "mean" in r]
    low = {}
    for r in v:
        if r.get(COST):
            low[r["task"]] = min(low.get(r["task"], float("inf")), r[COST] - r["ir_per_item"])
    v = [dict(r, _c=r[COST] if r.get(COST) else r["ir_per_item"] + max(0.0, low.get(r["task"], 0.0))) for r in v]
    top = {p["name"] for p in v if not any(_beaten(p, b, "_c") for b in v)}
    front = {p["name"] for p in v if p["name"] in top and p.get(COST)}
    rescore = {p["name"] for p in v if p["name"] in top and not p.get(COST)}
    prov = {p["name"] for p in v if p["name"] not in top and p.get(COST)
            and all(not b.get(COST) for b in v if _beaten(p, b, "_c"))}
    return front, prov, rescore


def front(rs):
    return fronts(rs)[0]


def recount(rs):
    """Old results (no whole-run count) on the front at the measured startup floor: listed for a rescore, never dropped."""
    return fronts(rs)[2]


def board(d):
    paths = sorted(glob.glob(os.path.join(d, "*", "result.json")))
    allr = [(p, json.load(open(p))) for p in paths]
    # "not scored" means the scoring job failed (e.g. a package mirror returned 503), not the proposal: it is pending a
    # rescore and stays out of the counts, so infrastructure faults can't lower the valid rate (2026-11-19, runs 44520331877, 44522134169)
    rs_pairs = [(p, r) for p, r in allr if r.get("extract") != "not scored"]
    rs = [r for _, r in rs_pairs]
    pending = [r["name"] for _, r in allr if r.get("extract") == "not scored"]
    n, k = len(rs), sum(r["valid"] for r in rs)
    lo, hi = wilson(k, n)
    calls = sum(r.get("calls", 1) for r in rs)
    fr, prov, rc = fronts(rs)
    design = {}
    for p, r in rs_pairs:
        folder = os.path.dirname(p)
        tp, ip = os.path.join(folder, "train.py"), os.path.join(folder, "infer.c")
        design[r["name"]] = None
        if os.path.exists(tp) and os.path.exists(ip):
            h = hashlib.sha256()
            with open(tp, "rb") as f:
                h.update(f.read())
            with open(ip, "rb") as f:
                h.update(f.read())
            design[r["name"]] = h.hexdigest()
    first_seen, dup = {}, {}
    for r in rs:
        h = design.get(r["name"])
        if h is None:
            continue
        if h in first_seen:
            dup[r["name"]] = first_seen[h]
        else:
            first_seen[h] = r["name"]
    valid_hashes = [design.get(r["name"]) for r in rs if r.get("valid")]
    distinct = len({h for h in valid_hashes if h is not None}) + sum(1 for h in valid_hashes if h is None)
    lines = ["# P5 proposals: leaderboard (generated by runner/proposals/judge.py; rules in runner/proposals/PLAN.md on main)", "",
             f"Proposals: {n}. Valid: {k} (rate {k / n if n else 0:.2f}, 95% Wilson {lo:.2f}-{hi:.2f}). "
             f"Beat label-linear: {sum(r['beats_linear'] for r in rs)}. Free calls per valid: "
             f"{calls / k if k else float('inf'):.1f}. On the accuracy-cost front of their task: {len(fr)} "
             f"(no other valid proposal at least as accurate within 1 SE and cheaper in whole-run Ir/item, startup "
             f"included; the marginal Ir/item is shown, not ranked).", "",
             f"Distinct designs: {distinct} (of {k} valid).", ""]
    if rc or prov:
        lines += [f"Scored before the whole-run count was recorded, on the front at the task's measured startup floor "
                  f"(marginal Ir/item plus the smallest startup cost measured on the task; front column 'rescore'): {len(rc)}"
                  + (f" ({', '.join(sorted(rc))})" if rc else "") + f". Beaten only by those, so decided by their rescore "
                  f"(front column 'provisional'): {len(prov)}.", ""]
    if pending:
        lines += [f"Not scored (scoring job failed; pending a rescore, not counted): {', '.join(pending)}.", ""]
    if n >= 20:
        first = rs[:20]
        kv = sum(r["valid"] for r in first)
        lines += [f"First 20: {kv} valid -> {'KEEP the one-call proposer' if kv >= 2 else 'SWITCH to an agent harness'} "
                  "(rule fixed in PLAN.md).", ""]
    lines += ["| proposal | task | extracted | reproduced | valid | dev accuracy | vs linear | Ir/item whole run | "
              "Ir/item marginal | ns/item | front | dup of |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rs:
        acc = f"{r['mean']:.4f} ± {r['se']:.4f}" if "mean" in r else ""
        vl = f"{r['over_linear']:+.4f} ± {r['over_linear_se']:.4f}" if "over_linear" in r else ""
        ir = f"{r['ir_per_item']:.0f}" if r.get("ir_per_item") else ""
        ns = f"{r['ns_per_item']:.0f}" if r.get("ns_per_item") else ""
        whole = f"{r[COST]:.0f}" if r.get(COST) else ""
        fc = "rescore" if r["name"] in rc else "provisional" if r["name"] in prov else r["name"] in fr
        lines.append(f"| {r['name']} | {r.get('task', '')} | {r['extracted']} | {r['reproduced']} | {r['valid']} | {acc} | {vl} | {whole} | {ir} | {ns} | {fc} | {dup.get(r['name'], '')} |")
    open(os.path.join(d, "LEADERBOARD.md"), "w").write("\n".join(lines) + "\n")
    return n, k


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "one":
        v = one(a[1], a[2], a[3], a[4], a[6] if len(a) > 6 else "")
        json.dump(v, open(a[5], "w"), indent=1)
        print(json.dumps(v))
    else:
        print("proposals %d, valid %d" % board(a[1]))
