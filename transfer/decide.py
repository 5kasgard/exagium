"""Apply PLAN.md's fixed rule to the meter outputs of all tasks: do rankings transfer?

python3 transfer/decide.py RESULT_JSON... [--out DECISION_JSON]

For every pair of families (A, B) and every task, an order holds when it clears the plan's bar:
  accuracy: D = mean over eval items of (mean over seeds of A correct) - (same for B);
            SE^2 = var(per-item differences)/n + var_seeds(acc A)/s_A + var_seeds(acc B)/s_B;
            A above B when D > 2 SE.
  instructions per item (deterministic): mean over seeds; A cheaper when Ir_A * 1.10 < Ir_B.
  native ns per item: per round, mean over seeds of A minus the same of B; A cheaper when the mean of those
            round differences is below -2 SE (SE = sd/sqrt(rounds)).
A flip: the order of A and B holds one way on one task and the other way on another task.
Verdict: "rankings differ" if any flip on any of the three measures; else "no flip found".
budget_check lists (A, B, task1, task2) for every accuracy flip: PLAN.md retrains A and B on both tasks at 4x the
budget and runs this script on those results; the flip "survives the budget check" if it is found again.
Exit codes: 0 done, 1 an input lists problems (the verdict is then marked invalid), 3 failure.
"""
import argparse, base64, itertools, json, math, statistics, sys

import numpy as np


def b64(s):
    return np.frombuffer(base64.b64decode(s), dtype=np.uint8)


def task_summary(res):
    y = b64(res["eval_y_b64"])
    fams = {}
    for name, a in res["artifacts"].items():
        fams.setdefault(a["family"], []).append(a)
    out = {}
    for f, arts in fams.items():
        arts.sort(key=lambda a: a["seed"])
        corr = np.stack([(b64(a["pred_b64"]) == y).astype(float) for a in arts])
        out[f] = dict(seeds=[a["seed"] for a in arts], acc_seeds=[float(c.mean()) for c in corr],
                      item_mean=corr.mean(axis=0), acc=float(corr.mean()),
                      ir=statistics.mean(a["ir_per_item"] for a in arts),
                      ns_rounds=np.mean([a["ns_rounds"] for a in arts], axis=0),
                      train_acc=[a["train"].get("train_accuracy") for a in arts])
    return out


def orders(s):
    """dict (A, B) -> {'acc': +1/-1/0, 'ir': ..., 'ns': ...} with +1 meaning A better (higher acc / cheaper)."""
    o = {}
    for A, B in itertools.combinations(sorted(s), 2):
        a, b = s[A], s[B]
        d = a["item_mean"] - b["item_mean"]
        var = d.var(ddof=1) / len(d)
        for x in (a, b):
            if len(x["acc_seeds"]) > 1:
                var += statistics.variance(x["acc_seeds"]) / len(x["acc_seeds"])
        D, se = float(d.mean()), math.sqrt(var)
        acc = 1 if D > 2 * se else -1 if D < -2 * se else 0
        ir = 1 if a["ir"] * 1.10 < b["ir"] else -1 if b["ir"] * 1.10 < a["ir"] else 0
        rd = a["ns_rounds"] - b["ns_rounds"]
        nD, nse = float(rd.mean()), float(rd.std(ddof=1) / math.sqrt(len(rd)))
        ns = 1 if nD < -2 * nse else -1 if nD > 2 * nse else 0
        o[(A, B)] = dict(acc=acc, acc_diff=D, acc_se=se, ir=ir, ir_ratio=a["ir"] / b["ir"], ns=ns, ns_diff=nD,
                         ns_se=nse)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = [json.load(open(p)) for p in a.results]
    problems = [f"{r['task']}: {p}" for r in res for p in r["problems"]]
    sums = {r["task"]: task_summary(r) for r in res}
    ords = {t: orders(s) for t, s in sums.items()}
    flips = []
    pairs = set().union(*[set(o) for o in ords.values()])
    for pair in sorted(pairs):
        for m in ("acc", "ir", "ns"):
            for t1, t2 in itertools.permutations(sorted(ords), 2):
                if pair in ords[t1] and pair in ords[t2] and ords[t1][pair][m] == 1 and ords[t2][pair][m] == -1:
                    flips.append(dict(measure=m, pair=list(pair), better_on=t1, worse_on=t2,
                                      **{f"{t}_{k}": ords[t][pair][k] for t in (t1, t2)
                                         for k in (("acc_diff", "acc_se") if m == "acc" else
                                                   ("ir_ratio",) if m == "ir" else ("ns_diff", "ns_se"))}))
    by = {m: sum(f["measure"] == m for f in flips) for m in ("acc", "ir", "ns")}
    verdict = "rankings differ" if flips else "no flip found"
    table = {t: {f: dict(acc=round(v["acc"], 4), acc_seeds=[round(x, 4) for x in v["acc_seeds"]],
                         train_acc=v["train_acc"], ir_per_item=round(v["ir"], 1),
                         ns_per_item=round(float(np.mean(v["ns_rounds"])), 1)) for f, v in s.items()}
             for t, s in sums.items()}
    rank = {t: {m: sorted(s, key=lambda f: (-s[f]["acc"] if m == "acc" else s[f]["ir"] if m == "ir"
                                            else float(np.mean(s[f]["ns_rounds"]))))
                for m in ("acc", "ir", "ns")} for t, s in sums.items()}
    out = dict(verdict=verdict, valid=not problems, problems=problems, flips_by_measure=by, flips=flips,
               table=table, rank=rank, majority={r["task"]: r.get("majority") for r in res},
               below_majority=[f"{t}/{f}" for r in res for t in [r["task"]] for f, v in table[t].items()
                               if r.get("majority") and v["acc"] <= r["majority"]["accuracy"]],
               budget_check=sorted({(f["pair"][0], f["pair"][1], *sorted((f["better_on"], f["worse_on"])))
                                    for f in flips if f["measure"] == "acc"}), hosts={r["task"]: r["host"] for r in res})
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, sort_keys=True)
    for t in sorted(table):
        print(f"\n{t} ({out['hosts'][t]['cpu']})")
        for f in rank[t]["acc"]:
            v = table[t][f]
            print(f"  {f:7s} acc {v['acc']:.4f} {v['acc_seeds']}  Ir/item {v['ir_per_item']:9.1f}  "
                  f"ns/item {v['ns_per_item']:8.1f}")
    for t, m in out["majority"].items():
        if m:
            print(f"  majority class on {t}: {m['accuracy']:.4f}")
    if out["below_majority"]:
        print("  AT OR BELOW THE MAJORITY CLASS:", out["below_majority"])
    print(f"\nflips: {by}")
    for f in flips:
        print("  ", json.dumps(f))
    print("VERDICT:", verdict, "" if not problems else "(INVALID: problems listed)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(3)
