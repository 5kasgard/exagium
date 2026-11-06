"""Apply plan 2's decision rule to one scoreboard JSON. Usage: python3 -I score.py FILE..."""
import json, sys

BASE = ("linear", "thermo")


def part(s):
    c = {x["name"]: x for x in s["candidates"]}
    p = {(x["a"], x["b"]): x for x in s["pairs"]}
    acc = {k: v["correct"] / v["n"] for k, v in c.items()}
    ir = {k: v["simulated_Ir_per_image"] for k, v in c.items()}
    lines = []
    lines.append(f"host {s['host']['cpu']} ({s['host']['machine']}), n={s['n']}, valid={s.get('valid')}, problems={s['problems']}")
    for k in c:
        lines.append(f"  {k}: correct {c[k]['correct']}/{c[k]['n']} acc {acc[k]:.6f} Ir/img {ir[k]:.3f} "
                     f"Ir/correct {c[k]['simulated_Ir_per_correct']:.3f} pred_sha {c[k]['predictions_sha256'][:16]}")

    def beats(x, b):
        d, se = p[(x, b)]["accuracy_diff"], p[(x, b)]["accuracy_diff_se"]
        lhs = ir[x] / (acc[x] - 2 * se)
        rhs = ir[b] / acc[b]
        return d, se, lhs, rhs, lhs < rhs

    # Part A
    okA = True
    for b in BASE:
        d, se, lhs, rhs, ok = beats("mlp16", b)
        ratio = ir["mlp16"] / ir[b]
        lines.append(f"  A mlp16 vs {b}: d={d:+.6f} SE={se:.6f}; cost {ir['mlp16']:.2f}/({acc['mlp16']:.6f}-2*{se:.6f})"
                     f"={lhs:.2f} < {ir[b]:.2f}/{acc[b]:.6f}={rhs:.2f}: {ok}; Ir ratio {ratio:.4f} <= 0.5: {ratio <= 0.5}")
        okA &= ok and ratio <= 0.5
    floor = max(acc["linear"], acc["thermo"]) - 0.005
    fl = acc["mlp16"] >= floor
    lines.append(f"  A floor: acc_mlp16 {acc['mlp16']:.6f} >= max({acc['linear']:.6f},{acc['thermo']:.6f})-0.005 = {floor:.6f}: {fl}")
    okA &= fl
    A = "PASS" if okA else "FAIL"
    # Part B
    passB, cost_any, fail_acc = False, False, True
    for x in ("mlp32", "mlp64"):
        acc_ok, cost_ok, nonpos = True, True, False
        for b in BASE:
            d, se, lhs, rhs, ok = beats(x, b)
            lines.append(f"  B {x} vs {b}: d={d:+.6f} SE={se:.6f} 2SE={2*se:.6f} z={d/se:.3f} d>2SE: {d > 2*se}; "
                         f"cost {lhs:.2f} < {rhs:.2f}: {ok}")
            acc_ok &= d > 2 * se
            cost_ok &= ok
            nonpos |= d <= 0
        passB |= acc_ok and cost_ok
        cost_any |= cost_ok
        fail_acc &= nonpos
    B = "PASS" if passB else ("FAIL" if (fail_acc or not cost_any) else "INCONCLUSIVE")
    lines.append(f"  => Part A {A}; Part B {B}")
    return A, B, lines, {k: c[k]["predictions_sha256"] for k in c}, {k: c[k]["correct"] for k in c}


if __name__ == "__main__":
    for f in sys.argv[1:]:
        s = json.load(open(f))
        A, B, lines, _, _ = part(s)
        print(f"== {f}")
        print("\n".join(lines))
