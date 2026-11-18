You are reviewing a pre-registered experiment BEFORE it runs. Be adversarial and specific. Do not praise. For each problem give: location (file and function or PLAN.md section), what is wrong, why it matters for the verdict, and a concrete fix. Grade each critical / major / minor. End with a verdict: run as is / fix then run / do not run.

CONTEXT
- The project has measured small classifiers on one task only (optdigits, 8x8 digits). This step asks whether the ORDER of the same model families (linear, mlp16/32/64, a logic-gate network "lgn") changes across three unlike tasks: optdigits, next-character text, UCI Adult tabular. Order on accuracy, on instructions per item, and on native ns per item.
- The rule in decide.py is fixed before running. Focus on: whether the rule can produce a false "rankings differ" or a false "no flip found"; statistical errors in the SE (per-item variance, seed variance, paired rounds); confounds (training budget, encodings, hosts); bugs in the code that would change outcomes; whether accuracy as the text metric is defensible; anything that makes the C-vs-numpy equality check pass wrongly.
- Not in scope: style, the absence of a jail (stated as a limit), the choice of only three tasks (stated).

=== transfer/PLAN.md ===
# Transfer plan, step 1: do rankings transfer across kinds of task?

Fixed on 2026-11-18, before any full run. The commit that adds this file is the registration; the run that
answers it must come from a later commit that leaves this file, `decide.py` and the hyperparameters in
`families.py` unchanged (a check in `transfer.yml` compares them with the registered commit).

## Why
Every result so far holds for one task: optdigits (8×8 digit images). If the order of families changes from one
kind of task to another, a one-task Forge misleads its own spec cards, and the next steps (a general task contract,
a mixed suite) are needed before any "best part" claim. If no order changes, the cheaper path stays open.

## Hypothesis
Rankings of the same families differ across kinds of task.

## Tasks (byte records, `tasks.py`; downloads pinned by sha256)
| task | kind | items: train / eval | bytes per item | classes |
|---|---|---|---|---|
| optdigits | vision: 8×8 pixel counts 0..16 | 2,867 / 956 (loop.py's dev split) | 64 ordinal | 10 |
| text | language: next character from the previous 16 bytes, tinyshakespeare | 30,000 / 5,000 (eval from the last 10% of the text) | 16 categorical | 65 |
| adult | tabular: UCI Adult income; 6 numeric (255 training-quantile bins), 8 categorical | 32,561 / 16,281 (official split) | 14 mixed | 2 |

Eval is never used for a choice: no hyperparameter is tuned, on any split. The text metric is next-character
**accuracy**, not bits per character: the logic-gate network and the C programs output a class, not a
distribution, and one metric must serve every family.

## Families (`families.py`, one shared encoding rule per feature kind)
- **linear**: logistic regression. **mlp16, mlp32, mlp64**: one ReLU layer. Ordinal bytes z-scored, categorical
  bytes one-hot. Adam lr 3e-3, L2 1e-4. Exported as per-feature lookup tables (`infer_table.c`).
- **lgn**: logic-gate network with learned Top-K wiring (as lgn_d): K = 8, widths 2000, 1000, 50 × classes,
  tau 4, Adam lr 0.01. Ordinal bytes as thermometer bits at the training quantiles 0.2/0.4/0.6/0.8; categorical
  bytes as one bit per value (64 most frequent). Exported as gates, run bit-sliced (`infer_lgn.c`).
- **Same budget for all:** epochs = min(100, ⌈600,000 / n_train⌉), minibatch 64; 3 seeds (1, 2, 3) each.
- Mandatory baselines: linear is in every comparison; the majority class is reported.
- **Not in this step:** a PFN (TabPFN needs PyTorch and downloaded weights, and its context limit is below
  adult's 32,561 rows; whether it fits CI's CPU limits is untested). Trees and boosting: step 2.

## Measures (`meter.py`, one CI host per task)
- **Per-item outcomes**: every artifact's class for every eval item, kept. The C output must equal the numpy run
  of the exported form byte for byte.
- **Accuracy** on eval, averaged over seeds.
- **Instructions per item**: cachegrind I refs at N = n_eval minus N = 0, over n_eval. Deterministic.
- **Native ns per item**: wall time at 20,000 items (eval repeated) minus N = 0, over 20,000; 7 rounds, random
  order each round, one pinned CPU.

## The rule (`decide.py`), fixed now
For each pair of families A, B and each task, an order holds when:
- **accuracy:** D = mean over items of (A's correct, averaged over seeds) − (B's); SE² = per-item variance of the
  differences / n + seed variance of A's accuracy / 3 + the same for B. A above B when D > 2 SE.
- **instructions:** A cheaper when Ir_A × 1.10 < Ir_B (a deterministic meter has no SE; 10% is a fixed tolerance).
- **native time:** A cheaper when the mean of per-round differences is below −2 SE.

**A flip:** an order holds one way on one task and the other way on another. **Verdict:** "rankings differ" if
there is any flip on any of the three measures; else "no flip found". Any problem in the meter (a C/numpy mismatch,
a failed run) marks the verdict invalid.

**What follows:** "rankings differ" → step 2 (a general task contract and a mixed suite) before any cross-task
claim. "No flip found" → step 2 is still needed for breadth, but a single-task result gains weight. Either
way, accuracy flips and cost flips are reported apart, because cost flips can follow from input shape alone.

## Limits, stated in advance
- Three tasks, one host each, one budget rule: a flip may come from the budget (logic-gate networks often need
  more epochs than MLPs) rather than from structure. Training accuracy is reported so that underfitting shows.
- The encodings are one choice each; a family with its own preferred encoding could rank differently.
- Seed variance from 3 seeds is a rough estimate.
- No jail: every program here was written in this plan; none was proposed by an agent. Step 4's proposals
  must go through loop's jail.
- The generic logic-gate program is not lgn_d's compiled circuit (no constant folding or pruning), so its costs
  are not lgn_d's numbers.

=== transfer/decide.py ===
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
               table=table, rank=rank, hosts={r["task"]: r["host"] for r in res})
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, sort_keys=True)
    for t in sorted(table):
        print(f"\n{t} ({out['hosts'][t]['cpu']})")
        for f in rank[t]["acc"]:
            v = table[t][f]
            print(f"  {f:7s} acc {v['acc']:.4f} {v['acc_seeds']}  Ir/item {v['ir_per_item']:9.1f}  "
                  f"ns/item {v['ns_per_item']:8.1f}")
    print(f"\nflips: {by}")
    for f in flips:
        print("  ", json.dumps(f))
    print("VERDICT:", verdict, "" if not problems else "(INVALID: problems listed)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
=== transfer/families.py ===
"""Train one family on one task card and export what the C programs run (PLAN.md, step 1).

python3 transfer/families.py TASK_DIR FAMILY SEED OUT_WEIGHTS
FAMILY: linear | mlp16 | mlp32 | mlp64 | lgn

Every family sees the same bytes and the same per-feature types, with one shared encoding rule per kind:
  - table families (linear, mlp*): an ordinal byte is z-scored with training mean and sd; a categorical byte is
    one-hot over the values seen in training (an unseen value encodes as all zeros);
  - lgn: an ordinal byte becomes thermometer bits v >= t for t at the training quantiles 0.2, 0.4, 0.6, 0.8
    (duplicates and a threshold at the minimum dropped); a categorical byte becomes one bit v == c for each of
    its (at most 64) most frequent training values.
Training budget, the same rule for every family: epochs = min(100, ceil(600000 / n_train)), minibatch 64,
Adam. Nothing is tuned on the eval split; every hyperparameter below is fixed in PLAN.md before any run.

Exported forms (little-endian):
  table (linear, mlp*): because each feature is one byte, layer 1 (or the linear scores) is a sum of per-feature
    lookups: T[j][v] = W_j^T enc_j(v) for every byte value v. File: int32 magic 0x54424C31, d, H (0 = linear), C;
    float32 T[d][256][W] (W = H, or C for linear), float32 b1[W]; if H > 0: float32 W2[H][C], float32 b2[C].
  lgn: int32 magic 0x4C474E31, d, C, g (gates per class group), U (input bits used), L (layers); U x (int32 j,
    kind 0 = '>=' / 1 = '==', value); per layer: int32 width, then width x (int32 a, b, op). Layer 0 indexes the
    U used input bits; class c scores the popcount of last-layer gates c*g .. c*g+g-1; argmax, first max wins.
Prints one JSON line: family, seed, epochs, train accuracy (of the exported form), sizes, seconds.
Also writes OUT_WEIGHTS + ".pred" (uint8 class per eval item, from a numpy run of the exported form), which the
meter checks against the C program's output byte for byte.
"""
import json, math, struct, sys, time

import numpy as np

QUANTILES = (0.2, 0.4, 0.6, 0.8)
MAX_CATS = 64
SAMPLES = 600000
MAX_EPOCHS = 100
BATCH = 64
TABLE_LR, TABLE_L2 = 3e-3, 1e-4
LGN_LR, LGN_K, LGN_TAU, LGN_GROUP, LGN_HIDDEN = 0.01, 8, 4.0, 50, (2000, 1000)


def load(task):
    card = json.load(open(f"{task}/card.json"))
    d = card["d"]
    rd = lambda n: np.fromfile(f"{task}/{n}", dtype=np.uint8)
    return card, rd("train.x").reshape(-1, d), rd("train.y").astype(np.int64), rd("eval.x").reshape(-1, d)


def epochs_for(n):
    return min(MAX_EPOCHS, math.ceil(SAMPLES / n))


# ---------------------------------------------------------------- table families
def table_encoder(card, X):
    """per feature: a function byte value (0..255) -> encoded vector; returns list of (256, k_j) float arrays."""
    enc = []
    for j, t in enumerate(card["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def encode(enc, X):
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)


def adam_step(params, grads, state, lr, t):
    for i, (p, g) in enumerate(zip(params, grads)):
        m, v = state[i]
        m *= 0.9; m += 0.1 * g
        v *= 0.999; v += 0.001 * g * g
        p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)


def softmax_grad(S, y):
    S = S - S.max(axis=1, keepdims=True)
    P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
    P[np.arange(len(y)), y] -= 1.0
    return P / len(y)


def train_table(card, X, y, H, seed):
    rng = np.random.default_rng(seed)
    enc = table_encoder(card, X)
    Z = encode(enc, X)
    n, k = Z.shape
    C = card["n_classes"]
    if H:
        W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [W1, b1, W2, b2]
    else:
        W1 = np.zeros((k, C), np.float32); b1 = np.zeros(C, np.float32)
        params = [W1, b1]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    ep = epochs_for(n); t = 0
    for _ in range(ep):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            if H:
                a = z @ W1 + b1; h = np.maximum(a, 0)
                g = softmax_grad(h @ W2 + b2, yy).astype(np.float32)
                gW2 = h.T @ g + TABLE_L2 * W2; gb2 = g.sum(0)
                gh = (g @ W2.T) * (a > 0)
                grads = [z.T @ gh + TABLE_L2 * W1, gh.sum(0), gW2, gb2]
            else:
                g = softmax_grad(z @ W1 + b1, yy).astype(np.float32)
                grads = [z.T @ g + TABLE_L2 * W1, g.sum(0)]
            t += 1
            adam_step(params, grads, state, TABLE_LR, t)
    # fold into per-feature tables over all 256 byte values
    off, T = 0, []
    for E in enc:
        kj = E.shape[1]
        T.append((E.astype(np.float32) @ W1[off:off + kj]).astype(np.float32))
        off += kj
    T = np.stack(T)  # (d, 256, W)
    model = dict(T=T, b1=b1, W2=params[2] if H else None, b2=params[3] if H else None, H=H, C=C)
    return model, ep


def table_predict(m, X):
    """exactly what infer_table.c computes: float32, sums in feature order, then in hidden-unit order."""
    acc = np.broadcast_to(m["b1"], (len(X), len(m["b1"]))).astype(np.float32).copy()
    for j in range(X.shape[1]):
        acc += m["T"][j][X[:, j]]
    if not m["H"]:
        return acc.argmax(axis=1)
    h = np.maximum(acc, np.float32(0))
    s = np.broadcast_to(m["b2"], (len(X), m["C"])).astype(np.float32).copy()
    for u in range(m["H"]):
        s += h[:, u:u + 1] * m["W2"][u]
    return s.argmax(axis=1)


def table_bytes(m, d):
    W = m["H"] or m["C"]
    out = struct.pack("<4i", 0x54424C31, d, m["H"], m["C"])
    out += m["T"].astype("<f4").tobytes() + m["b1"].astype("<f4").tobytes()
    if m["H"]:
        out += m["W2"].astype("<f4").tobytes() + m["b2"].astype("<f4").tobytes()
    assert m["T"].shape == (d, 256, W)
    return out


# ---------------------------------------------------------------- logic-gate network
# 16 two-input ops as coefficients over [1, a, b, ab] (as in lgn_d)
M = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, -1], [0, 1, 0, 0], [0, 0, 1, -1], [0, 0, 1, 0],
              [0, 1, 1, -2], [0, 1, 1, -1], [1, -1, -1, 1], [1, -1, -1, 2], [1, 0, -1, 0], [1, 0, -1, 1],
              [1, -1, 0, 0], [1, -1, 0, 1], [1, 0, 0, -1], [1, 0, 0, 0]], dtype=np.float32)


def lgn_bits(card, X):
    """list of (j, kind, value): kind 0 means v >= value, kind 1 means v == value."""
    bits = []
    for j, t in enumerate(card["types"]):
        v = X[:, j]
        if t == "ordinal":
            ts = sorted({int(np.quantile(v, q, method="higher")) for q in QUANTILES} - {int(v.min())})
            bits += [(j, 0, tv) for tv in ts]
        else:
            vals, cnt = np.unique(v, return_counts=True)
            keep = vals[np.argsort(-cnt, kind="stable")[:MAX_CATS]]
            bits += [(j, 1, int(c)) for c in sorted(keep)]
    return bits


def apply_bits(bits, X):
    out = np.zeros((len(X), len(bits)), dtype=bool)
    for i, (j, kind, val) in enumerate(bits):
        out[:, i] = X[:, j] >= val if kind == 0 else X[:, j] == val
    return out


def softmax(w):
    e = np.exp(w - w.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def scatter_cols(V, idx, P):
    """sum columns of V (B, n) into P output columns by idx (n,), fast (sort + reduceat)."""
    order = np.argsort(idx, kind="stable")
    si = idx[order]
    starts = np.flatnonzero(np.r_[True, si[1:] != si[:-1]])
    out = np.zeros((V.shape[0], P), dtype=V.dtype)
    out[:, si[starts]] = np.add.reduceat(V[:, order], starts, axis=1)
    return out


def train_lgn(card, X, y, seed):
    rng = np.random.default_rng(seed)
    bits = lgn_bits(card, X)
    B0 = apply_bits(bits, X).astype(np.float32)
    C = card["n_classes"]
    widths = list(LGN_HIDDEN) + [C * LGN_GROUP]
    layers, prev = [], len(bits)
    for w in widths:
        layers.append(dict(ca=rng.integers(0, prev, (w, LGN_K)), cb=rng.integers(0, prev, (w, LGN_K)),
                           lo=rng.standard_normal((w, 16)).astype(np.float32),
                           la=np.zeros((w, LGN_K), np.float32), lb=np.zeros((w, LGN_K), np.float32)))
        prev = w
    params = [L[k] for L in layers for k in ("lo", "la", "lb")]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    n = len(y); ep = epochs_for(n); t = 0
    for _ in range(ep):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x = B0[idx]; cache = []
            for L in layers:
                pa, pb, po = softmax(L["la"]), softmax(L["lb"]), softmax(L["lo"])
                Xa, Xb = x[:, L["ca"]], x[:, L["cb"]]
                a = np.einsum("bwk,wk->bw", Xa, pa); b = np.einsum("bwk,wk->bw", Xb, pb)
                c = po @ M
                cache.append((x, Xa, Xb, a, b, pa, pb, po, c))
                x = c[:, 0] + c[:, 1] * a + c[:, 2] * b + c[:, 3] * (a * b)
            S = x.reshape(len(idx), C, LGN_GROUP).sum(axis=2) / LGN_TAU
            gy = np.repeat(softmax_grad(S, y[idx]).astype(np.float32) / LGN_TAU, LGN_GROUP, axis=1)
            grads = []
            for li in range(len(layers) - 1, -1, -1):
                L = layers[li]
                xin, Xa, Xb, a, b, pa, pb, po, c = cache[li]
                gc = np.stack([gy.sum(0), (gy * a).sum(0), (gy * b).sum(0), (gy * a * b).sum(0)], axis=1)
                gpo = gc @ M.T
                glo = po * (gpo - (po * gpo).sum(1, keepdims=True))
                ga = gy * (c[:, 1] + c[:, 3] * b); gb = gy * (c[:, 2] + c[:, 3] * a)
                gpa = np.einsum("bw,bwk->wk", ga, Xa); gpb = np.einsum("bw,bwk->wk", gb, Xb)
                gla = pa * (gpa - (pa * gpa).sum(1, keepdims=True))
                glb = pb * (gpb - (pb * gpb).sum(1, keepdims=True))
                grads = [glo, gla, glb] + grads
                if li:
                    P = xin.shape[1]
                    Va = (ga[:, :, None] * pa).reshape(len(idx), -1)
                    Vb = (gb[:, :, None] * pb).reshape(len(idx), -1)
                    gy = scatter_cols(Va, L["ca"].ravel(), P) + scatter_cols(Vb, L["cb"].ravel(), P)
            t += 1
            adam_step(params, grads, state, LGN_LR, t)
    net = []
    for L in layers:
        w = len(L["lo"])
        net.append((L["ca"][np.arange(w), L["la"].argmax(1)], L["cb"][np.arange(w), L["lb"].argmax(1)],
                    L["lo"].argmax(1)))
    # keep only input bits the first layer reads, renumbered
    used = sorted(set(net[0][0]) | set(net[0][1]))
    remap = {u: i for i, u in enumerate(used)}
    a0, b0, o0 = net[0]
    net[0] = (np.array([remap[i] for i in a0]), np.array([remap[i] for i in b0]), o0)
    return dict(bits=[bits[u] for u in used], net=net, C=C, g=LGN_GROUP), ep


def OP(k, a, b):
    return [np.zeros_like(a), a & b, a & ~b, a, ~a & b, b, a ^ b, a | b, ~(a | b), ~(a ^ b), ~b, a | ~b, ~a,
            ~a | b, ~(a & b), np.ones_like(a)][k]


def lgn_predict(m, X):
    x = apply_bits(m["bits"], X)
    for ia, ib, ops in m["net"]:
        a, b = x[:, ia], x[:, ib]
        out = np.zeros((len(x), len(ops)), dtype=bool)
        for k in range(16):
            sel = ops == k
            if sel.any():
                out[:, sel] = OP(k, a[:, sel], b[:, sel])
        x = out
    return x.reshape(len(x), m["C"], m["g"]).sum(axis=2).argmax(axis=1)


def lgn_bytes(m, d):
    out = struct.pack("<6i", 0x4C474E31, d, m["C"], m["g"], len(m["bits"]), len(m["net"]))
    out += np.array(m["bits"], dtype="<i4").tobytes()
    for ia, ib, ops in m["net"]:
        out += struct.pack("<i", len(ops)) + np.stack([ia, ib, ops], axis=1).astype("<i4").tobytes()
    return out


def main():
    task, fam, seed, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    card, X, y, EX = load(task)
    t0 = time.time()
    if fam == "lgn":
        m, ep = train_lgn(card, X, y, seed)
        blob, pred, ptr = lgn_bytes(m, card["d"]), lgn_predict(m, EX), lgn_predict(m, X)
        size = dict(input_bits=len(m["bits"]), gates=int(sum(len(o) for _, _, o in m["net"])))
    else:
        H = {"linear": 0, "mlp16": 16, "mlp32": 32, "mlp64": 64}[fam]
        m, ep = train_table(card, X, y, H, seed)
        blob, pred, ptr = table_bytes(m, card["d"]), table_predict(m, EX), table_predict(m, X)
        size = dict(hidden=H, table_floats=int(m["T"].size))
    open(out, "wb").write(blob)
    pred.astype(np.uint8).tofile(out + ".pred")
    print(json.dumps(dict(task=card["task"], family=fam, seed=seed, epochs=ep,
                          train_accuracy=round(float((ptr == y).mean()), 6), bytes=len(blob),
                          seconds=round(time.time() - t0, 1), **size)))


if __name__ == "__main__":
    main()
=== transfer/meter.py ===
"""Score and meter every trained artifact of one task (PLAN.md, step 1).

python3 transfer/meter.py TASK_DIR WEIGHTS_DIR OUT_JSON [--rounds 7] [--timing-items 20000] [--seed 1]

WEIGHTS_DIR holds <family>-s<seed>.bin (+ .pred, + .json: the training log line) and the compiled programs
infer_table and infer_lgn. For each artifact:
  - per-item outcomes: the C program's class for every eval item, which must equal families.py's numpy run of
    the same exported form byte for byte (else the artifact is listed under "problems");
  - instructions per item: valgrind (cachegrind, no cache simulation) "I refs" of one process classifying all
    eval items, minus the same at N = 0, divided by n_eval: the marginal count, start-up removed;
  - native ns per item: wall time of one process on a timing file (eval items repeated in order up to
    --timing-items), minus the same process at N = 0, divided by the item count. Rounds are the unit of
    replication; every round runs every artifact in a fresh random order, pinned to one CPU.
No jail: every program measured here was written in this plan, not proposed (PLAN.md, "Limits").
Exit codes: 0 done, 1 a check failed (see "problems"), 3 failure.
"""
import argparse, base64, glob, json, os, platform, random, re, subprocess, sys, time

import numpy as np


def ir(prog, w, x, n):
    r = subprocess.run(["valgrind", "--tool=cachegrind", "--cache-sim=no", "--cachegrind-out-file=/dev/null",
                        prog, w, x, str(n)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    m = re.search(r"I\s+refs:\s+([\d,]+)", r.stderr)
    if r.returncode or not m:
        raise RuntimeError(f"valgrind failed on {w}: {r.stderr[-400:]}")
    return int(m.group(1).replace(",", ""))


def wall_ns(prog, w, x, n, cpu):
    t0 = time.perf_counter_ns()
    r = subprocess.run([prog, w, x, str(n)], stdout=subprocess.DEVNULL,
                       preexec_fn=lambda: os.sched_setaffinity(0, {cpu}))
    t1 = time.perf_counter_ns()
    if r.returncode:
        raise RuntimeError(f"{prog} {w} exited {r.returncode}")
    return t1 - t0


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task"); ap.add_argument("wdir"); ap.add_argument("out")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--timing-items", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    card = json.load(open(os.path.join(a.task, "card.json")))
    d, n = card["d"], card["n_eval"]
    ex = os.path.join(a.task, "eval.x")
    y = np.fromfile(os.path.join(a.task, "eval.y"), dtype=np.uint8)
    reps = -(-a.timing_items // n)
    tx = os.path.join(a.wdir, "timing.x")
    np.tile(np.fromfile(ex, dtype=np.uint8).reshape(n, d), (reps, 1))[:a.timing_items].tofile(tx)
    problems, arts = [], {}
    for w in sorted(glob.glob(os.path.join(a.wdir, "*-s*.bin"))):
        name = os.path.basename(w)[:-4]
        fam = name.rsplit("-s", 1)[0]
        prog = os.path.join(a.wdir, "infer_lgn" if fam == "lgn" else "infer_table")
        r = subprocess.run([prog, w, ex, str(n)], stdout=subprocess.PIPE)
        pred = np.frombuffer(r.stdout, dtype=np.uint8)
        ref = np.fromfile(w + ".pred", dtype=np.uint8)
        if r.returncode or len(pred) != n or not np.array_equal(pred, ref):
            problems.append(f"{name}: C output differs from the numpy run of the exported form")
        ir_n, ir_0 = ir(prog, w, ex, n), ir(prog, w, ex, 0)
        arts[name] = dict(family=fam, seed=int(name.rsplit("-s", 1)[1]), prog=os.path.basename(prog),
                          train=json.load(open(w[:-4] + ".json")), accuracy=float((pred == y).mean()),
                          ir_per_item=(ir_n - ir_0) / n, ir_startup=ir_0, ns_rounds=[],
                          pred_b64=base64.b64encode(pred.tobytes()).decode())
    rng = random.Random(a.seed)
    cpu = sorted(os.sched_getaffinity(0))[-1]
    names = sorted(arts)
    for _ in range(a.rounds):
        rng.shuffle(names)
        for name in names:
            w = os.path.join(a.wdir, name + ".bin")
            prog = os.path.join(a.wdir, arts[name]["prog"])
            t0 = wall_ns(prog, w, tx, 0, cpu)
            tn = wall_ns(prog, w, tx, a.timing_items, cpu)
            arts[name]["ns_rounds"].append((tn - t0) / a.timing_items)
    res = dict(task=card["task"], card=card, n_eval=n, eval_y_b64=base64.b64encode(y.tobytes()).decode(),
               rounds=a.rounds, timing_items=a.timing_items, host=dict(cpu=cpu_model(), machine=platform.machine(),
               kernel=platform.release(), runner=os.environ.get("RUNNER_NAME", ""),
               run_id=os.environ.get("GITHUB_RUN_ID", "")), artifacts=arts, problems=problems)
    json.dump(res, open(a.out, "w"), indent=1, sort_keys=True)
    for name in sorted(arts):
        s = arts[name]
        print(f"{name:12s} acc {s['accuracy']:.4f}  Ir/item {s['ir_per_item']:9.1f}  "
              f"ns/item {np.median(s['ns_rounds']):8.1f}")
    for p in problems:
        print("PROBLEM", p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print("FAILURE", e); sys.exit(3)
=== transfer/infer_lgn.c ===
/* Logic-gate networks of the transfer plan: ./infer_lgn weights x.bin N > pred.bin
 * Bit-sliced: 64 items per uint64 word. Input bits are built from the bytes (v >= t or v == c), the gates run
 * per word, each class's group is counted with bit-sliced counters, and the argmax (first max wins) is
 * taken per item. Prints N class indices as raw bytes. N = 0 is start-up only. Format: see families.py.
 * Compile: gcc -O2 -std=c11 -ffp-contract=off infer_lgn.c -o infer_lgn */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void *slurp(const char *p, long *n) {
    FILE *f = fopen(p, "rb");
    if (!f) { perror(p); exit(3); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    void *b = malloc(*n ? *n : 1);
    if (fread(b, 1, *n, f) != (size_t)*n) { fprintf(stderr, "short read %s\n", p); exit(3); }
    fclose(f);
    return b;
}

static inline uint64_t op(int k, uint64_t a, uint64_t b) {
    switch (k) {
    case 0: return 0;
    case 1: return a & b;
    case 2: return a & ~b;
    case 3: return a;
    case 4: return ~a & b;
    case 5: return b;
    case 6: return a ^ b;
    case 7: return a | b;
    case 8: return ~(a | b);
    case 9: return ~(a ^ b);
    case 10: return ~b;
    case 11: return a | ~b;
    case 12: return ~a;
    case 13: return ~a | b;
    case 14: return ~(a & b);
    default: return ~(uint64_t)0;
    }
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer_lgn weights x.bin N\n"); return 2; }
    long nw, nx, N = atol(argv[3]);
    int32_t *w = slurp(argv[1], &nw);
    if (nw < 24 || w[0] != 0x4C474E31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = w[1], C = w[2], g = w[3], U = w[4], L = w[5];
    if (C < 1 || C > 255 || g < 1 || g > 255) { fprintf(stderr, "bad header\n"); return 3; }
    int32_t *bits = w + 6, *p = bits + 3 * U;
    int32_t **ga = malloc(sizeof(int32_t *) * L);
    int *wid = malloc(sizeof(int) * L), maxw = U;
    for (int l = 0; l < L; l++) {
        wid[l] = *p++; ga[l] = p; p += 3 * wid[l];
        if (wid[l] > maxw) maxw = wid[l];
        if ((char *)p > (char *)w + nw) { fprintf(stderr, "truncated weights\n"); return 3; }
    }
    if ((char *)p != (char *)w + nw || wid[L - 1] != C * g) { fprintf(stderr, "bad weights size\n"); return 3; }
    uint8_t *x = N ? slurp(argv[2], &nx) : NULL;
    if (N < 0 || (N && nx < N * d)) { fprintf(stderr, "bad N\n"); return 2; }
    uint64_t *bufA = malloc(8 * (size_t)maxw), *bufB = malloc(8 * (size_t)maxw);
    int planes = 0; while ((1 << planes) <= g) planes++;
    uint64_t *cnt = malloc(8 * (size_t)C * planes);
    uint8_t *out = malloc(N ? N : 1);
    for (long base = 0; base < N; base += 64) {
        int lanes = N - base < 64 ? (int)(N - base) : 64;
        const uint8_t *xb = x + base * d;
        for (int u = 0; u < U; u++) {
            int j = bits[3 * u], kind = bits[3 * u + 1], v = bits[3 * u + 2];
            uint64_t word = 0;
            for (int i = 0; i < lanes; i++) {
                int bit = kind ? xb[(long)i * d + j] == v : xb[(long)i * d + j] >= v;
                word |= (uint64_t)bit << i;
            }
            bufA[u] = word;
        }
        uint64_t *in = bufA, *o = bufB;
        for (int l = 0; l < L; l++) {
            const int32_t *G = ga[l];
            for (int k = 0; k < wid[l]; k++) o[k] = op(G[3 * k + 2], in[G[3 * k]], in[G[3 * k + 1]]);
            uint64_t *t = in; in = o; o = t;
        }
        for (int c = 0; c < C; c++) {
            uint64_t *pl = cnt + (size_t)c * planes;
            for (int q = 0; q < planes; q++) pl[q] = 0;
            for (int k = 0; k < g; k++) {
                uint64_t carry = in[c * g + k];
                for (int q = 0; q < planes && carry; q++) { uint64_t t = pl[q] & carry; pl[q] ^= carry; carry = t; }
            }
        }
        for (int i = 0; i < lanes; i++) {
            int best = 0, bv = -1;
            for (int c = 0; c < C; c++) {
                const uint64_t *pl = cnt + (size_t)c * planes;
                int v = 0;
                for (int q = 0; q < planes; q++) v |= (int)((pl[q] >> i) & 1) << q;
                if (v > bv) { bv = v; best = c; }
            }
            out[base + i] = (uint8_t)best;
        }
    }
    fwrite(out, 1, N, stdout);
    return 0;
}
