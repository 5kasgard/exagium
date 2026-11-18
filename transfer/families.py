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
