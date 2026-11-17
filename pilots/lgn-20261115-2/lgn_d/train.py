"""Logic-gate network (Petersen et al. 2022, arXiv 2210.08277) on thermometer bits, compiled to one circuit.

python3 train.py TRAIN_CSV OUT_WEIGHTS

Inputs: each pixel v in 0..16 becomes 4 bits [v >= t, t = 2, 6, 10, 14] (256 bits; LILogicNet, arXiv
2511.12340, CIFAR-10: 7 thresholds beat 4 in 13 of 14 pairs, by 0.02-1.12 points; on this task's inner split, 4
cost nothing measurable).
Model: layers of 2-input gates. Wiring is learned (Top-K, after LILogicNet): each gate input softly chooses among
K = 8 fixed random candidates from the previous layer (a softmax over K logits), and keeps the most likely one
after training. Each gate also learns a softmax over the 16 binary ops,
relaxed to real values in [0,1] (a AND b = ab, a XOR b = a + b - 2ab, ...). The last layer is split into 10
equal groups; a class score is its group's sum / TAU; softmax cross-entropy, Adam (lr 0.01, minibatch 64).
After training each gate keeps its most likely op. No hidden-layer weights remain: inference is bitwise logic.

Hyperparameters (WIDTHS, TAU, K, THRESH, EPOCHS, SEED) are fixed. They were chosen in the 2026-11-15 pilot on an inner
split of the first 75% of the training file (first 2,150 rows to train, next 717 to judge), so the dev split
(last 25%) played no part in choosing them.

Compiled form (what infer.c runs): one circuit of 2-input gates from the same 16 ops. It takes the 5 bit-planes
of each pixel, builds the threshold bits v >= t from them, runs the learned gates, counts each class's group
with a full-adder tree, and takes the argmax (first maximum wins, as numpy's argmax) as 4 output bits. The
circuit is constant-folded, hashed, has its NOT gates folded into their readers, is pruned of dead gates,
sorted by depth then op, and given reusable slots.
train.py checks that the compiled circuit predicts exactly what the trained network does on every training row.

Determinism: training uses only elementwise + - * / max, indexing, explicit sequential sums and np.add.at
(sequential), and exp from + * / (no libm, no BLAS), as in mlp16's train.py.

weights.bin (uint16 little-endian): n_slots, n_groups, per group: op, count, count x (a, b, out); then 4 slots.
"""
import sys
import numpy as np

# 16 binary ops as coefficients over [1, a, b, ab]
M = np.array([
    [0, 0, 0, 0],   # 0  false
    [0, 0, 0, 1],   # 1  a and b
    [0, 1, 0, -1],  # 2  a and not b
    [0, 1, 0, 0],   # 3  a
    [0, 0, 1, -1],  # 4  not a and b
    [0, 0, 1, 0],   # 5  b
    [0, 1, 1, -2],  # 6  xor
    [0, 1, 1, -1],  # 7  or
    [1, -1, -1, 1],  # 8  nor
    [1, -1, -1, 2],  # 9  xnor
    [1, 0, -1, 0],  # 10 not b
    [1, 0, -1, 1],  # 11 a or not b
    [1, -1, 0, 0],  # 12 not a
    [1, -1, 0, 1],  # 13 not a or b
    [1, 0, 0, -1],  # 14 nand
    [1, 0, 0, 0],   # 15 true
], dtype=np.float64)


def exp_neg(x):
    y = np.maximum(x, -60.0) * (1.0 / 256.0)
    p = np.ones_like(y)
    for k in range(10, 0, -1):
        p = 1.0 + p * y * (1.0 / k)
    for _ in range(8):
        p = p * p
    return p


def rowsum(A):  # sum over axis 0, sequential
    acc = A[0].copy()
    for i in range(1, A.shape[0]):
        acc += A[i]
    return acc


def lastsum(A):  # sum over last axis, sequential
    acc = A[..., 0].copy()
    for k in range(1, A.shape[-1]):
        acc += A[..., k]
    return acc


def softmax16(w):
    e = exp_neg(w - w.max(axis=1, keepdims=True))
    return e / lastsum(e)[:, None]


def coef(p):  # (N,16) @ M (16,4) sequentially
    c = p[:, 0:1] * M[0]
    for k in range(1, 16):
        c = c + p[:, k:k + 1] * M[k]
    return c


def thermo(V, T):
    return (V[:, :, None] >= np.asarray(T)[None, None, :]).reshape(len(V), -1)


def make_net(rng, n_in, widths):
    conns, logits = [], []
    prev = n_in
    for w in widths:
        # each previous unit used at least once where possible: a random permutation, then random fill
        need = 2 * w
        base = np.concatenate([rng.permutation(prev) for _ in range(need // prev + 1)])[:need]
        a, b = base[:w].copy(), base[w:].copy()
        same = a == b
        b[same] = (b[same] + 1) % prev
        conns.append((a, b))
        logits.append(rng.standard_normal((w, 16)))
        prev = w
    return conns, logits


def forward(X, conns, logits, keep=False):
    acts, cs = [X], []
    x = X
    for (ia, ib), w in zip(conns, logits):
        c = coef(softmax16(w))
        a, b = x[:, ia], x[:, ib]
        x = c[:, 0] + c[:, 1] * a + c[:, 2] * b + c[:, 3] * (a * b)
        acts.append(x); cs.append(c)
    return acts, cs


def group_scores(x, ncls, tau):
    g = x.shape[1] // ncls
    return lastsum(x.reshape(len(x), ncls, g)) / tau


def train(X, y, widths, epochs, seed, tau, lr=0.01, bs=64):
    rng = np.random.default_rng(seed)
    conns, logits = make_net(rng, X.shape[1], widths)
    n = len(y)
    Y = np.zeros((n, 10)); Y[np.arange(n), y] = 1.0
    Mo = [np.zeros_like(w) for w in logits]; So = [np.zeros_like(w) for w in logits]
    p1 = p2 = 1.0
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, bs):
            idx = order[s:s + bs]
            m = len(idx)
            acts, cs = forward(X[idx], conns, logits)
            o = group_scores(acts[-1], 10, tau)
            o = o - o.max(axis=1, keepdims=True)
            e = exp_neg(o)
            pr = e / lastsum(e)[:, None]
            go = (pr - Y[idx]) * (1.0 / m)
            g = np.repeat(go / tau, acts[-1].shape[1] // 10, axis=1)
            p1 *= 0.9; p2 *= 0.999
            c1, c2 = 1 - p1, 1 - p2
            for L in range(len(logits) - 1, -1, -1):
                ia, ib = conns[L]
                xin = acts[L]
                a, b = xin[:, ia], xin[:, ib]
                c = cs[L]
                ab = a * b
                gc = np.stack([rowsum(g), rowsum(g * a), rowsum(g * b), rowsum(g * ab)], axis=1)
                if L > 0:
                    ga = g * (c[:, 1] + c[:, 3] * b)
                    gb = g * (c[:, 2] + c[:, 3] * a)
                    gx = np.zeros_like(xin)
                    np.add.at(gx, (slice(None), ia), ga)
                    np.add.at(gx, (slice(None), ib), gb)
                p = softmax16(logits[L])
                gp = gc[:, 0:1] * M[:, 0] + gc[:, 1:2] * M[:, 1] + gc[:, 2:3] * M[:, 2] + gc[:, 3:4] * M[:, 3]
                gw = p * (gp - lastsum(p * gp)[:, None])
                Mo[L] *= 0.9; Mo[L] += 0.1 * gw
                So[L] *= 0.999; So[L] += 0.001 * (gw * gw)
                logits[L] -= lr * (Mo[L] / c1) / (np.sqrt(So[L] / c2) + 1e-8)
                if L > 0:
                    g = gx
    return conns, [w.argmax(axis=1) for w in logits]


def OPS(k, a, b):
    return [np.zeros_like(a), a & b, a & ~b, a, ~a & b, b, a ^ b, a | b, ~(a | b), ~(a ^ b), ~b, a | ~b, ~a,
            ~a | b, ~(a & b), np.ones_like(a)][k]


def predict_discrete(X, conns, gates, tau=1.0):
    x = X.astype(bool)
    for (ia, ib), gk in zip(conns, gates):
        a, b = x[:, ia], x[:, ib]
        out = np.zeros((len(x), len(gk)), dtype=bool)
        for k in range(16):
            sel = gk == k
            if sel.any():
                out[:, sel] = OPS(k, a[:, sel], b[:, sel])
        x = out
    g = x.shape[1] // 10
    return x.reshape(len(x), 10, g).sum(axis=2).argmax(axis=1)


OPF = [lambda a, b: 0, lambda a, b: a & b, lambda a, b: a & (1 - b), lambda a, b: a, lambda a, b: (1 - a) & b,
       lambda a, b: b, lambda a, b: a ^ b, lambda a, b: a | b, lambda a, b: 1 - (a | b), lambda a, b: 1 - (a ^ b),
       lambda a, b: 1 - b, lambda a, b: a | (1 - b), lambda a, b: 1 - a, lambda a, b: (1 - a) | b,
       lambda a, b: 1 - (a & b), lambda a, b: 1]
TT = [tuple(OPF[k](a, b) for a in (0, 1) for b in (0, 1)) for k in range(16)]  # truth tables (00,01,10,11)
BY_TT = {t: k for k, t in enumerate(TT)}
AND, XOR, OR, NOT_A = 1, 6, 7, 12
NPLANE = 5
C0, C1 = ('c', 0), ('c', 1)


class Circuit:
    def __init__(self):
        self.gates = []    # (op, a, b) ; signal ('g', i)
        self.hash = {}

    def g(self, op, a, b):
        t = TT[op]
        if a[0] == 'c' and b[0] == 'c':
            return ('c', t[2 * a[1] + b[1]])
        if a[0] == 'c':  # function of b only
            y0, y1 = t[2 * a[1]], t[2 * a[1] + 1]
            return self._unary(y0, y1, b)
        if b[0] == 'c':
            y0, y1 = t[b[1]], t[2 + b[1]]
            return self._unary(y0, y1, a)
        if a == b:
            return self._unary(t[0], t[3], a)
        # depends on one input only?
        if t[0] == t[1] and t[2] == t[3]:
            return self._unary(t[0], t[2], a)
        if t[0] == t[2] and t[1] == t[3]:
            return self._unary(t[0], t[1], b)
        if a > b:  # canonical order: swap inputs, permute the truth table
            a, b = b, a
            op = BY_TT[(t[0], t[2], t[1], t[3])]
        key = (op, a, b)
        s = self.hash.get(key)
        if s is None:
            s = ('g', len(self.gates))
            self.gates.append(key)
            self.hash[key] = s
        return s

    def _unary(self, y0, y1, x):
        if y0 == y1:
            return ('c', y0)
        if (y0, y1) == (0, 1):
            return x
        key = (NOT_A, x, x)
        s = self.hash.get(key)
        if s is None:
            s = ('g', len(self.gates)); self.gates.append(key); self.hash[key] = s
        return s

    def NOT(self, x): return self.g(NOT_A, x, x) if x[0] != 'c' else ('c', 1 - x[1])
    def AND(self, a, b): return self.g(AND, a, b)
    def OR(self, a, b): return self.g(OR, a, b)
    def XOR(self, a, b): return self.g(XOR, a, b)

    def mux(self, s, x, y):  # s ? x : y
        return self.OR(self.AND(s, x), self.g(4, s, y))  # op 4 = (not a) and b


def ge_const(C, planes, t):
    """v >= t for v in 0..16 held in 5 bit-planes (LSB first)."""
    if t <= 0:
        return C1
    if t >= 17:
        return C0
    if t == 16:
        return planes[4]
    r = C1  # compare low 4 bits against t (t < 16), msb first, then OR with bit 4
    for i in range(4):  # from LSB up: ge over bits [0..i]
        if (t >> i) & 1:
            r = C.AND(planes[i], r)
        else:
            r = C.OR(planes[i], r)
    return C.OR(planes[4], r)


def popcount(C, bits):
    """Full-adder tree; returns bits of the count, LSB first."""
    cols = [list(bits)]
    i = 0
    while i < len(cols):
        col = cols[i]
        while len(col) >= 3:
            a, b, c = col.pop(), col.pop(), col.pop()
            ab = C.XOR(a, b)
            s = C.XOR(ab, c)
            cy = C.OR(C.AND(a, b), C.AND(c, ab))
            col.insert(0, s)
            if i + 1 == len(cols): cols.append([])
            cols[i + 1].append(cy)
        if len(col) == 2:
            a, b = col.pop(), col.pop()
            s, cy = C.XOR(a, b), C.AND(a, b)
            col.append(s)
            if i + 1 == len(cols): cols.append([])
            cols[i + 1].append(cy)
        i += 1
    return [c[0] if c else C0 for c in cols]


def greater(C, x, y):
    """x > y, both LSB-first lists of equal length."""
    gt = C0
    for i in range(len(x)):  # LSB to MSB: gt_i = (x_i & ~y_i) | (~(x_i ^ y_i) & gt_{i-1})
        gt = C.OR(C.g(2, x[i], y[i]), C.AND(C.g(9, x[i], y[i]), gt))
    return gt


def build(gate_conns, gate_ops, feats, ncls=10):
    """feats: list of (pixel, threshold) for each network input. Returns Circuit and the 4 output signals."""
    C = Circuit()
    planes = [[('s', p * NPLANE + k) for k in range(NPLANE)] for p in range(64)]
    x = [ge_const(C, planes[p], t) for p, t in feats]
    for (ia, ib), ops in zip(gate_conns, gate_ops):
        x = [C.g(int(k), x[a], x[b]) for a, b, k in zip(ia, ib, ops)]
    G = len(x) // ncls
    counts = [popcount(C, x[c * G:(c + 1) * G]) for c in range(ncls)]
    w = max(len(c) for c in counts)
    counts = [c + [C0] * (w - len(c)) for c in counts]
    best, idx = counts[0], [C0] * 4
    for c in range(1, ncls):
        s = greater(C, counts[c], best)
        best = [C.mux(s, a, b) for a, b in zip(counts[c], best)]
        idx = [C.mux(s, ('c', (c >> k) & 1), idx[k]) for k in range(4)]
    return C, idx


def absorb_not(C):
    """Fold every NOT into the gates that read it: a gate reading NOT(x) on an input becomes the gate with that
    input complemented, reading x. Exact (same truth function); NOTs no longer read by any gate fall to DCE."""
    def neg_src(s):
        if s[0] == 'g':
            op, a, b = C.gates[s[1]]
            if op == NOT_A and a == b:
                return a
        return None
    for i, (op, a, b) in enumerate(C.gates):
        t = TT[op]
        x = neg_src(a)
        if x is not None:
            a, t = x, (t[2], t[3], t[0], t[1])
        y = neg_src(b)
        if y is not None:
            b, t = y, (t[1], t[0], t[3], t[2])
        C.gates[i] = (BY_TT[t], a, b)


def finalize(C, outs, n_in):
    """DCE, levels, group by op within level, slot allocation. Returns (n_slots, groups, out_slots, n_gates)."""
    live = [False] * len(C.gates)
    stack = [o[1] for o in outs if o[0] == 'g']
    while stack:
        i = stack.pop()
        if live[i]: continue
        live[i] = True
        _, a, b = C.gates[i]
        for s in (a, b):
            if s[0] == 'g' and not live[s[1]]: stack.append(s[1])
    level = [0] * len(C.gates)
    for i, (op, a, b) in enumerate(C.gates):  # gates are created in topological order
        if live[i]:
            level[i] = 1 + max(level[s[1]] if s[0] == 'g' else 0 for s in (a, b))
    order = sorted((i for i in range(len(C.gates)) if live[i]), key=lambda i: (level[i], C.gates[i][0], i))
    # last use (position in order) of each gate signal
    pos = {i: n for n, i in enumerate(order)}
    last = {}
    for n, i in enumerate(order):
        _, a, b = C.gates[i]
        for s in (a, b):
            if s[0] == 'g': last[s[1]] = n
    for o in outs:
        if o[0] == 'g': last[o[1]] = len(order)
    fixed = n_in + 2  # input planes, const0, const1
    free, nslots, slot = [], fixed, {}
    expire = {}
    for i, n in last.items(): expire.setdefault(n, []).append(i)

    def sid(s):
        if s[0] == 's': return s[1]
        if s[0] == 'c': return n_in + s[1]
        return slot[s[1]]

    rows = []
    for n, i in enumerate(order):
        op, a, b = C.gates[i]
        sa, sb = sid(a), sid(b)
        for j in expire.get(n, []):  # inputs whose last use is here: free before allocating the output
            free.append(slot[j])
        if free:
            free.sort(reverse=True); o = free.pop()
        else:
            o = nslots; nslots += 1
        slot[i] = o
        rows.append((level[i], op, sa, sb, o))
    groups = []
    for lv, op, sa, sb, o in rows:
        if groups and groups[-1][0] == (lv, op):
            groups[-1][1].append((sa, sb, o))
        else:
            groups.append(((lv, op), [(sa, sb, o)]))
    return nslots, [(k[1], r) for k, r in groups], [sid(o) for o in outs], len(order)


def simulate(n_slots, groups, out_slots, V, n_in):
    """Run the finalized circuit on images V (n,64) in numpy, in slot form; returns predicted digits."""
    S = np.zeros((n_slots, len(V)), dtype=np.uint8)
    for k in range(NPLANE):
        S[np.arange(64) * NPLANE + k] = (V.T >> k) & 1
    S[n_in + 1] = 1
    for op, rows in groups:
        for sa, sb, o in rows:
            a, b = S[sa], S[sb]
            S[o] = OPF[op](a, b) if op not in (0, 15) else np.full_like(a, op == 15)
    return sum(S[out_slots[k]].astype(np.int64) << k for k in range(4))


def softmaxK(c):
    e = exp_neg(c - c.max(axis=1, keepdims=True))
    return e / lastsum(e)[:, None]


def make_topk(rng, n_in, widths, K):
    cand, logits = [], []
    prev = n_in
    for w in widths:
        ca = rng.integers(0, prev, size=(w, K)); cb = rng.integers(0, prev, size=(w, K))
        cand.append((ca, cb))
        logits.append([rng.standard_normal((w, 16)), np.zeros((w, K)), np.zeros((w, K))])
        prev = w
    return cand, logits


def mix(x, idx, p):  # sum_k p[:,k] * x[:, idx[:,k]], sequential in k
    acc = x[:, idx[:, 0]] * p[:, 0]
    for k in range(1, idx.shape[1]):
        acc = acc + x[:, idx[:, k]] * p[:, k]
    return acc


def train_topk(X, y, widths, epochs, seed, tau, K, lr=0.01, bs=64):
    rng = np.random.default_rng(seed)
    cand, P = make_topk(rng, X.shape[1], widths, K)
    n = len(y)
    Y = np.zeros((n, 10)); Y[np.arange(n), y] = 1.0
    Mo = [[np.zeros_like(t) for t in L] for L in P]; So = [[np.zeros_like(t) for t in L] for L in P]
    p1 = p2 = 1.0
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, bs):
            idx = order[s:s + bs]; m = len(idx)
            xs, cache = [X[idx]], []
            for (ca, cb), (w, la, lb) in zip(cand, P):
                pa, pb = softmaxK(la), softmaxK(lb)
                a, b = mix(xs[-1], ca, pa), mix(xs[-1], cb, pb)
                c = coef(softmax16(w))
                xs.append(c[:, 0] + c[:, 1] * a + c[:, 2] * b + c[:, 3] * (a * b))
                cache.append((pa, pb, a, b, c))
            o = group_scores(xs[-1], 10, tau)
            o = o - o.max(axis=1, keepdims=True)
            e = exp_neg(o); pr = e / lastsum(e)[:, None]
            go = (pr - Y[idx]) * (1.0 / m)
            g = np.repeat(go / tau, xs[-1].shape[1] // 10, axis=1)
            p1 *= 0.9; p2 *= 0.999; c1, c2 = 1 - p1, 1 - p2
            for L in range(len(P) - 1, -1, -1):
                (ca, cb), (pa, pb, a, b, c) = cand[L], cache[L]
                xin = xs[L]
                gc = np.stack([rowsum(g), rowsum(g * a), rowsum(g * b), rowsum(g * (a * b))], axis=1)
                ga = g * (c[:, 1] + c[:, 3] * b); gb = g * (c[:, 2] + c[:, 3] * a)
                grads = []
                p = softmax16(P[L][0])
                gp = gc[:, 0:1] * M[:, 0] + gc[:, 1:2] * M[:, 1] + gc[:, 2:3] * M[:, 2] + gc[:, 3:4] * M[:, 3]
                grads.append(p * (gp - lastsum(p * gp)[:, None]))
                for gin, ci, pi in ((ga, ca, pa), (gb, cb, pb)):
                    gk = np.stack([rowsum(gin * xin[:, ci[:, k]]) for k in range(K)], axis=1)
                    grads.append(pi * (gk - lastsum(pi * gk)[:, None]))
                if L > 0:
                    gx = np.zeros_like(xin)
                    for gin, ci, pi in ((ga, ca, pa), (gb, cb, pb)):
                        for k in range(K):
                            np.add.at(gx, (slice(None), ci[:, k]), gin * pi[:, k])
                for j, gw in enumerate(grads):
                    Mo[L][j] *= 0.9; Mo[L][j] += 0.1 * gw
                    So[L][j] *= 0.999; So[L][j] += 0.001 * (gw * gw)
                    P[L][j] -= lr * (Mo[L][j] / c1) / (np.sqrt(So[L][j] / c2) + 1e-8)
                if L > 0:
                    g = gx
    conns = [(ca[np.arange(len(ca)), la.argmax(axis=1)], cb[np.arange(len(cb)), lb.argmax(axis=1)])
             for (ca, cb), (w, la, lb) in zip(cand, P)]
    return conns, [w.argmax(axis=1) for w, la, lb in P]

WIDTHS = (2000, 1000, 500)
TAU = 4.0
EPOCHS = 100
SEED = 1
THRESH = [2, 6, 10, 14]
K = 8
NIN = 64 * NPLANE


def main(train_csv, out):
    d = np.loadtxt(train_csv, delimiter=",", dtype=np.int64)
    V, y = d[:, :64], d[:, 64]
    X = thermo(V, THRESH).astype(np.float64)
    conns, gates = train_topk(X, y, WIDTHS, EPOCHS, SEED, TAU, K)
    feats = [(p, t) for p in range(64) for t in THRESH]
    C, outs = build(conns, gates, feats)
    absorb_not(C)
    ns, groups, out_slots, ng = finalize(C, outs, NIN)
    ref = predict_discrete(X, conns, gates)
    got = simulate(ns, groups, out_slots, V, NIN)
    assert (ref == got).all(), "compiled circuit disagrees with the network"
    print(f"gates {ng}, slots {ns}, groups {len(groups)}, training accuracy {(ref == y).mean():.4f}", file=sys.stderr)
    words = [ns, len(groups)]
    for op, rows in groups:
        assert len(rows) < 65536
        words += [op, len(rows)]
        for r in rows:
            words += list(r)
    words += out_slots
    assert max(words) < 65536
    with open(out, "wb") as f:
        f.write(np.array(words, dtype="<u2").tobytes())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
