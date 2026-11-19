"""Gradient-boosted trees for label tasks with fixed-size byte items (queue P3 step 1; outside-strength baseline).
python3 -s -P train.py card.json train.in train.key weights.bin   (deterministic, single thread, float64)
One multi-output tree per round on softmax gradients (second order: g = p - y, h = p(1 - p)); a split sends x <= thr
left, on the byte value of one column (categorical columns are split on their codes, a stated simplification); gain is
sum over classes of GL^2/(HL+L2) + GR^2/(HR+L2) - G^2/(H+L2); first best wins (column order, then threshold). Leaf
values -LR * G/(H+L2) per class. Start: log class frequencies (+1 smoothing).
Tuning: the last 20% of train.in (file order). Depth in (2, 4, 6), learning rate 0.1, up to MAX_ROUNDS rounds; the
setting and round count with the lowest tuning log loss are chosen (first best wins), then the model is refitted on all
of train with them. Weights: int32 magic 0x31544447, C, ntrees, nnodes, nleaves; float32 base[C];
per node: int16 feature (-1 = leaf), uint8 thr, uint8 pad, int32 left, int32 right (for a leaf, left = leaf index);
tree roots int32[ntrees]; float32 leaf[nleaves][C]."""
import json, struct, sys

import numpy as np

DEPTHS = (2, 4, 6)
LR, L2, MIN_H, MAX_ROUNDS = 0.1, 1.0, 1e-3, 300
TUNE = 0.2


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def softmax(F):
    F = F - F.max(axis=1, keepdims=True)
    P = np.exp(F)
    return P / P.sum(axis=1, keepdims=True)


class Model:
    def __init__(self, C):
        self.C, self.nodes, self.roots, self.leaves = C, [], [], []

    def tree(self, X, order, G, H, depth, Xt):
        """Grow one tree on all rows of X; returns the leaf values for X's rows and for Xt's rows (routed alongside)."""
        n, d = X.shape
        upd, updt = np.zeros((n, self.C)), np.zeros((len(Xt), self.C))
        self.roots.append(len(self.nodes))
        stack = [(np.ones(n, bool), np.ones(len(Xt), bool), depth, self._new())]
        while stack:
            mask, maskt, dleft, node = stack.pop()
            Gs, Hs = G[mask].sum(axis=0), H[mask].sum(axis=0)
            best = None
            if dleft > 0 and mask.sum() >= 2:
                base = (Gs ** 2 / (Hs + L2)).sum()
                for j in range(d):
                    o = order[j][mask[order[j]]]                    # this node's rows, sorted by column j
                    v = X[o, j]
                    cut = np.flatnonzero(v[:-1] < v[1:])            # split after position p: left = o[:p+1]
                    if not len(cut):
                        continue
                    GL, HL = np.cumsum(G[o], axis=0)[cut], np.cumsum(H[o], axis=0)[cut]
                    GR, HR = Gs - GL, Hs - HL
                    ok = (HL.sum(axis=1) >= MIN_H) & (HR.sum(axis=1) >= MIN_H)
                    gain = np.where(ok, (GL ** 2 / (HL + L2)).sum(axis=1) + (GR ** 2 / (HR + L2)).sum(axis=1) - base, -np.inf)
                    k = int(np.argmax(gain))
                    if gain[k] > 0 and (best is None or gain[k] > best[0]):
                        best = (gain[k], j, int(v[cut[k]]))
            if best is None:
                val = -LR * Gs / (Hs + L2)
                self.nodes[node] = (-1, 0, len(self.leaves), 0)
                self.leaves.append(val)
                upd[mask] = val
                updt[maskt] = val
                continue
            _, j, thr = best
            go_left, go_left_t = X[:, j] <= thr, Xt[:, j] <= thr
            left, right = self._new(), self._new()
            self.nodes[node] = (j, thr, left, right)
            stack.append((mask & ~go_left, maskt & ~go_left_t, dleft - 1, right))
            stack.append((mask & go_left, maskt & go_left_t, dleft - 1, left))
        return upd, updt

    def _new(self):
        self.nodes.append(None)
        return len(self.nodes) - 1


def fit(X, y, C, depth, rounds, Xt=None, yt=None):
    n, d = X.shape
    order = [np.argsort(X[:, j], kind="stable") for j in range(d)]
    Y = np.eye(C)[y]
    base = np.log((np.bincount(y, minlength=C) + 1.0) / (n + C))
    F = np.tile(base, (n, 1))
    m = Model(C)
    Xt = X[:0] if Xt is None else Xt
    Ft = np.tile(base, (len(Xt), 1))
    losses = []
    for _ in range(rounds):
        P = softmax(F)
        u, ut = m.tree(X, order, P - Y, P * (1 - P), depth, Xt)
        F += u
        Ft += ut
        if yt is not None:
            Pt = softmax(Ft)
            losses.append(float(-np.log(Pt[np.arange(len(yt)), yt] + 1e-12).mean()))
    return m, base, losses


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    cut = len(X) - int(len(X) * TUNE)
    best = None
    for depth in DEPTHS:
        _, _, losses = fit(X[:cut], y[:cut], C, depth, MAX_ROUNDS, X[cut:], y[cut:])
        r = int(np.argmin(losses))
        print(f"depth={depth} best rounds {r + 1} tuning log loss {losses[r]:.4f}", file=sys.stderr)
        if best is None or losses[r] < best[0]:
            best = (losses[r], depth, r + 1)
    _, depth, rounds = best
    m, base, _ = fit(X, y, C, depth, rounds)
    out = struct.pack("<5i", 0x31544447, C, len(m.roots), len(m.nodes), len(m.leaves)) + base.astype("<f4").tobytes()
    for f, thr, l, r in m.nodes:
        out += struct.pack("<hBBii", f, thr, 0, l, r)
    out += np.array(m.roots, "<i4").tobytes() + np.array(m.leaves, "<f4").reshape(-1, C).tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
