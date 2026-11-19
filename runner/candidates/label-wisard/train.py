"""WiSARD n-tuple classifier (weightless: lookup tables only) for label tasks with fixed-size byte items (queue P3 step
1; the lookup-table family for Q54). python3 -s -P train.py card.json train.in train.key weights.bin  (deterministic)
Bits per column, by the card's types, T per column: ordinal, a thermometer (bit t = x > threshold t, thresholds at the
training quantiles (t+1)/(T+1), duplicates dropped); categorical, one-hot over the T most frequent training values
(ties to the lower value). The bits are shuffled once (seed 1) and cut into tuples of n bits (the last tuple wraps to
the start). Each class keeps, per tuple, a count per address (uint16, saturating). Inference with bleaching: b = 1;
while more than one class has the top response (the number of tuples whose count at the item's address is >= b),
raise b; stop when the top response would be 0; the lowest tied class wins.
Tuning: the last 20% of train.in (file order); (n, T) in ((8, 8), (8, 16), (12, 8), (12, 16)) chosen on it (first best
wins), then the tables are filled from all of train. Weights: int32 magic 0x31445357, d, T, n, ntup, C;
uint8 nv[d]; uint8 kind[d] (0 ordinal, 1 categorical); uint8 val[d][T]; uint16 map[ntup][n]; uint16 cnt[C][ntup][2^n]."""
import json, struct, sys

import numpy as np

GRID = ((8, 8), (8, 16), (12, 8), (12, 16))
TUNE = 0.2
SEED = 1


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def encoder(X, types, T):
    nv, kind, val = [], [], np.zeros((len(types), T), np.uint8)
    for j, t in enumerate(types):
        col = X[:, j]
        if t == "ordinal":
            q = np.unique(np.floor(np.quantile(col.astype(np.float64), [(i + 1) / (T + 1) for i in range(T)])).astype(int))
            v, k = q[q < 255], 0
        else:
            counts = np.bincount(col, minlength=256)
            order = np.lexsort((np.arange(256), -counts))           # most frequent first, ties to the lower value
            v = order[:T][counts[order[:T]] > 0]
            k = 1
        nv.append(len(v)); kind.append(k); val[j, :len(v)] = v
    return np.array(nv, np.uint8), np.array(kind, np.uint8), val


def bits(X, nv, kind, val):
    cols = []
    for j in range(X.shape[1]):
        x = X[:, j][:, None].astype(np.int64)
        v = val[j, :nv[j]][None, :].astype(np.int64)
        b = (x > v) if kind[j] == 0 else (x == v)
        pad = np.zeros((len(X), val.shape[1] - nv[j]), bool)   # unused slots never fire
        cols.append(np.concatenate([b, pad], axis=1))
    return np.concatenate(cols, axis=1)                         # (m, d*T)


def mapping(B, n):
    perm = np.random.default_rng(SEED).permutation(B)
    ntup = -(-B // n)
    return np.resize(perm, ntup * n).reshape(ntup, n).astype(np.int64)


def addresses(Bits, mp):
    w = (1 << np.arange(mp.shape[1], dtype=np.int64))
    return (Bits[:, mp].astype(np.int64) * w).sum(axis=2)       # (m, ntup)


def fill(A, y, C, n):
    ntup = A.shape[1]
    cnt = np.zeros((C, ntup, 1 << n), np.int64)
    np.add.at(cnt, (np.repeat(y, ntup), np.tile(np.arange(ntup), len(y)), A.ravel()), 1)
    return np.minimum(cnt, 65535).astype(np.uint16)


def predict(cnt, A):
    """Bleaching, exactly as infer.c: b = 1 sets the winners (the classes with the top response); while more than one
    class wins and the top response is above 0, raise b; a raise whose top response is 0 keeps the previous winners."""
    C, ntup, _ = cnt.shape
    G = cnt[:, np.arange(ntup)[None, :], A].transpose(1, 0, 2)  # (m, C, ntup) counts at each item's addresses
    r = (G >= 1).sum(axis=2)
    top = r.max(axis=1)
    W = r == top[:, None]
    open_ = (W.sum(axis=1) > 1) & (top > 0)
    b = 1
    while open_.any():
        b += 1
        idx = np.flatnonzero(open_)
        r = (G[idx] >= b).sum(axis=2)
        top = r.max(axis=1)
        stop = top == 0
        keep = idx[~stop]
        W[keep] = (r == top[:, None])[~stop]
        open_[idx[stop]] = False
        open_[keep] = W[keep].sum(axis=1) > 1
    return W.argmax(axis=1)                                      # the lowest tied class


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C, types = card["classes"], card["input"]["types"]
    cut = len(X) - int(len(X) * TUNE)
    best, best_acc = GRID[0], -1.0
    for n, T in GRID:
        nv, kind, val = encoder(X[:cut], types, T)
        mp = mapping(d * T, n)
        cnt = fill(addresses(bits(X[:cut], nv, kind, val), mp), y[:cut], C, n)
        acc = float((predict(cnt, addresses(bits(X[cut:], nv, kind, val), mp)) == y[cut:]).mean())
        print(f"n={n} T={T} tuning accuracy {acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best, best_acc = (n, T), acc
    n, T = best
    nv, kind, val = encoder(X, types, T)
    mp = mapping(d * T, n)
    cnt = fill(addresses(bits(X, nv, kind, val), mp), y, C, n)
    out = struct.pack("<6i", 0x31445357, d, T, n, mp.shape[0], C) + nv.tobytes() + kind.tobytes() + val.tobytes() \
        + mp.astype("<u2").tobytes() + cnt.astype("<u2").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
