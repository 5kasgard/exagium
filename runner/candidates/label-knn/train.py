"""k nearest neighbours for any label task with fixed-size byte items (queue P3 step 1).
python3 -s -P train.py card.json train.in train.key weights.bin   (deterministic, single thread)
Distance: the sum over columns of an integer cost table, by the card's per-byte types: ordinal |a-b| scaled by the
column's training standard deviation (round(1000 * |a-b| / (sd + 1))), categorical 0 if equal else 1000. Integer
costs make the tuning here and the C inference agree exactly. Neighbours: the k smallest distances, ties to the lower
training index. Vote: the most frequent class; a tie goes to the tied class met first in neighbour order.
Tuning: the last 20% of train.in (file order) is the tuning split; k in (1, 3, 5, 9, 15, 25, 41, 61) is chosen on it
with the first 80% as the reference set (first best wins), then the weights hold all of train.
Weights: int32 magic 0x314E4E4B, d, n, C, k; uint16 cost[d][256][256]; uint8 X[n][d]; uint8 y[n]."""
import json, struct, sys

import numpy as np

KS = (1, 3, 5, 9, 15, 25, 41, 61)
TUNE = 0.2
CHUNK = 64


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def costs(X, types):
    a = np.arange(256)
    out = []
    for j, t in enumerate(types):
        if t == "ordinal":
            sd = float(X[:, j].astype(np.float64).std())
            c = np.rint(1000.0 * np.abs(a[:, None] - a[None, :]) / (sd + 1.0))
        else:
            c = np.where(a[:, None] == a[None, :], 0, 1000)
        out.append(np.minimum(c, 65535).astype(np.uint16))
    return np.stack(out)


def vote(nb_labels, C):
    """nb_labels: (m, k) labels in neighbour order -> predicted class per row (ties: first met in order)."""
    m, k = nb_labels.shape
    counts = np.zeros((m, C), np.int64)
    np.add.at(counts, (np.repeat(np.arange(m), k), nb_labels.ravel()), 1)
    best = counts.max(axis=1)
    tied = counts == best[:, None]                       # (m, C)
    first = np.full((m, C), k, np.int64)                 # position where each class first appears
    for p in range(k - 1, -1, -1):
        first[np.arange(m), nb_labels[:, p]] = p
    first = np.where(tied, first, k + 1)
    return first.argmin(axis=1)


def predict(cost, ref_X, ref_y, Q, k, C):
    d = ref_X.shape[1]
    out = []
    for s in range(0, len(Q), CHUNK):
        q = Q[s:s + CHUNK]
        D = np.zeros((len(q), len(ref_X)), np.int64)
        for j in range(d):
            D += cost[j][q[:, j][:, None], ref_X[:, j][None, :]]
        nb = np.argsort(D, axis=1, kind="stable")[:, :k]
        out.append(vote(ref_y[nb], C))
    return np.concatenate(out)


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    cut = len(X) - int(len(X) * TUNE)
    cost = costs(X[:cut], card["input"]["types"])
    best_k, best_acc = KS[0], -1.0
    for k in KS:
        acc = float((predict(cost, X[:cut], y[:cut], X[cut:], k, C) == y[cut:]).mean())
        print(f"k={k} tuning accuracy {acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_k, best_acc = k, acc
    cost = costs(X, card["input"]["types"])
    out = struct.pack("<5i", 0x314E4E4B, d, len(X), C, best_k) + cost.astype("<u2").tobytes() + X.tobytes() \
        + y.astype(np.uint8).tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
