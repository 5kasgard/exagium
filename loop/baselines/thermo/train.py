"""Baseline 2: thermometer encoding plus linear.

Each pixel v in 0..16 becomes 16 bits [v >= t for t = 1..16] (1,024 features), then multinomial
logistic regression, trained exactly like baseline 1 (same optimiser, iterations, L2 grid, holdout).
python3 train.py TRAIN_CSV OUT_WEIGHTS

Because the encoding is a thermometer, a linear model over it equals a lookup table:
score[c] = b[c] + sum_p T[p][v_p][c], with T[p][v][c] = sum_{t<=v} W[p,t,c]. The weights file holds
that table, float32 little-endian, T[64][17][10] then b[10]: the honest cheap way to run this baseline.
"""
import sys
import numpy as np

LAMBDAS = (1e-4, 1e-3, 1e-2)
ITERS = 3000


def features(x):
    t = np.arange(1, 17)
    return (x[:, :, None] >= t).reshape(len(x), 64 * 16).astype(np.float64)


def fit(X, y, lam):
    n, d = X.shape
    Y = np.zeros((n, 10)); Y[np.arange(n), y] = 1.0
    W = np.zeros((d, 10)); b = np.zeros(10)
    vW = np.zeros_like(W); vb = np.zeros_like(b)
    lr, mom = 0.5, 0.9
    for _ in range(ITERS):  # Nesterov momentum, full batch
        Wl, bl = W + mom * vW, b + mom * vb
        Z = X @ Wl + bl
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
        G = (P - Y) / n
        gW = X.T @ G + lam * Wl
        gb = G.sum(axis=0)
        vW = mom * vW - lr * gW; vb = mom * vb - lr * gb
        W += vW; b += vb
    return W, b


def accuracy(X, y, W, b):
    return float(((X @ W + b).argmax(axis=1) == y).mean())


def main(train_csv, out):
    d = np.loadtxt(train_csv, delimiter=",", dtype=np.int64)
    X, y = features(d[:, :64]), d[:, 64]
    cut = len(y) * 3 // 4
    scores = [(accuracy(X[cut:], y[cut:], *fit(X[:cut], y[:cut], lam)), -lam, lam) for lam in LAMBDAS]
    best = max(scores)[2]
    for acc, _, lam in scores:
        print(f"lambda {lam:g}: holdout accuracy {acc:.4f}", file=sys.stderr)
    W, b = fit(X, y, best)
    T = np.zeros((64, 17, 10))
    T[:, 1:, :] = np.cumsum(W.reshape(64, 16, 10), axis=1)
    with open(out, "wb") as f:
        f.write(T.astype("<f4").tobytes())
        f.write(b.astype("<f4").tobytes())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
