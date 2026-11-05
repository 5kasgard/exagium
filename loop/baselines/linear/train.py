"""Baseline 1: linear (multinomial logistic regression) on raw pixels / 16.

python3 train.py TRAIN_CSV OUT_WEIGHTS
Deterministic: no randomness, fixed iteration counts, float64. L2 strength is picked on a fixed
holdout (the last 25% of the training file's rows), then the model is retrained on every row.
Weights: float32 little-endian, W[10][64] then b[10].
"""
import sys
import numpy as np

LAMBDAS = (1e-4, 1e-3, 1e-2)
ITERS = 3000


def features(x):
    return x / 16.0


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
    X, y = features(d[:, :64].astype(np.float64)), d[:, 64]
    cut = len(y) * 3 // 4
    scores = [(accuracy(X[cut:], y[cut:], *fit(X[:cut], y[:cut], lam)), -lam, lam) for lam in LAMBDAS]
    best = max(scores)[2]
    for acc, _, lam in scores:
        print(f"lambda {lam:g}: holdout accuracy {acc:.4f}", file=sys.stderr)
    W, b = fit(X, y, best)
    with open(out, "wb") as f:
        f.write(np.ascontiguousarray(W.T, dtype="<f4").tobytes())  # W[10][64]: class-major
        f.write(b.astype("<f4").tobytes())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
