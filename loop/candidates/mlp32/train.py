"""Small MLP 64 -> H -> 10 (ReLU), trained deterministically, frozen as int16 per-pixel tables.

python3 train.py TRAIN_CSV OUT_WEIGHTS

Model: z = b1 + W1^T (x/16), h = relu(z), scores = W2^T h + b2; softmax cross-entropy, L2, Adam (minibatch 32,
step decay), He-uniform init from np.random.default_rng(SEED). Like the baselines, L2 is picked on the last 25%
of the training file (train on the first 75%), then the model is refit on all rows.

Determinism: training uses only elementwise + - * / sqrt max and indexing. Matrix products are explicit
sequential accumulations (no BLAS) and exp is computed from + * / (no libm), so the bytes do not depend on the
CPU's BLAS kernel, SIMD width or libm.

Frozen form (what infer.c runs): because the input pixel v is an integer 0..16, layer 1 is a lookup table
T[p][v][h] = round(s_h * v/16 * W1[p,h]) in int16, with per-unit scale s_h chosen so |b1 + sum_p T| <= 32767 for
every possible image (no overflow). ReLU is positive-homogeneous, so 1/s_h is folded into W2 (float32).
weights.bin (little-endian): int16 T[64][17][H], int16 B1[H], float32 W2[H][12] (classes padded to 12 with 0),
float32 b2[12] (padding -1e30).
"""
import sys
import numpy as np

H = 32


def mm(A, B):  # A (m,k) @ B (k,n), accumulated over k in order
    acc = A[:, 0:1] * B[0]
    for k in range(1, A.shape[1]):
        acc = acc + A[:, k:k + 1] * B[k]
    return acc


def mtm(A, G):  # A.T @ G, accumulated over rows in order
    acc = A[0][:, None] * G[0][None, :]
    for i in range(1, A.shape[0]):
        acc = acc + A[i][:, None] * G[i][None, :]
    return acc


def exp_neg(x):  # exp(x) for x <= 0: clip, scale by 2^-8, Taylor to degree 10, square 8 times
    y = np.maximum(x, -60.0) * (1.0 / 256.0)
    p = np.ones_like(y)
    for k in range(10, 0, -1):
        p = 1.0 + p * y * (1.0 / k)
    for _ in range(8):
        p = p * p
    return p


def fit(V, y, H, epochs, lam, seed, lr=0.003, bs=32):
    """V: (n,64) ints 0..16. Returns float64 W1 (64,H), b1, W2 (H,10), b2."""
    rng = np.random.default_rng(seed)
    n = len(y)
    X0 = V / 16.0
    W1 = (rng.random((64, H)) * 2 - 1) * np.sqrt(6.0 / 64)
    b1 = np.zeros(H)
    W2 = (rng.random((H, 10)) * 2 - 1) * np.sqrt(6.0 / (H + 10))
    b2 = np.zeros(10)
    P = [W1, b1, W2, b2]
    M = [np.zeros_like(p) for p in P]; S = [np.zeros_like(p) for p in P]
    Y = np.zeros((n, 10)); Y[np.arange(n), y] = 1.0
    p1 = p2 = 1.0  # 0.9^t and 0.999^t by repeated multiplication (no libm pow)
    for ep in range(epochs):
        X = X0
        order = rng.permutation(n)
        lr_ep = lr
        if ep >= epochs * 0.6:
            lr_ep = lr * 0.3
        if ep >= epochs * 0.85:
            lr_ep = lr * 0.05
        for s in range(0, n, bs):
            idx = order[s:s + bs]
            xb, yb = X[idx], Y[idx]
            m = len(idx)
            z = mm(xb, W1) + b1
            h = np.maximum(z, 0.0)
            o = mm(h, W2) + b2
            o = o - o.max(axis=1, keepdims=True)
            e = exp_neg(o)
            p = e / e.sum(axis=1, keepdims=True)
            go = (p - yb) * (1.0 / m)
            gW2 = mtm(h, go) + lam * W2
            gb2 = go.sum(axis=0)
            gz = mm(go, W2.T) * (z > 0)
            gW1 = mtm(xb, gz) + lam * W1
            gb1 = gz.sum(axis=0)
            p1 *= 0.9; p2 *= 0.999
            c1, c2 = 1 - p1, 1 - p2
            for i, g in enumerate((gW1, gb1, gW2, gb2)):
                M[i] *= 0.9; M[i] += 0.1 * g
                S[i] *= 0.999; S[i] += 0.001 * (g * g)
                P[i] -= lr_ep * (M[i] / c1) / (np.sqrt(S[i] / c2) + 1e-8)
    return W1, b1, W2, b2


def predict(V, W1, b1, W2, b2):
    h = np.maximum(mm(V / 16.0, W1) + b1, 0.0)
    return (mm(h, W2) + b2).argmax(axis=1)


def quantize(W1, b1, W2, b2):
    """int16 tables T[p][v][h] = round(s_h * v/16 * W1[p,h]), bias round(s_h b1); |sum| < 32767 guaranteed.
    W2 rows divided by s_h (ReLU is positive-homogeneous), float32."""
    H = W1.shape[1]
    bound = np.abs(b1) + np.abs(W1).sum(axis=0)
    s = (32767.0 - 66.0) / np.maximum(bound, 1e-12)
    v = np.arange(17) / 16.0
    T = np.round(v[None, :, None] * W1[:, None, :] * s[None, None, :]).astype(np.int64)
    B1 = np.round(b1 * s).astype(np.int64)
    W2q = (W2 / s[:, None]).astype(np.float32)
    return T, B1, W2q, b2.astype(np.float32)


def predict_q(V, T, B1, W2q, b2):
    z = T[np.arange(64)[None, :], V].sum(axis=1) + B1
    assert np.abs(z).max() <= 32767
    h = np.maximum(z, 0).astype(np.float32)
    o = b2[None, :].astype(np.float32).copy()
    for j in range(h.shape[1]):
        o = o + h[:, j:j + 1] * W2q[j]
    return o.argmax(axis=1)


SEED = 1
EPOCHS = 100
LAMBDAS = (1e-4, 1e-3)


def main(train_csv, out):
    d = np.loadtxt(train_csv, delimiter=",", dtype=np.int64)
    V, y = d[:, :64], d[:, 64]
    cut = len(y) * 3 // 4
    scores = []
    for lam in LAMBDAS:
        q = quantize(*fit(V[:cut], y[:cut], H, EPOCHS, lam, SEED))
        acc = float((predict_q(V[cut:], *q) == y[cut:]).mean())
        print(f"H {H} lambda {lam:g}: holdout accuracy (int16 frozen form) {acc:.4f}", file=sys.stderr)
        scores.append((acc, -lam, lam))
    best = max(scores)[2]
    T, B1, W2q, b2 = quantize(*fit(V, y, H, EPOCHS, best, SEED))
    W2p = np.zeros((H, 12), dtype=np.float32); W2p[:, :10] = W2q
    b2p = np.full(12, -1e30, dtype=np.float32); b2p[:10] = b2
    with open(out, "wb") as f:
        f.write(T.astype("<i2").tobytes())
        f.write(B1.astype("<i2").tobytes())
        f.write(W2p.astype("<f4").tobytes())
        f.write(b2p.astype("<f4").tobytes())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
