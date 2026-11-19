```hypothesis
Family: label-fm (Factorization Machine with shared latent factors).
Why: optdigits has spatial stroke structure; pairwise pixel interactions captured by FM (k=8) add only ~1.7 MB weights and ~1700 ops/item vs label-linear's 640, but should close much of the gap to MLP (97.7%). The shared factors V[d][k] across classes keep params small (d*k + C*k + d*C = 1232) while modeling interactions.
Failure: If dev accuracy ≤ 0.9613 (label-linear) or cost ≥ 2× label-linear, the hypothesis fails.
```
```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy.
Encodes ordinal bytes as z-scored per feature. Learns linear weights W[d][C],
shared factor weights V[d][k], class-specific factor weights U[C][k], bias b[C].
Exports per-byte lookup tables: int32 magic 0x464D3100, d, k, C;
float32 T_lin[d][256][C], T_V[d][256][k], T_V2[d][256][k], b1[C], U[C][k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    # Split last 20% for validation (tuning split)
    split = int(n * 0.8)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]
    n_tr = X_tr.shape[0]

    rng = np.random.default_rng(SEED)

    # Encode features: ordinal z-scored per byte position
    enc = np.zeros((d, 256), dtype=np.float32)
    for j in range(d):
        v = X_tr[:, j].astype(np.float64)
        mu, sd = v.mean(), v.std() or 1.0
        enc[j] = ((np.arange(256, dtype=np.float64) - mu) / sd).astype(np.float32)

    # Encode training and validation
    Z_tr = enc[np.arange(d), X_tr.T].T  # (n_tr, d)
    Z_val = enc[np.arange(d), X_val.T].T  # (n_val, d)

    # Initialize parameters
    # W: (d, C), V: (d, K), U: (C, K), b: (C,)
    scale_W = math.sqrt(6.0 / d)
    scale_V = math.sqrt(6.0 / d)
    scale_U = math.sqrt(6.0 / K)
    W = (rng.uniform(-1, 1, (d, C)) * scale_W).astype(np.float32)
    V = (rng.uniform(-1, 1, (d, K)) * scale_V).astype(np.float32)
    U = (rng.uniform(-1, 1, (C, K)) * scale_U).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = [W, V, U, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def forward(Z):
        """Z: (n, d) -> scores: (n, C), also returns S_f, Q_f for grads"""
        # Linear part: (n, d) @ (d, C) -> (n, C)
        lin = Z @ W + b  # (n, C)
        # Factor part: for each f, S_f = Z @ V[:,f] (n,), Q_f = (Z^2) @ (V[:,f]^2) (n,)
        Z2 = Z * Z
        V2 = V * V
        S = Z @ V  # (n, K)
        Q = Z2 @ V2  # (n, K)
        I = 0.5 * (S * S - Q)  # (n, K)
        # Scores: lin + I @ U.T
        scores = lin + I @ U.T  # (n, C)
        return scores, S, Q, I

    def softmax_grad(scores, y_idx):
        """scores: (n, C), y_idx: (n,) -> grad: (n, C) = (P - Y) / n"""
        S = scores - scores.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(y_idx)), y_idx] -= 1.0
        return (P / len(y_idx)).astype(np.float32)

    def accuracy(Z, y):
        scores, _, _, _ = forward(Z)
        pred = scores.argmax(axis=1)
        return (pred == y).mean()

    t = 0
    best_val = -1.0
    best_params = None
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))

    for epoch in range(max_epochs):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            Z_b = Z_tr[idx]
            y_b = y_tr[idx]

            scores, S, Q, I = forward(Z_b)
            g = softmax_grad(scores, y_b)  # (B, C)

            # Gradients
            # dW = Z_b.T @ g + L2 * W
            # db = g.sum(0)
            # dU = I.T @ g + L2 * U
            # dV: for each f, sum_c U[c,f] * g[:,c] * (S[:,f] - V[:,f] * Z_b)  [vectorized]
            dW = Z_b.T @ g + L2 * W
            db = g.sum(axis=0)
            dU = I.T @ g + L2 * U

            # dV computation: (d, K)
            # For each sample i, factor f: sum_c U[c,f] * g[i,c] = (g @ U)[:,f]  (B, K)
            # Then dV[:,f] = Z_b.T @ (gU[:,f] * (S[:,f][:,None] - V[:,f] * Z_b)) + L2 * V[:,f]
            gU = g @ U  # (B, K)
            dV = np.zeros_like(V)
            for f in range(K):
                coeff = gU[:, f]  # (B,)
                # S[:,f] is (B,), V[:,f] is (d,), Z_b is (B, d)
                # (S[:,f][:,None] - V[:,f] * Z_b) is (B, d)
                term = coeff[:, None] * (S[:, f, None] - V[None, :, f] * Z_b)  # (B, d)
                dV[:, f] = Z_b.T @ term + L2 * V[:, f]

            grads = [dW, dV, dU, db]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr; v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Validate
        val_acc = accuracy(Z_val, y_val)
        if val_acc > best_val:
            best_val = val_acc
            best_params = [p.copy() for p in params]

    # Retrain on full data with best hyperparams (here just use best_params as initialization and train a bit more? 
    # Simpler: just use best_params and compute tables on full data encoding)
    W, V, U, b = best_params

    # Re-encode on full training data for final tables
    enc_full = np.zeros((d, 256), dtype=np.float32)
    for j in range(d):
        v = X[:, j].astype(np.float64)
        mu, sd = v.mean(), v.std() or 1.0
        enc_full[j] = ((np.arange(256, dtype=np.float64) - mu) / sd).astype(np.float32)

    # Build lookup tables
    # T_lin[j][byte][c] = W[j,c] * enc_full[j][byte]
    # T_V[j][byte][f] = V[j,f] * enc_full[j][byte]
    # T_V2[j][byte][f] = V[j,f]^2 * enc_full[j][byte]^2
    T_lin = np.empty((d, 256, C), dtype=np.float32)
    T_V = np.empty((d, 256, K), dtype=np.float32)
    T_V2 = np.empty((d, 256, K), dtype=np.float32)
    for j in range(d):
        e = enc_full[j]  # (256,)
        T_lin[j] = W[j, :, None].T * e[:, None]  # (256, C) via broadcast
        T_V[j] = V[j, :, None].T * e[:, None]  # (256, K)
        T_V2[j] = (V[j, :, None] ** 2).T * (e[:, None] ** 2)

    # Pack weights: magic, d, K, C; then T_lin, T_V, T_V2, b, U
    magic = 0x464D3100  # 'FM1\0'
    hdr = struct.pack("<4i", magic, d, K, C)
    out = hdr
    out += T_lin.astype("<f4").tobytes()
    out += T_V.astype("<f4").tobytes()
    out += T_V2.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    out += U.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: 1 byte class (argmax, first max wins).
 * Weights format: int32 magic=0x464D3100, d, k, C; float32 T_lin[d*256*C], T_V[d*256*k], T_V2[d*256*k], b1[C], U[C*k]. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *slurp(const char *p, long *n) {
    FILE *f = fopen(p, "rb");
    if (!f) { perror(p); exit(3); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *b = malloc(*n ? *n : 1);
    if (fread(b, 1, *n, f) != (size_t)*n) { fprintf(stderr, "short read %s\n", p); exit(3); }
    fclose(f);
    return b;
}

static const unsigned char *next_rec(const unsigned char *buf, long n, long *pos, uint32_t *len) {
    if (*pos + 4 > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    uint32_t k = buf[*pos] | buf[*pos + 1] << 8 | buf[*pos + 2] << 16 | (uint32_t)buf[*pos + 3] << 24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k >> 8 & 255, k >> 16 & 255, k >> 24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D3100) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected_floats = (long)d * 256 * (C + 2L * K) + C + (long)C * K;
    long expected_bytes = 16 + 4 * expected_floats;
    if (nw != expected_bytes) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected_bytes); return 3; }

    const float *T_lin = (const float *)(hd + 4);
    const float *T_V = T_lin + (long)d * 256 * C;
    const float *T_V2 = T_V + (long)d * 256 * K;
    const float *b1 = T_V2 + (long)d * 256 * K;
    const float *U = b1 + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * C);
    float *sum_vx = malloc(sizeof(float) * K);
    float *sum_v2x2 = malloc(sizeof(float) * K);
    if (!acc || !sum_vx || !sum_v2x2) { fprintf(stderr, "oom\n"); return 3; }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) acc[c] = b1[c];
        for (int f = 0; f < K; f++) { sum_vx[f] = 0.0f; sum_v2x2[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char byte_val = xi[j];
            long base_lin = ((long)j * 256 + byte_val) * C;
            long base_V = ((long)j * 256 + byte_val) * K;
            for (int c = 0; c < C; c++) acc[c] += T_lin[base_lin + c];
            for (int f = 0; f < K; f++) {
                sum_vx[f] += T_V[base_V + f];
                sum_v2x2[f] += T_V2[base_V + f];
            }
        }

        for (int c = 0; c < C; c++) {
            float s = 0.0f;
            long base_U = (long)c * K;
            for (int f = 0; f < K; f++) {
                float sv = sum_vx[f];
                s += 0.5f * U[base_U + f] * (sv * sv - sum_v2x2[f]);
            }
            acc[c] += s;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```