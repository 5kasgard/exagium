```hypothesis
Factorization Machine with CP-decomposed pairwise interactions (FM2): each position-value pair gets a shared k-dim vector V; each class gets a k-dim vector U and linear weights W. Pairwise term per class is sum_f U_cf * 0.5*( (sum_j V_jf)^2 - sum_j V_jf^2 ). Captures bigram-style interactions critical for character sequences at O(d*k + C*k) inference. Beats label-linear by modeling position interactions; fails if k=16 is too low-rank or interactions are not predictive.
```

```python
"""FM2: Factorization Machine with CP-decomposed pairwise interactions for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy. Exports: magic 0x464D3231, d, k, C; float32 V[d][256][k], U[C][k], W[C][d][256], b[C]."""
import json, math, struct, sys

import numpy as np

K = 16
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def softmax_grad(logits, y):
    """Return gradient of cross-entropy w.r.t logits: (P - Y) / n."""
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return (probs / len(y)).astype(np.float32)


def predict_scores(Z_lin, V_batch, U, W_lin, b):
    """Compute scores for a batch.
    Z_lin: (B, C) linear part sum_j W_cj[x_j] + b_c
    V_batch: (B, d, k) embeddings for each position
    U: (C, k)
    Returns (B, C) scores.
    """
    # Pairwise features: for each f, 0.5*( (sum_j V_jf)^2 - sum_j V_jf^2 )
    sum_v = V_batch.sum(axis=1)          # (B, k)
    sum_sq = (V_batch * V_batch).sum(axis=1)  # (B, k)
    pairwise = 0.5 * (sum_v * sum_v - sum_sq)  # (B, k)
    # Class-specific pairwise contribution: pairwise @ U.T  -> (B, C)
    return Z_lin + pairwise @ U.T


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    n = X.shape[0]

    # Use last 20% for validation to pick K (here fixed K=16, but structure allows tuning)
    split = int(0.8 * n)
    idx_tr, idx_va = np.arange(split), np.arange(split, n)
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_va, y_va = X[idx_va], y[idx_va]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    V = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)
    U = (rng.normal(0, 0.01, (C, K))).astype(np.float32)
    W = np.zeros((C, d, 256), np.float32)
    b = np.zeros(C, np.float32)

    params = [V, U, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # Precompute linear indices for fast linear part: for each class c, pos j, value v -> W[c,j,v]
    # We'll compute linear scores per batch by gathering.

    t = 0
    best_val = -1
    best_params = None
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / max(1, len(X_tr))))

    for epoch in range(epochs):
        order = rng.permutation(len(X_tr))
        for s in range(0, len(X_tr), BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            B = len(yb)

            # Gather embeddings: (B, d, K)
            V_batch = V[np.arange(d)[None, :], xb[:, :, None], :]  # (B, d, K) - fancy indexing

            # Linear logits: b_c + sum_j W[c, j, x_j]
            # W is (C, d, 256), xb is (B, d) -> for each c: sum_j W[c, j, xb[:,j]]
            Z_lin = b[None, :].copy()  # (B, C)
            for j in range(d):
                Z_lin += W[:, j, xb[:, j]].T  # (B, C)

            # Full scores
            scores = predict_scores(Z_lin, V_batch, U, W, b)

            # Gradient
            g = softmax_grad(scores, yb)  # (B, C)

            # Gradients
            # dL/db = sum_b g
            g_b = g.sum(axis=0)  # (C,)

            # dL/dW[c,j,v] = sum_{batch where x_j=v} g_{b,c}
            g_W = np.zeros_like(W)
            for j in range(d):
                # For each class c, value v: sum over batch items where xb[:,j]==v of g[:,c]
                # Use bincount per class
                for c in range(C):
                    np.add.at(g_W[c, j], xb[:, j], g[:, c])

            # dL/dU[c,f] = sum_b g_{b,c} * pairwise_{b,f}
            sum_v = V_batch.sum(axis=1)
            sum_sq = (V_batch * V_batch).sum(axis=1)
            pairwise = 0.5 * (sum_v * sum_v - sum_sq)  # (B, K)
            g_U = g.T @ pairwise  # (C, K)

            # dL/dV[j,v,f] = sum_{b where x_{b,j}=v} g_{b,c} * U_{c,f} * (sum_v_{b,f} - V_{b,j,f})
            # For each position j, value v:
            g_V = np.zeros_like(V)
            # Precompute for each batch item and factor: h_{b,f} = sum_v_{b,f} - V_{b,j,f} for each j? No, depends on j.
            # Instead loop over j
            for j in range(d):
                V_j = V_batch[:, j, :]  # (B, K)
                diff = sum_v - V_j      # (B, K)
                # For each value v, we need sum over batch where xb[:,j]==v of g_{b,c} * U_{c,f} * diff_{b,f}
                # This is: for each v, (g[:,:,None] * U[None,:,:] * diff[:,None,:]).sum over batch where xb[:,j]==v and over c
                # Actually: dL/dV_{j,v,f} = sum_{b: x_{b,j}=v} sum_c g_{b,c} * U_{c,f} * diff_{b,f}
                # Let M_{b,f} = diff_{b,f} * sum_c g_{b,c} * U_{c,f} = diff_{b,f} * (g_b @ U_{:,f})
                M = diff * (g @ U)  # (B, K)
                for v in range(256):
                    mask = (xb[:, j] == v)
                    if mask.any():
                        g_V[j, v] = M[mask].sum(axis=0)

            # L2 regularization
            g_V += L2 * V
            g_U += L2 * U
            g_W += L2 * W
            # no L2 on bias

            grads = [g_V, g_U, g_W, g_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Validation accuracy
        if len(X_va) > 0:
            V_batch_va = V[np.arange(d)[None, :], X_va[:, :, None], :]
            Z_lin_va = b[None, :].copy()
            for j in range(d):
                Z_lin_va += W[:, j, X_va[:, j]].T
            scores_va = predict_scores(Z_lin_va, V_batch_va, U, W, b)
            preds = scores_va.argmax(axis=1)
            acc = (preds == y_va).mean()
            if acc > best_val:
                best_val = acc
                best_params = [p.copy() for p in params]

    # Use best validation params if available, else last
    if best_params is not None:
        V, U, W, b = best_params

    # Retrain on full data for a few epochs? The spec says: choose settings on tuning split, then train on all.
    # We'll do a few epochs on full data starting from best params.
    X_full, y_full = X, y
    for epoch in range(3):  # few epochs on full
        order = rng.permutation(len(X_full))
        for s in range(0, len(X_full), BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_full[idx], y_full[idx]
            B = len(yb)

            V_batch = V[np.arange(d)[None, :], xb[:, :, None], :]
            Z_lin = b[None, :].copy()
            for j in range(d):
                Z_lin += W[:, j, xb[:, j]].T
            scores = predict_scores(Z_lin, V_batch, U, W, b)
            g = softmax_grad(scores, yb)

            g_b = g.sum(axis=0)
            g_W = np.zeros_like(W)
            for j in range(d):
                for c in range(C):
                    np.add.at(g_W[c, j], xb[:, j], g[:, c])

            sum_v = V_batch.sum(axis=1)
            sum_sq = (V_batch * V_batch).sum(axis=1)
            pairwise = 0.5 * (sum_v * sum_v - sum_sq)
            g_U = g.T @ pairwise

            g_V = np.zeros_like(V)
            for j in range(d):
                V_j = V_batch[:, j, :]
                diff = sum_v - V_j
                M = diff * (g @ U)
                for v in range(256):
                    mask = (xb[:, j] == v)
                    if mask.any():
                        g_V[j, v] = M[mask].sum(axis=0)

            g_V += L2 * V
            g_U += L2 * U
            g_W += L2 * W

            grads = [g_V, g_U, g_W, g_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    # Magic, d, k, C
    out = struct.pack("<4i", 0x464D3231, d, K, C)
    out += V.astype("<f4").tobytes()
    out += U.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* FM2 inference: ./infer weights.bin in.rec N
 * Weights: magic 0x464D3231, int32 d, K, C; float32 V[d][256][K], U[C][K], W[C][d][256], b[C].
 * Input record: d bytes. Output: 1 byte class index. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
    if (nw < 16 || hd[0] != 0x464D3231) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * K + (long)C * K + (long)C * d * 256 + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *V = p; p += (long)d * 256 * K;
    const float *U = p; p += (long)C * K;
    const float *W = p; p += (long)C * d * 256;
    const float *b = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *sum_v = malloc(sizeof(float) * K);
    float *sum_sq = malloc(sizeof(float) * K);
    float *pairwise = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Zero accumulators
        for (int f = 0; f < K; f++) { sum_v[f] = 0.0f; sum_sq[f] = 0.0f; }

        // Gather V vectors and compute sum_v, sum_sq
        for (int j = 0; j < d; j++) {
            const float *v_ptr = V + ((long)j * 256 + xi[j]) * K;
            for (int f = 0; f < K; f++) {
                float val = v_ptr[f];
                sum_v[f] += val;
                sum_sq[f] += val * val;
            }
        }

        // Pairwise features
        for (int f = 0; f < K; f++) {
            pairwise[f] = 0.5f * (sum_v[f] * sum_v[f] - sum_sq[f]);
        }

        // Linear scores: b_c + sum_j W[c][j][xi[j]]
        for (int c = 0; c < C; c++) {
            float s = b[c];
            for (int j = 0; j < d; j++) {
                s += W[((long)c * d + j) * 256 + xi[j]];
            }
            scores[c] = s;
        }

        // Add pairwise contribution: scores[c] += sum_f U[c][f] * pairwise[f]
        for (int c = 0; c < C; c++) {
            const float *u_ptr = U + (long)c * K;
            float add = 0.0f;
            for (int f = 0; f < K; f++) add += u_ptr[f] * pairwise[f];
            scores[c] += add;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(sum_v); free(sum_sq); free(pairwise); free(scores);
    return 0;
}
```