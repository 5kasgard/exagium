```hypothesis
Family: label-fm (Factorization Machine, degree 2). Adult has strong pairwise interactions (education×occupation, marital-status×relationship, capital-gain×hours-per-week). FM captures all pairwise interactions with O(d·K) parameters (K=8 latent dim) instead of O(d²), giving non-linear power near linear-model cost. Shared embeddings across classes, per-class linear weights. Trained with Adam on cross-entropy. Failure signal: dev accuracy ≤ 0.8524 (label-linear) or Ir/item > 1000.
```
```python
"""Factorization Machine (degree 2) for tabular label tasks with byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Architecture: shared embeddings V[f][v][K], per-class linear W[c][f][v], bias b[c].
Forward: score_c = b[c] + Σ_f W[c][f][x_f] + ½ Σ_k [(Σ_f V[f][x_f][k])² - Σ_f V[f][x_f][k]²]
Exports: magic 0x464D3231, d, K, C; float32 V[d][256][K], W[C][d][256], b[C]."""
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
    assert C == 2
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Initialize parameters
    V = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)      # shared embeddings
    W = (rng.normal(0, 0.01, (C, d, 256))).astype(np.float32)       # per-class linear
    b = np.zeros(C, dtype=np.float32)                                 # per-class bias

    params = [V, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    # Precompute feature indices for speed
    f_idx = np.arange(d, dtype=np.int32)
    v_idx = X  # (n, d) uint8

    for epoch in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = v_idx[idx]           # (B, d)
            yb = y[idx]               # (B,)
            B = len(yb)

            # ---- Forward ----
            # V lookup: (B, d, K)
            V_x = V[f_idx, xb]        # advanced indexing
            # Linear term per class: (C, B)
            W_x = W[:, f_idx, xb]     # (C, B, d)
            linear = b[:, None] + W_x.sum(axis=2)   # (C, B)

            # Interaction term (shared across classes): (B,)
            sum_V = V_x.sum(axis=1)                 # (B, K)
            sum_V2 = (V_x * V_x).sum(axis=1)        # (B, K)
            inter = 0.5 * ((sum_V * sum_V - sum_V2).sum(axis=1))   # (B,)

            scores = linear + inter[None, :]        # (C, B)

            # Softmax + cross-entropy gradient
            scores -= scores.max(axis=0, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=0, keepdims=True)   # (C, B)
            probs[yb, np.arange(B)] -= 1.0
            grad_scores = probs / B   # (C, B)

            # ---- Backward ----
            # Gradient wrt linear term = grad_scores (C, B)
            # Gradient wrt interaction term = grad_scores.sum(axis=0) (B,)
            grad_inter = grad_scores.sum(axis=0)    # (B,)

            # Gradient for V: dL/dV[f][v][k] = Σ_{i: x_i[f]=v} grad_inter[i] * (sum_V[i,k] - V_x[i,f,k])
            # Efficient accumulation using scatter-add
            grad_V = np.zeros_like(V)
            # For each feature f
            for f in range(d):
                vf = xb[:, f]                       # (B,)
                V_xf = V_x[:, f, :]                 # (B, K)
                sum_V_f = sum_V - V_xf              # (B, K)
                # grad_inter[:, None] * sum_V_f  -> (B, K)
                g = (grad_inter[:, None] * sum_V_f).astype(np.float32)
                np.add.at(grad_V[f], vf, g)

            # Gradient for W: dL/dW[c][f][v] = Σ_{i: x_i[f]=v} grad_scores[c, i]
            grad_W = np.zeros_like(W)
            for c in range(C):
                for f in range(d):
                    vf = xb[:, f]
                    g = grad_scores[c].astype(np.float32)
                    np.add.at(grad_W[c, f], vf, g)

            # Gradient for b
            grad_b = grad_scores.sum(axis=1).astype(np.float32)   # (C,)

            # L2 regularization
            grad_V += L2 * V
            grad_W += L2 * W
            # no L2 on bias typically, but fine to add
            grad_b += L2 * b

            grads = [grad_V, grad_W, grad_b]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # ---- Export weights ----
    # Layout: magic, d, K, C; then V[d][256][K], W[C][d][256], b[C] all float32 little-endian
    out = struct.pack("<4i", 0x464D3231, d, K, C)
    out += V.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine (degree 2) inference for label tasks.
 * ./infer weights.bin in.rec N
 * Each input record: d bytes. Output: 1 byte class index (argmax). */
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
    if (C != 2 || d < 1 || d > 256 || K < 1 || K > 64) { fprintf(stderr, "bad header dims\n"); return 3; }

    // Expected size: 16 + 4*(d*256*K + C*d*256 + C)
    long expected = 16L + 4L * ((long)d * 256 * K + (long)C * d * 256 + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *ptr = (const float *)(hd + 4);
    const float *V = ptr;                                 // [d][256][K]
    ptr += (long)d * 256 * K;
    const float *W = ptr;                                 // [C][d][256]
    ptr += (long)C * d * 256;
    const float *b = ptr;                                 // [C]

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_V = malloc(sizeof(float) * K);
    float *V_xf = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Compute sum_V[k] = Σ_f V[f][xi[f]][k]
        for (int kk = 0; kk < K; kk++) sum_V[kk] = 0.0f;
        for (int f = 0; f < d; f++) {
            const float *vf = V + ((long)f * 256 + xi[f]) * K;
            for (int kk = 0; kk < K; kk++) sum_V[kk] += vf[kk];
        }

        // Interaction term: 0.5 * Σ_k (sum_V[k]^2 - Σ_f V[f][xi[f]][k]^2)
        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            float sum_sq = 0.0f;
            for (int f = 0; f < d; f++) {
                float v = V[((long)f * 256 + xi[f]) * K + kk];
                sum_sq += v * v;
            }
            inter += sum_V[kk] * sum_V[kk] - sum_sq;
        }
        inter *= 0.5f;

        // Scores per class
        float best_score = -1e30f;
        unsigned char best_class = 0;
        for (int c = 0; c < C; c++) {
            float score = b[c];
            for (int f = 0; f < d; f++) {
                score += W[((long)c * d + f) * 256 + xi[f]];
            }
            score += inter;
            if (score > best_score) { best_score = score; best_class = (unsigned char)c; }
        }
        put_rec(stdout, &best_class, 1);
    }
    free(sum_V); free(V_xf);
    return 0;
}
```