```hypothesis
Factorization Machine (FM) with latent dim k=4 captures pairwise feature interactions crucial for tabular data like Adult, where combinations (e.g., education+occupation) matter. Inference is O(d*k) = 56 multiply-adds per item, similar to linear's O(d). If dev accuracy ≤ 0.8524 (label-linear) or latency ≥ 2× linear, the hypothesis fails.
```
```python
"""Factorization Machine for tabular label tasks: linear per-byte weights + low-rank pairwise interactions.
python3 -s -P train.py card.json train.in train.key weights.bin  (deterministic: seed 1, single thread)
Encoding: per-byte lookup tables for linear (T_lin) and latent (T_lat[k]) weights. Exported as:
int32 magic 0x464D3130, d, k, C; float32 T_lin[d][256][C], T_lat[k][d][256], bias[C]."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, L2_LAT, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 1e-4, 64, 600000, 100


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
    C = card["classes"]
    d = X.shape[1]
    rng = np.random.default_rng(SEED)

    # Build per-byte embedding indices (same as linear: one-hot for cat, z-scored for ordinal)
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals))); E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    # Flatten to get total embedding dimension per feature
    # For FM, we keep features separate: each feature j has 256 possible values, each with a k-dim latent vector
    # and a scalar linear weight per class.

    # Parameters:
    # W_lin: (d, 256, C)  -- linear weights
    # W_lat: (k, d, 256)  -- latent vectors (shared across classes for binary; for multi-class would be (k,d,256,C))
    # bias:  (C,)
    # For binary C=2, we can do one-vs-rest or just 2 outputs. Let's do C outputs for consistency.

    W_lin = (rng.uniform(-1, 1, (d, 256, C)) * math.sqrt(6.0 / 256)).astype(np.float32)
    W_lat = (rng.uniform(-1, 1, (K, d, 256)) * math.sqrt(6.0 / (d * 256))).astype(np.float32)
    bias = np.zeros(C, np.float32)

    params = [W_lin, W_lat, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi = X[idx]          # (B, d) uint8
            yy = y[idx]          # (B,)
            B = len(yy)

            # Forward pass
            # Linear part: sum_j W_lin[j, xi_j, :]  -> (B, C)
            lin = bias[None, :].copy()
            for j in range(d):
                lin += W_lin[j, xi[:, j], :]

            # FM interaction part: 0.5 * sum_f [ (sum_j v_f[j, xi_j])^2 - sum_j v_f[j, xi_j]^2 ]
            # v_f shape: (k, d, 256)
            # For each sample, for each f: sum over j of v_f[j, xi_j] -> (B, k)
            sum_v = np.zeros((B, K), np.float32)
            sum_v2 = np.zeros((B, K), np.float32)
            for j in range(d):
                v_j = W_lat[:, j, xi[:, j]].T   # (B, k)
                sum_v += v_j
                sum_v2 += v_j * v_j
            inter = 0.5 * ((sum_v * sum_v - sum_v2).sum(axis=1, keepdims=True))  # (B, 1)
            # Broadcast to C classes (same interaction for all classes in standard FM)
            logits = lin + inter  # (B, C)

            # Softmax + cross-entropy gradient
            logits = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yy] -= 1.0
            grad_logits = (probs / B).astype(np.float32)  # (B, C)

            # Gradients
            # dL/dW_lin[j, v, c] = sum_{i: xi_j=v} grad_logits[i, c] + L2 * W_lin
            g_W_lin = np.zeros_like(W_lin)
            for j in range(d):
                for c in range(C):
                    np.add.at(g_W_lin[j, :, c], xi[:, j], grad_logits[:, c])
            g_W_lin += L2 * W_lin

            # dL/dW_lat[f, j, v] = sum_{i: xi_j=v} (sum_v[i,f] - v_f[j,v]) * inter_grad[i] + L2_lat * W_lat
            # where inter_grad = dL/dinter = sum_c grad_logits[:,c]  (since inter added to all logits)
            inter_grad = grad_logits.sum(axis=1)  # (B,)
            g_W_lat = np.zeros_like(W_lat)
            for j in range(d):
                v_j = W_lat[:, j, xi[:, j]].T   # (B, k)
                # For each f: (sum_v[:,f] - v_j[:,f]) * inter_grad
                diff = (sum_v - v_j) * inter_grad[:, None]  # (B, k)
                for f in range(K):
                    np.add.at(g_W_lat[f, j, :], xi[:, j], diff[:, f])
            g_W_lat += L2_LAT * W_lat

            g_bias = grad_logits.sum(axis=0)

            grads = [g_W_lin, g_W_lat, g_bias]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr; v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export weights
    # Layout: magic, d, k, C; then W_lin (d*256*C), W_lat (k*d*256), bias (C)
    magic = 0x464D3130  # 'FM10'
    out = struct.pack("<4i", magic, d, K, C)
    out += W_lin.astype("<f4").tobytes()
    out += W_lat.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: 1 byte class index (argmax).
 * Weights layout: int32 magic(0x464D3130), d, k, C;
 * float32 W_lin[d][256][C], W_lat[k][d][256], bias[C]. */
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * C + (long)K * d * 256 + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *W_lin = p; p += (long)d * 256 * C;
    const float *W_lat = p; p += (long)K * d * 256;
    const float *bias = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(sizeof(float) * C);
    float *sum_v = malloc(sizeof(float) * K);
    float *sum_v2 = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Linear part
        for (int c = 0; c < C; c++) logits[c] = bias[c];
        for (int j = 0; j < d; j++) {
            const float *w = W_lin + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) logits[c] += w[c];
        }

        // FM interaction part
        for (int f = 0; f < K; f++) { sum_v[f] = 0.0f; sum_v2[f] = 0.0f; }
        for (int j = 0; j < d; j++) {
            const float *v = W_lat + ((long)f * d * 256 + j * 256 + xi[j]); // wait, f is loop var
        }
        // Fix: need to loop over f inside j or j inside f. Let's do j inside f for cache.
        for (int f = 0; f < K; f++) {
            float sv = 0.0f, sv2 = 0.0f;
            for (int j = 0; j < d; j++) {
                float v = W_lat[(long)f * d * 256 + j * 256 + xi[j]];
                sv += v;
                sv2 += v * v;
            }
            sum_v[f] = sv;
            sum_v2[f] = sv2;
        }
        float inter = 0.0f;
        for (int f = 0; f < K; f++) {
            inter += 0.5f * (sum_v[f] * sum_v[f] - sum_v2[f]);
        }
        for (int c = 0; c < C; c++) logits[c] += inter;

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```