```hypothesis
Family: label-fm (Factorization Machine with per-feature embeddings).
Reason: Adult income depends on feature interactions (e.g., education×occupation). FM captures all pairwise interactions via low-rank embeddings (k=8) with O(d·k) inference, cheaper than MLP/GBDT and more expressive than linear.
Failure: If dev accuracy ≤ 0.8524 (label-linear) or inference cost > 719 Ir/item, the added complexity isn't justified.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Each feature (0-255) gets an embedding vector.
Two-class FM: bias, linear weight, and embedding per class. Adam, L2, minibatch 64.
Exported weights: magic 0x464D5F31, d, k, C; then per class: bias, linear[d][256], emb[d][256][k] (float32 LE)."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 50

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def adam_update(param, grad, m, v, t):
    m *= 0.9; m += 0.1 * grad
    v *= 0.999; v += 0.001 * grad * grad
    m_hat = m / (1 - 0.9 ** t)
    v_hat = v / (1 - 0.999 ** t)
    param -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # Parameters per class
    biases = np.zeros(C, np.float32)
    linears = [np.zeros((d, 256), np.float32) for _ in range(C)]
    embs = [np.zeros((d, 256, K), np.float32) for _ in range(C)]

    # Initialize only seen categories
    for j in range(d):
        vals = np.unique(X[:, j])
        scale_lin = math.sqrt(6.0 / 256)
        scale_emb = math.sqrt(6.0 / (256 + K))
        for c in range(C):
            linears[c][j, vals] = rng.uniform(-scale_lin, scale_lin, size=len(vals)).astype(np.float32)
            embs[c][j, vals] = rng.uniform(-scale_emb, scale_emb, size=(len(vals), K)).astype(np.float32)

    # Adam state
    m_bias = [np.zeros_like(b) for b in biases]
    v_bias = [np.zeros_like(b) for b in biases]
    m_lin = [np.zeros_like(l) for l in linears]
    v_lin = [np.zeros_like(l) for l in linears]
    m_emb = [np.zeros_like(e) for e in embs]
    v_emb = [np.zeros_like(e) for e in embs]

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward: compute logits and cache intermediates
            logits = np.zeros((B, C), np.float32)
            S_cache = []
            V_cache = []
            for c in range(C):
                lin_term = np.zeros(B, np.float32)
                for j in range(d):
                    lin_term += linears[c][j, xb[:, j]]
                S = np.zeros((B, K), np.float32)
                V = np.zeros((B, d, K), np.float32)
                for j in range(d):
                    V[:, j, :] = embs[c][j, xb[:, j]]
                    S += V[:, j, :]
                inter = 0.5 * np.sum(S * S - np.sum(V * V, axis=1), axis=1)
                logits[:, c] = biases[c] + lin_term + inter
                S_cache.append(S)
                V_cache.append(V)

            # Softmax and loss gradient
            max_logits = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            dlogits = probs
            dlogits[np.arange(B), yb] -= 1.0
            dlogits /= B

            # Backward per class
            for c in range(C):
                dl = dlogits[:, c]  # (B,)
                # bias
                db = np.sum(dl)
                # linear
                dL = np.zeros((d, 256), np.float32)
                for j in range(d):
                    np.add.at(dL[j], xb[:, j], dl)
                dL += L2 * linears[c]
                # embedding
                dV = np.zeros((d, 256, K), np.float32)
                S = S_cache[c]
                V = V_cache[c]
                for j in range(d):
                    grad = dl[:, None] * (S - V[:, j, :])  # (B, K)
                    np.add.at(dV[j], xb[:, j], grad)
                dV += L2 * embs[c]

                # Adam updates
                t += 1
                adam_update(biases[c], db, m_bias[c], v_bias[c], t)
                adam_update(linears[c], dL, m_lin[c], v_lin[c], t)
                adam_update(embs[c], dV, m_emb[c], v_emb[c], t)

    # Write weights
    magic = 0x464D5F31  # 'FM_1'
    header = struct.pack("<4i", magic, d, K, C)
    payload = bytearray()
    for c in range(C):
        payload += struct.pack("<f", biases[c])
        payload += linears[c].astype("<f4").tobytes()
        payload += embs[c].astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + payload)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (one per feature). Output: one byte class index (argmax of two logits).
 * Weights format: magic 0x464D5F31, int32 d, K, C; then per class: float bias, linear[d][256], emb[d][256][K] (all float32 LE). */
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
    if (nw < 16 || hd[0] != 0x464D5F31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d <= 0 || K <= 0 || d > 256 || K > 256) { fprintf(stderr, "bad header dims\n"); return 3; }

    size_t expected = 16 + (size_t)C * (1 + (size_t)d * 256 + (size_t)d * 256 * K) * sizeof(float);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size: got %ld expected %zu\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *biases = p; p += C;
    const float *linear = p; p += (size_t)C * d * 256;
    const float *emb = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float logits[2];
    float S[256];  // max K=256, but we use K up to 256
    float V[14][256];  // max d=14, K=256

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            logits[c] = biases[c];
            // linear term
            const float *lin_c = linear + (size_t)c * d * 256;
            for (int j = 0; j < d; j++) {
                logits[c] += lin_c[j * 256 + xi[j]];
            }
            // interaction term
            const float *emb_c = emb + (size_t)c * d * 256 * K;
            memset(S, 0, K * sizeof(float));
            for (int j = 0; j < d; j++) {
                const float *v = emb_c + (size_t)j * 256 * K + (size_t)xi[j] * K;
                for (int f = 0; f < K; f++) {
                    float val = v[f];
                    S[f] += val;
                    V[j][f] = val;
                }
            }
            float inter = 0.0f;
            for (int f = 0; f < K; f++) {
                float sq_sum = 0.0f;
                for (int j = 0; j < d; j++) sq_sum += V[j][f] * V[j][f];
                inter += S[f] * S[f] - sq_sum;
            }
            logits[c] += 0.5f * inter;
        }

        unsigned char best = (logits[1] > logits[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```