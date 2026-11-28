```hypothesis
Factorization Machine (label-fm) with k=8 latent factors captures pairwise feature interactions efficiently, improving over linear model on tabular data with mixed types, while keeping inference cost low (O(d*k) per item). If dev accuracy does not exceed 0.855, the model does not justify its added complexity.
```
```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Encoding: same as label-linear (ordinal z-scored, categorical one-hot).
Model: linear logits + FM interaction (k=8) -> logits.
Adam lr=0.01, L2=1e-4, batch=64, epochs=min(50, ceil(600000/n)).
Exported: magic 0x464D3130, d, k, C; float32 tables T[d][256][C+2k]; float32 W2[k][C], b2[C]."""
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


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # Build encoding matrices per feature
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

    # Concatenate encoded features for training
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, D = Z.shape

    # Initialize parameters
    scale_lin = math.sqrt(6.0 / D)
    scale_fm = math.sqrt(6.0 / D)
    scale_w2 = math.sqrt(6.0 / K)
    W_lin = (rng.uniform(-1, 1, (D, C)) * scale_lin).astype(np.float32)
    b_lin = np.zeros(C, np.float32)
    V = (rng.uniform(-1, 1, (D, K)) * scale_fm).astype(np.float32)
    W2 = (rng.uniform(-1, 1, (K, C)) * scale_w2).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [W_lin, b_lin, V, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            bsz = z.shape[0]

            # Forward
            lin_logits = z @ W_lin + b_lin
            sum_V = z @ V
            sum_V2 = (z * z) @ (V * V)
            inter = 0.5 * (sum_V * sum_V - sum_V2)
            fm_logits = inter @ W2 + b2
            logits = lin_logits + fm_logits

            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yy] -= 1.0
            grad_logits = probs / bsz

            # Gradients
            grad_W_lin = z.T @ grad_logits + L2 * W_lin
            grad_b_lin = grad_logits.sum(0)
            grad_fm_logits = grad_logits
            grad_W2 = inter.T @ grad_fm_logits + L2 * W2
            grad_b2 = grad_fm_logits.sum(0)
            grad_inter = grad_fm_logits @ W2.T
            grad_sum_V = grad_inter * sum_V
            grad_sum_V2 = -0.5 * grad_inter
            grad_V = z.T @ grad_sum_V + (z * z).T @ grad_sum_V2 + L2 * V

            grads = [grad_W_lin, grad_b_lin, grad_V, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr; v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Precompute per-feature lookup tables: for each feature j, byte value v -> vector of length (C + 2*K)
    # Contains: linear contribution (C), V contribution (K), V^2 contribution (K)
    T_list = []
    for j, E in enumerate(enc):
        d_j = E.shape[1]
        off = sum(enc[i].shape[1] for i in range(j))
        W_lin_j = W_lin[off:off + d_j]
        V_j = V[off:off + d_j]
        V2_j = V_j * V_j
        # For each byte value, compute contributions
        lin_contrib = E @ W_lin_j  # (256, C)
        V_contrib = E @ V_j        # (256, K)
        V2_contrib = E @ V2_j      # (256, K)
        T_j = np.concatenate([lin_contrib, V_contrib, V2_contrib], axis=1)  # (256, C+2K)
        T_list.append(T_j.astype(np.float32))

    d = X.shape[1]
    out = struct.pack("<4i", 0x464D3130, d, K, C)
    out += np.stack(T_list).astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: one byte class index (argmax). */
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
    if (C < 1 || C > 256 || K < 1 || K > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    int stride = C + 2 * K;
    long expected = 16L + 4L * d * 256 * stride + 4L * K * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *W2 = T + (long)d * 256 * stride;
    const float *b2 = W2 + (long)K * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *lin = malloc(sizeof(float) * C);
    float *sumV = malloc(sizeof(float) * K);
    float *sumV2 = malloc(sizeof(float) * K);
    float *sc = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) lin[c] = 0.0f;
        for (int u = 0; u < K; u++) { sumV[u] = 0.0f; sumV2[u] = 0.0f; }

        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * stride;
            for (int c = 0; c < C; c++) lin[c] += t[c];
            for (int u = 0; u < K; u++) {
                float v = t[C + u];
                sumV[u] += v;
                sumV2[u] += t[C + K + u];
            }
        }

        for (int c = 0; c < C; c++) sc[c] = lin[c];
        for (int u = 0; u < K; u++) {
            float inter = 0.5f * (sumV[u] * sumV[u] - sumV2[u]);
            for (int c = 0; c < C; c++) sc[c] += inter * W2[(long)u * C + c];
        }
        for (int c = 0; c < C; c++) sc[c] += b2[c];

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```