```hypothesis
Family: label-fm (Factorization Machine with linear layer). 
Why: Captures pairwise interactions between the 16 context bytes via low-rank embeddings, which are crucial for character-level language modeling, while remaining fast at inference (O(d*k + k*C) per item). 
Failure signal: Dev accuracy ≤ label-linear (0.3375) or inference cost > 2× label-linear.
```

```python
"""Factorization Machine with linear layer for next-character prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weight format: magic 0x464D454D, int32 d,k,C; float32 embeddings[d][256][k], linear[d][256][C], out_w[2k][C], out_b[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
K = 8                # embedding dimension
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

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
    d = X.shape[1]          # 16
    C = card["classes"]     # 65
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Embeddings: (d, 256, K)
    emb = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)
    # Linear weights: (d, 256, C)
    lin = np.zeros((d, 256, C), dtype=np.float32)
    # Output layer: (2K, C)
    out_w = (rng.normal(0, 0.01, (2*K, C))).astype(np.float32)
    out_b = np.zeros(C, dtype=np.float32)

    params = [emb, lin, out_w, out_b]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    def forward(Xb):
        # Xb: (B, d) uint8
        B = Xb.shape[0]
        # Sum embeddings
        s = np.zeros((B, K), dtype=np.float32)
        sq = np.zeros((B, K), dtype=np.float32)
        for j in range(d):
            e = emb[j, Xb[:, j]]          # (B, K)
            s += e
            sq += e * e
        inter = 0.5 * (s * s - sq)        # (B, K)
        feat = np.concatenate([s, inter], axis=1)   # (B, 2K)
        # Linear part
        lin_part = np.zeros((B, C), dtype=np.float32)
        for j in range(d):
            lin_part += lin[j, Xb[:, j]]
        logits = feat @ out_w + lin_part + out_b   # (B, C)
        return logits, feat, s, sq, lin_part

    def softmax_cross_entropy(logits, yb):
        # logits (B, C), yb (B,)
        max_log = logits.max(axis=1, keepdims=True)
        exp_log = np.exp(logits - max_log)
        probs = exp_log / exp_log.sum(axis=1, keepdims=True)
        loss = -np.log(probs[np.arange(len(yb)), yb] + 1e-12).mean()
        grad = probs.copy()
        grad[np.arange(len(yb)), yb] -= 1.0
        grad /= len(yb)
        return loss, grad

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for ep in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, BATCH):
            idx = order[start:start+BATCH]
            Xb, yb = X[idx], y[idx]
            logits, feat, s, sq, lin_part = forward(Xb)
            loss, grad_logits = softmax_cross_entropy(logits, yb)  # (B, C)

            # Gradients
            # out_w: (2K, C) <- feat.T @ grad_logits
            # out_b: (C,) <- grad_logits.sum(0)
            grad_out_w = feat.T @ grad_logits
            grad_out_b = grad_logits.sum(axis=0)

            # Backprop through feat = [s, inter]
            grad_feat = grad_logits @ out_w.T          # (B, 2K)
            grad_s = grad_feat[:, :K]                  # (B, K)
            grad_inter = grad_feat[:, K:]              # (B, K)

            # inter = 0.5*(s^2 - sq) => d_inter/ds = s, d_inter/dsq = -0.5
            grad_s += grad_inter * s
            grad_sq = -0.5 * grad_inter

            # s = sum_j emb[j, x_j], sq = sum_j emb[j, x_j]^2
            grad_emb = np.zeros_like(emb)
            grad_lin = np.zeros_like(lin)
            for j in range(d):
                xj = Xb[:, j]
                # grad_emb[j, xj] += grad_s + 2 * emb[j, xj] * grad_sq
                # We'll accumulate per category
                for cat in range(256):
                    mask = (xj == cat)
                    if not mask.any():
                        continue
                    gs = grad_s[mask].sum(axis=0)          # (K,)
                    gq = grad_sq[mask].sum(axis=0)         # (K,)
                    e = emb[j, cat]                        # (K,)
                    grad_emb[j, cat] += gs + 2 * e * gq
                # linear part gradient
                for cat in range(256):
                    mask = (xj == cat)
                    if mask.any():
                        grad_lin[j, cat] += grad_logits[mask].sum(axis=0)

            # Add L2 regularization
            grad_emb += L2 * emb
            grad_lin += L2 * lin
            grad_out_w += L2 * out_w
            # out_b no L2

            grads = [grad_emb, grad_lin, grad_out_w, grad_out_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x464D454D  # "MEFM"
    with open(sys.argv[4], "wb") as f:
        f.write(struct.pack("<4i", magic, d, K, C))
        f.write(emb.astype("<f4").tobytes())
        f.write(lin.astype("<f4").tobytes())
        f.write(out_w.astype("<f4").tobytes())
        f.write(out_b.astype("<f4").tobytes())

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Each input record: 16 bytes. Output: 1 byte class index (record format). */
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
    if (nw < 16 || hd[0] != 0x464D454D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 16 || K <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad header values\n"); return 3; }

    size_t emb_sz = (size_t)d * 256 * K;
    size_t lin_sz = (size_t)d * 256 * C;
    size_t out_w_sz = (size_t)2 * K * C;
    size_t out_b_sz = C;
    size_t total_floats = emb_sz + lin_sz + out_w_sz + out_b_sz;
    if (nw != 16 + (long)total_floats * 4) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *lin = emb + emb_sz;
    const float *out_w = lin + lin_sz;
    const float *out_b = out_w + out_w_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *s = malloc(sizeof(float) * K);
    float *sq = malloc(sizeof(float) * K);
    float *inter = malloc(sizeof(float) * K);
    float *feat = malloc(sizeof(float) * 2 * K);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Zero accumulators
        for (int u = 0; u < K; u++) { s[u] = 0.0f; sq[u] = 0.0f; }

        // Sum embeddings and squares
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((size_t)j * 256 + xi[j]) * K;
            for (int u = 0; u < K; u++) {
                float val = e[u];
                s[u] += val;
                sq[u] += val * val;
            }
        }

        // Interaction
        for (int u = 0; u < K; u++) {
            inter[u] = 0.5f * (s[u] * s[u] - sq[u]);
        }

        // Concat s and inter -> feat
        for (int u = 0; u < K; u++) feat[u] = s[u];
        for (int u = 0; u < K; u++) feat[K + u] = inter[u];

        // Linear part
        for (int c = 0; c < C; c++) logits[c] = out_b[c];
        for (int j = 0; j < d; j++) {
            const float *lj = lin + ((size_t)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) logits[c] += lj[c];
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            for (int u = 0; u < 2*K; u++) sum += feat[u] * out_w[u * C + c];
            logits[c] += sum;
        }

        // Argmax
        unsigned char best = 0;
        float best_val = logits[0];
        for (int c = 1; c < C; c++) {
            if (logits[c] > best_val) { best_val = logits[c]; best = (unsigned char)c; }
        }
        put_rec(stdout, &best, 1);
    }

    free(s); free(sq); free(inter); free(feat); free(logits);
    if (x) free(x);
    free(hd);
    return 0;
}
```