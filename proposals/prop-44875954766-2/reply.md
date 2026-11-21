```hypothesis
Family: label-fm (factorization machine with linear layer). Learns per-position byte embeddings, computes pairwise interactions via FM trick, and adds a linear layer to 65 classes. Captures position-specific interactions with far fewer parameters than label-linear (68k vs 266k), should generalize better on this character-level task. If dev accuracy ≤ 0.3375 or inference cost > 10k Ir/item, the hypothesis fails.
```

```python
"""label-fm: Factorization Machine with linear layer for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Weights format: magic 0x464D4C31, d=16, K, C=65; float32 embeddings[d][256][K], W[2K][C], b[C]."""
import json, math, struct, sys

import numpy as np

K = 16          # embedding size
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 256
EPOCHS = 20
SAMPLES = 600000

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(logits, y):
    """Return gradient of cross-entropy w.r.t logits (B, C)."""
    B = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(B), y] -= 1.0
    return probs / B

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # embeddings: (d, 256, K)
    emb = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)
    # W: (2K, C)
    W = rng.normal(0, math.sqrt(2.0 / (2 * K)), (2 * K, C)).astype(np.float32)
    b = np.zeros(C, np.float32)

    # Adam state
    m_emb = np.zeros_like(emb); v_emb = np.zeros_like(emb)
    m_W = np.zeros_like(W); v_W = np.zeros_like(W)
    m_b = np.zeros_like(b); v_b = np.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    max_epochs = min(EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # forward: gather embeddings
            # emb_j shape (B, K) for each j
            sum_v = np.zeros((B, K), np.float32)
            sum_v_sq = np.zeros((B, K), np.float32)
            emb_used = np.zeros((d, B, K), np.float32)
            for j in range(d):
                e = emb[j, xb[:, j]]      # (B, K)
                emb_used[j] = e
                sum_v += e
                sum_v_sq += e * e

            inter = 0.5 * (sum_v * sum_v - sum_v_sq)   # (B, K)
            hidden = np.concatenate([sum_v, inter], axis=1)   # (B, 2K)
            logits = hidden @ W + b                        # (B, C)

            # loss gradient
            dlogits = softmax_cross_entropy_grad(logits, yb)   # (B, C)

            # gradients for W, b
            dW = hidden.T @ dlogits + L2 * W   # (2K, C)
            db = dlogits.sum(axis=0)           # (C,)

            # gradient for hidden
            dhidden = dlogits @ W.T            # (B, 2K)
            dsum_v = dhidden[:, :K]
            dinter = dhidden[:, K:]

            # gradient for embeddings
            demb = np.zeros_like(emb)   # (d, 256, K)
            for j in range(d):
                e = emb_used[j]                # (B, K)
                # dinter/demb_j = dinter * (sum_v - e)
                grad_j = dsum_v + dinter * (sum_v - e)
                # scatter-add
                np.add.at(demb[j], xb[:, j], grad_j)
            demb += L2 * emb

            # Adam update
            t += 1
            for param, grad, m, v in [
                (emb, demb, m_emb, v_emb),
                (W, dW, m_W, v_W),
                (b, db, m_b, v_b)
            ]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # write weights
    magic = 0x464D4C31  # "FML1"
    out = struct.pack("<4i", magic, d, K, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: one byte class index.
 * Weights: magic 0x464D4C31, d=16, K, C=65; embeddings[d][256][K], W[2K][C], b[C] (all float32 LE). */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 16 || C != 65 || K <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long emb_sz = (long)d * 256 * K;
    long W_sz = (long)2 * K * C;
    long b_sz = C;
    long expected = 16 + 4 * (emb_sz + W_sz + b_sz);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + emb_sz;
    const float *b = W + W_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_v[64], sum_v_sq[64], hidden[128];  // K <= 64 assumed
    unsigned char best;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int kk = 0; kk < K; kk++) { sum_v[kk] = 0.0f; sum_v_sq[kk] = 0.0f; }
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * K;
            for (int kk = 0; kk < K; kk++) {
                float v = e[kk];
                sum_v[kk] += v;
                sum_v_sq[kk] += v * v;
            }
        }
        for (int kk = 0; kk < K; kk++) {
            float inter = 0.5f * (sum_v[kk] * sum_v[kk] - sum_v_sq[kk]);
            hidden[kk] = sum_v[kk];
            hidden[K + kk] = inter;
        }
        // compute logits
        int best_c = 0;
        float best_score = -1e30f;
        for (int c = 0; c < C; c++) {
            float score = b[c];
            for (int h = 0; h < 2 * K; h++) score += hidden[h] * W[(long)h * C + c];
            if (score > best_score) { best_score = score; best_c = c; }
        }
        best = (unsigned char)best_c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```