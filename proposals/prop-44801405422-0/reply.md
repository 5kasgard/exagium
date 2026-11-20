```hypothesis
Family: label-fm (factorization machine). It models pairwise feature interactions via low-rank embeddings, capturing non-linearities that linear models miss, while keeping inference fast via the FM trick (O(d*K) per item). On Adult, categorical features interact strongly (e.g., education × occupation), so FM should beat label-linear. If dev accuracy ≤ label-linear (0.8524) or cost exceeds label-mlp16, the hypothesis fails.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports: int32 magic 0x464D4231, d, K, 1; float32 bias, linear[d][256], emb[d][256][K]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 100


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
    d = X.shape[1]
    C = card["classes"]
    assert C == 2, "Binary classification only"
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Parameters
    bias = np.float32(0.0)
    linear = np.zeros((d, 256), dtype=np.float32)
    emb = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_lin = np.zeros_like(linear)
    v_lin = np.zeros_like(linear)
    m_emb = np.zeros_like(emb)
    v_emb = np.zeros_like(emb)
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx].astype(np.float32)  # (B,)
            B = xb.shape[0]

            # Forward
            linear_part = np.zeros(B, np.float32)
            S = np.zeros((B, K), np.float32)
            sum_sq = np.zeros((B, K), np.float32)
            for i in range(d):
                vals = xb[:, i]
                linear_part += linear[i, vals]
                e = emb[i, vals]          # (B, K)
                S += e
                sum_sq += e * e
            pairwise = 0.5 * (np.sum(S * S, axis=1) - np.sum(sum_sq, axis=1))
            logits = bias + linear_part + pairwise
            p = 1.0 / (1.0 + np.exp(-logits))
            dlogits = p - yb              # (B,)

            # Gradients
            dbias = np.mean(dlogits) + L2 * bias
            dlinear = np.zeros_like(linear)
            demb = np.zeros_like(emb)
            for i in range(d):
                vals = xb[:, i]
                # linear gradient
                for v in range(256):
                    mask = (vals == v)
                    if np.any(mask):
                        dlinear[i, v] = np.mean(dlogits[mask]) + L2 * linear[i, v]
                # embedding gradient
                e = emb[i, vals]          # (B, K)
                # dlogits[:, None] * (S - e)
                grad = dlogits[:, None] * (S - e)   # (B, K)
                for v in range(256):
                    mask = (vals == v)
                    if np.any(mask):
                        demb[i, v] = np.mean(grad[mask], axis=0) + L2 * emb[i, v]

            # Adam update
            t += 1
            # bias
            m_bias = 0.9 * m_bias + 0.1 * dbias
            v_bias = 0.999 * v_bias + 0.001 * dbias * dbias
            bias -= LR * (m_bias / (1 - 0.9 ** t)) / (np.sqrt(v_bias / (1 - 0.999 ** t)) + 1e-8)
            # linear
            m_lin = 0.9 * m_lin + 0.1 * dlinear
            v_lin = 0.999 * v_lin + 0.001 * dlinear * dlinear
            linear -= LR * (m_lin / (1 - 0.9 ** t)) / (np.sqrt(v_lin / (1 - 0.999 ** t)) + 1e-8)
            # emb
            m_emb = 0.9 * m_emb + 0.1 * demb
            v_emb = 0.999 * v_emb + 0.001 * demb * demb
            emb -= LR * (m_emb / (1 - 0.9 ** t)) / (np.sqrt(v_emb / (1 - 0.999 ** t)) + 1e-8)

    # Write weights
    magic = 0x464D4231  # 'FMB1'
    out = struct.pack("<4i", magic, d, K, 1)
    out += np.array([bias], dtype="<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for binary label tasks.
 * ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D4231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 1) { fprintf(stderr, "only binary (C=1) supported\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 * (1 + K));
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *linear = p + 1;
    const float *emb = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *S = (float *)malloc(K * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float score = bias;
        memset(S, 0, K * sizeof(float));
        float sum_sq = 0.0f;
        for (int j = 0; j < d; j++) {
            int val = xi[j];
            score += linear[(long)j * 256 + val];
            const float *e = emb + ((long)j * 256 + val) * K;
            for (int kk = 0; kk < K; kk++) {
                float ek = e[kk];
                S[kk] += ek;
                sum_sq += ek * ek;
            }
        }
        float pairwise = 0.0f;
        for (int kk = 0; kk < K; kk++) pairwise += S[kk] * S[kk];
        pairwise = 0.5f * (pairwise - sum_sq);
        score += pairwise;
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```