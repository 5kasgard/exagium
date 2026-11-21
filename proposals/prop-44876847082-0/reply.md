```hypothesis
We propose a Factorization Machine (FM) with a linear term and a low-rank (K=4) interaction term.
Adult income prediction benefits from pairwise feature interactions (e.g., education × occupation).
The linear baseline reaches 85.24%; adding cheap pairwise interactions should lift accuracy
while keeping inference cost near linear (only 14·256·4 extra floats). If dev accuracy does not
exceed label-linear by at least 0.2% (i.e., >85.44%), the hypothesis fails.
```

```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x464D4C31, d, K, C; float32 bias[C], w_fm[C],
         linear_weight[C][d][256], embedding[d][256][K]."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-3, 1e-4, 64, 600000, 100


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

    rng = np.random.default_rng(SEED)

    # Parameters
    bias = np.zeros(C, np.float32)
    w_fm = np.zeros(C, np.float32)
    linear_w = np.zeros((C, d, 256), np.float32)
    emb = (rng.uniform(-1, 1, (d, 256, K)) * math.sqrt(6.0 / (d * 256))).astype(np.float32)

    params = [bias, w_fm, linear_w, emb]
    shapes = [p.shape for p in params]
    # Adam state
    m = [np.zeros(s, np.float32) for s in shapes]
    v = [np.zeros(s, np.float32) for s in shapes]

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward
            logits = bias[None, :] + w_fm[None, :] * 0.0  # (B, C) will add FM later
            # Linear part
            for j in range(d):
                logits += linear_w[:, j, xb[:, j]].T   # (B, C)

            # FM part
            # emb_j: (B, K) for each feature
            sum_emb = np.zeros((B, K), np.float32)
            sum_emb_sq = np.zeros((B, K), np.float32)
            for j in range(d):
                e = emb[j, xb[:, j]]       # (B, K)
                sum_emb += e
                sum_emb_sq += e * e
            fm = 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=1)  # (B,)
            logits += w_fm[None, :] * fm[:, None]                      # (B, C)

            # Softmax + cross-entropy gradient
            max_logits = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            dlogits = probs.astype(np.float32) / B   # (B, C)

            # Gradients
            grad_bias = np.sum(dlogits, axis=0)                     # (C,)
            grad_w_fm = np.sum(dlogits * fm[:, None], axis=0)       # (C,)

            grad_linear = np.zeros_like(linear_w)
            for j in range(d):
                v = xb[:, j]
                for c in range(C):
                    np.add.at(grad_linear[c, j], v, dlogits[:, c])

            grad_emb = np.zeros_like(emb)
            # g_fm = dlogits @ w_fm  (B,)
            g_fm = dlogits @ w_fm
            for j in range(d):
                e = emb[j, xb[:, j]]               # (B, K)
                # grad for this feature = g_fm[:, None] * (sum_emb - e)
                g = g_fm[:, None] * (sum_emb - e)   # (B, K)
                v = xb[:, j]
                for k in range(K):
                    np.add.at(grad_emb[j, :, k], v, g[:, k])

            # L2 regularization
            grad_bias += L2 * bias
            grad_w_fm += L2 * w_fm
            grad_linear += L2 * linear_w
            grad_emb += L2 * emb

            grads = [grad_bias, grad_w_fm, grad_linear, grad_emb]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] *= 0.9; m[i] += 0.1 * gr
                v[i] *= 0.999; v[i] += 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    # header: magic, d, K, C
    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    out += bias.astype("<f4").tobytes()
    out += w_fm.astype("<f4").tobytes()
    out += linear_w.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte: the argmax class. */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1 || K < 1) { fprintf(stderr, "bad dimensions\n"); return 3; }

    long expected_floats = 2L*C + (long)C*d*256 + (long)d*256*K;
    long expected_bytes = 16 + 4 * expected_floats;
    if (nw != expected_bytes) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *bias = p; p += C;
    const float *w_fm = p; p += C;
    const float *linear_w = p; p += (long)C * d * 256;
    const float *emb = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * K);
    float *sum_emb_sq = malloc(sizeof(float) * K);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) logits[c] = bias[c];
        memset(sum_emb, 0, sizeof(float) * K);
        memset(sum_emb_sq, 0, sizeof(float) * K);

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lw = linear_w + ((long)j * 256 + v) * C;
            for (int c = 0; c < C; c++) logits[c] += lw[c];

            const float *e = emb + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = e[kk];
                sum_emb[kk] += val;
                sum_emb_sq[kk] += val * val;
            }
        }

        float fm = 0.0f;
        for (int kk = 0; kk < K; kk++) fm += sum_emb[kk] * sum_emb[kk] - sum_emb_sq[kk];
        fm *= 0.5f;

        for (int c = 0; c < C; c++) logits[c] += w_fm[c] * fm;

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```