```hypothesis
Family: label-emb (embedding-based linear model). Each of the 16 context positions gets a learned 8-dimensional embedding per byte value; the 16 embeddings are summed and fed to a linear output layer (8×65). This captures byte similarities (e.g., letters near each other) with far fewer parameters than one-hot linear (33k vs 266k), should generalize better on small data, and is faster at inference (16×8 adds + 8×65 MACs). If it fails to beat label-linear, the one-hot model's lack of bottleneck is actually helpful for this task.
```

```python
"""label-emb: embedding sum + linear output for 16-byte context, 65 classes.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

E = 8                 # embedding dimension
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
EPOCHS = 10
CLIP = 1.0            # gradient clipping

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
    d, C = X.shape[1], card["classes"]          # d=16, C=65
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # embeddings: (d, 256, E)
    emb = (rng.normal(0, 1.0 / math.sqrt(E), (d, 256, E))).astype(np.float32)
    # output layer: (E, C) and (C,)
    W = (rng.normal(0, 1.0 / math.sqrt(E), (E, C))).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = [emb, W, b]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    idx = np.arange(n)
    for _ in range(EPOCHS):
        rng.shuffle(idx)
        for s in range(0, n, BATCH):
            batch_idx = idx[s:s + BATCH]
            xb = X[batch_idx]          # (B, d)
            yb = y[batch_idx]          # (B,)
            B = len(yb)

            # forward: sum embeddings
            # emb_sum[b, e] = sum_j emb[j, xb[b, j], e]
            emb_sum = np.zeros((B, E), np.float32)
            for j in range(d):
                emb_sum += emb[j, xb[:, j]]       # (B, E)

            logits = emb_sum @ W + b               # (B, C)
            # stable softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            probs /= B                               # (B, C) gradient wrt logits

            # gradients
            grad_W = emb_sum.T @ probs + L2 * W       # (E, C)
            grad_b = probs.sum(axis=0)                # (C,)
            grad_emb_sum = probs @ W.T                # (B, E)

            # backprop to embeddings
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                # accumulate grad_emb_sum[b] into grad_emb[j, xb[b, j]]
                np.add.at(grad_emb[j], xb[:, j], grad_emb_sum)
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W, grad_b]

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights.bin
    # magic 0x454D424C ('L','B','M','E' little-endian), d, E, C
    header = struct.pack("<4i", 0x454D424C, d, E, C)
    emb_bytes = emb.astype("<f4").tobytes()
    W_bytes = W.astype("<f4").tobytes()
    b_bytes = b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + emb_bytes + W_bytes + b_bytes)

if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 16 || E != 8 || C != 65) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * E + (long)E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + (long)E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * E);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int e = 0; e < E; e++) acc[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) acc[e] += eptr[e];
        }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int e = 0; e < E; e++) {
            float ae = acc[e];
            const float *wptr = W + (long)e * C;
            for (int c = 0; c < C; c++) scores[c] += ae * wptr[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```