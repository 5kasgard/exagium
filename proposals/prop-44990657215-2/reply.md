```hypothesis
Family: label-mlp (1-hidden-layer MLP with 64 units and per-position byte embeddings).
Why: Character-level language modeling benefits from modeling pairwise interactions between context positions; a 64-unit ReLU layer over learned embeddings captures this better than linear (H=0) or the smaller label-mlp16 (H=16), while staying far cheaper than GBDT.
Failure signal: If dev accuracy ≤ 0.34, the hidden size or training schedule is insufficient for this task.
```

```python
"""label-mlp: 1-hidden-layer MLP with per-position byte embeddings for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin"""
import json, math, struct, sys

import numpy as np

H = 64
SEED = 1
LR = 1e-3
LR_MIN = 1e-5
L2 = 1e-4
BATCH = 256
SAMPLES = 1200000
MAX_EPOCHS = 200


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = b"".join(records(sys.argv[2]))
    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j]
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(v)
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)

    Z_parts = [E[X[:, j]] for j, E in enumerate(enc)]
    Z = np.concatenate(Z_parts, axis=1).astype(np.float32)
    k = Z.shape[1]
    W = H

    W1 = (rng.normal(0, math.sqrt(2.0 / k), (k, W))).astype(np.float32)
    b1 = np.zeros(W, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / W), (W, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    steps_per_epoch = (n + BATCH - 1) // BATCH
    total_steps = min(MAX_EPOCHS * steps_per_epoch, (SAMPLES + BATCH - 1) // BATCH)
    t = 0
    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            if t >= total_steps:
                break
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]

            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2

            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)

            grad_W2 = h.T @ g + L2 * W2
            grad_b2 = g.sum(axis=0)
            gh = g @ W2.T * (a > 0)
            grad_W1 = z.T @ gh + L2 * W1
            grad_b1 = gh.sum(axis=0)

            grads = [grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            lr = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * t / total_steps))
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        if t >= total_steps:
            break

    T_list = []
    off = 0
    for E in enc:
        n_vals = E.shape[1]
        Tj = (E @ W1[off:off + n_vals]).astype(np.float32)
        T_list.append(Tj)
        off += n_vals

    magic = 0x54424C31
    out = struct.pack("<4i", magic, d, H, C)
    out += np.stack(T_list).astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-mlp inference: ./infer weights.bin in.rec N
 * Table family (one ReLU layer) for label tasks with fixed-size byte items. */
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
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0))) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4), *b1 = T + (long)d * 256 * W, *W2 = b1 + W, *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W), *s = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        const float *sc = acc;
        if (H) {
            for (int c = 0; c < C; c++) s[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) s[c] += h * W2[(long)u * C + c];
            }
            sc = s;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```