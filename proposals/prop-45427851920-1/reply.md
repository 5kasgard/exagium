```hypothesis
Family: label-catlin (categorical linear). Pixels are 0..16; treating them as categorical (one-hot) instead of ordinal captures non-monotonic intensity effects. Expects ~0.97 dev accuracy (beating label-linear's 0.9613) at similar or lower CPU cost: 64*10=640 adds/item vs label-linear's 64*10, but with 16x smaller weight tables (64*17*10 vs 64*256*10) for better cache usage. Failure: accuracy ≤ 0.9613 or instruction count ≥ label-linear.
```
```python
"""label-catlin: linear classifier on one-hot pixel values (0..16).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
V = 17  # pixel values 0..16


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # one-hot design matrix: n x (d*V)
    Z = np.zeros((n, d * V), np.float32)
    for i in range(n):
        for j in range(d):
            v = X[i, j]
            if v < V:
                Z[i, j * V + v] = 1.0

    W = (rng.uniform(-1, 1, (d * V, C)) * math.sqrt(6.0 / (d * V))).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            logits = z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)
            grad_W = z.T @ g + L2 * W
            grad_b = g.sum(0)
            t += 1
            for p, gr, (m, v) in zip(params, [grad_W, grad_b], state):
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # export compact format: magic, d, V, C, table[d][V][C], bias[C]
    magic = 0x4341544C  # 'CATL'
    W_table = W.reshape(d, V, C).astype(np.float32)
    bias = b.astype(np.float32)
    out = struct.pack("<4I", magic, d, V, C)
    out += W_table.tobytes()
    out += bias.tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-catlin inference: ./infer weights.bin in.rec N
 * weights: magic(0x4341544C), d, V, C, table[d][V][C] float32, bias[C] float32
 * input record: d bytes (pixels 0..16). output record: 1 byte class. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x4341544C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], V = hd[2], C = hd[3];
    if (C < 1 || C > 256 || V != 17 || d != 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + (long)d * V * C * 4 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *table = (const float *)(hd + 4);
    const float *bias = table + (long)d * V * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            if (v >= V) continue;
            const float *w = table + ((long)j * V + v) * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```