```hypothesis
label-fm: Factorization Machine with rank-k embeddings per feature value. Captures pairwise interactions efficiently (O(d·k) params, O(d·k) inference) which should help on Adult where feature crosses matter (e.g., education×occupation). Uses raw 14-byte features directly (256 values each). If it doesn't beat linear, interactions aren't predictive or rank is too low.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Chooses rank k on last 20% of train, then trains on all.
Exported: int32 magic 0x464D4C31, d, k, C; float32 w0, w[d][256], V[d][256][k]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
K_GRID = [4, 8, 16, 32]  # ranks to try on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def fm_forward(X, w0, w, V):
    """X: (B, d) uint8. Returns logits (B,)."""
    B, d = X.shape
    k = V.shape[2]
    # linear part
    lin = w0 + np.sum(w[np.arange(d), X.T], axis=0)  # (B,)
    # interaction part
    inter = np.zeros(B, dtype=np.float32)
    for f in range(k):
        Vf = V[:, :, f]  # (d, 256)
        vals = Vf[np.arange(d), X.T]  # (d, B)
        sum_f = np.sum(vals, axis=0)  # (B,)
        sum_sq_f = np.sum(vals * vals, axis=0)
        inter += (sum_f * sum_f - sum_sq_f) * 0.5
    return lin + inter


def train_fm(X, y, k, rng, epochs):
    """Train FM on (X, y) for given epochs. Returns (w0, w, V)."""
    n, d = X.shape
    C = 2
    # init
    w0 = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    V = (rng.normal(0, 0.01, (d, 256, k)) * math.sqrt(2.0 / (d * 256))).astype(np.float32)
    params = [w0, w, V]
    # Adam state
    states = []
    for p in params:
        if p.ndim == 0:
            states.append((np.float32(0.0), np.float32(0.0)))
        else:
            states.append((np.zeros_like(p), np.zeros_like(p)))
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            B = len(yb)
            # forward
            logits = fm_forward(xb, w0, w, V)
            # logistic loss gradient
            p = 1.0 / (1.0 + np.exp(-logits))
            p[np.arange(B), yb] -= 1.0
            g = p / B  # (B,)
            # gradients
            gw0 = np.float32(g.sum())
            gw = np.zeros_like(w)
            gV = np.zeros_like(V)
            for f in range(k):
                Vf = V[:, :, f]
                vals = Vf[np.arange(d), xb.T]  # (d, B)
                sum_f = np.sum(vals, axis=0)  # (B,)
                # dL/dV_{i,v,f} = g * (sum_f - V_{i,v,f})
                for i in range(d):
                    vi = vals[i]  # (B,)
                    gw[i, xb[:, i]] += g
                    gV[i, xb[:, i], f] += g * (sum_f - vi)
            gw += L2 * w
            gV += L2 * V
            grads = [gw0, gw, gV]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = states[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                if p.ndim == 0:
                    p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
                else:
                    p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return w0, w, V


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = len(y)
    # tuning split: last 20%
    split = int(n * 0.8)
    rng = np.random.default_rng(SEED)
    best_k, best_acc = K_GRID[0], -1.0
    epochs_tune = min(MAX_EPOCHS, math.ceil(SAMPLES / split))
    for k in K_GRID:
        w0, w, V = train_fm(X[:split], y[:split], k, np.random.default_rng(SEED), epochs_tune)
        logits = fm_forward(X[split:], w0, w, V)
        preds = (logits > 0).astype(np.int64)
        acc = (preds == y[split:]).mean()
        if acc > best_acc:
            best_acc, best_k = acc, k
    # retrain on all data
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    w0, w, V = train_fm(X, y, best_k, np.random.default_rng(SEED), epochs_full)
    # save
    out = struct.pack("<4i", 0x464D4C31, d, best_k, C)
    out += struct.pack("<f", w0)
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: 1 byte class (0 or 1). */
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
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d < 1 || k < 1 || k > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L + 4L * d * 256 + 4L * d * 256 * k;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }
    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *V = p + 1 + d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }
        float logit = w0;
        for (int j = 0; j < d; j++) logit += w[j * 256 + xi[j]];
        for (int f = 0; f < k; f++) {
            float sum_f = 0.0f, sum_sq_f = 0.0f;
            for (int j = 0; j < d; j++) {
                float v = V[(j * 256 + xi[j]) * k + f];
                sum_f += v;
                sum_sq_f += v * v;
            }
            logit += (sum_f * sum_f - sum_sq_f) * 0.5f;
        }
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```