```hypothesis
Family: label-emb (per-column embeddings + linear head). Treats all 14 columns as categorical (256 bins each), learns a dense embedding per column, concatenates them, and applies a linear classifier. This captures feature interactions implicitly and reduces dimensionality vs. one-hot linear. Adult has many high-cardinality categoricals where embeddings should generalize better. If dev accuracy ≤ 0.8524 (label-linear), the hypothesis fails.
```

```python
"""label-emb: per-column embeddings + linear classifier.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Weights format: magic 0x454D4231, d, E, C; float32 embeddings[d][256][E], W2[d*E][C], b2[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
EMB_DIM_CHOICES = [4, 8, 16, 32]  # tuned on last 20% of train.in


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_model(X_tr, y_tr, X_val, y_val, emb_dim, rng_seed):
    n, d = X_tr.shape
    C = 2
    rng = np.random.default_rng(rng_seed)
    # embeddings: list of d arrays (256, emb_dim)
    emb = [(rng.uniform(-1, 1, (256, emb_dim)) * math.sqrt(6.0 / (256 + emb_dim))).astype(np.float32)
           for _ in range(d)]
    W2 = (rng.uniform(-1, 1, (d * emb_dim, C)) * math.sqrt(6.0 / (d * emb_dim))).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = emb + [W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def forward(Xb):
        # Xb: (batch, d) uint8
        batch = Xb.shape[0]
        cat = np.empty((batch, d * emb_dim), np.float32)
        for j in range(d):
            cat[:, j*emb_dim:(j+1)*emb_dim] = emb[j][Xb[:, j]]
        scores = cat @ W2 + b2
        return cat, scores

    def loss_grad(scores, yb):
        probs = np.exp(scores - scores.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        probs[np.arange(len(yb)), yb] -= 1.0
        return probs.astype(np.float32) / len(yb)

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            cat, scores = forward(Xb)
            g_scores = loss_grad(scores, yb)
            # gradients
            g_W2 = cat.T @ g_scores + L2 * W2
            g_b2 = g_scores.sum(axis=0)
            g_cat = g_scores @ W2.T  # (batch, d*emb_dim)
            g_emb = [np.zeros_like(e) for e in emb]
            for j in range(d):
                gj = g_cat[:, j*emb_dim:(j+1)*emb_dim]
                np.add.at(g_emb[j], Xb[:, j], gj)
                g_emb[j] += L2 * emb[j]
            grads = g_emb + [g_W2, g_b2]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # validation accuracy
    cat_val, scores_val = forward(X_val)
    pred = scores_val.argmax(axis=1)
    acc = (pred == y_val).mean()
    return acc, emb, W2, b2


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    best_acc, best_emb_dim, best_params = -1, None, None
    for i, E in enumerate(EMB_DIM_CHOICES):
        # deterministic seed per trial
        trial_seed = SEED + 1000 * (i + 1)
        acc, emb, W2, b2 = train_model(X_tr, y_tr, X_val, y_val, E, trial_seed)
        if acc > best_acc:
            best_acc, best_emb_dim, best_params = acc, E, (emb, W2, b2)

    # retrain on full data with best_emb_dim
    final_seed = SEED + 9999
    _, emb, W2, b2 = train_model(X, y, X[:1], y[:1], best_emb_dim, final_seed)  # dummy val

    # write weights.bin
    magic, d, E, C = 0x454D4231, d, best_emb_dim, C
    header = struct.pack("<4i", magic, d, E, C)
    emb_bytes = b"".join(e.astype("<f4").tobytes() for e in emb)
    W2_bytes = W2.astype("<f4").tobytes()
    b2_bytes = b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + emb_bytes + W2_bytes + b2_bytes)


if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: d bytes (one per column). Output: one byte = argmax class.
 * Weights format: magic 0x454D4231, int32 d, E, C; float32 embeddings[d][256][E], W2[d*E][C], b2[C]. */
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x454D4231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 14) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16L
        + 4L * d * 256 * E
        + 4L * d * E * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W2 = emb + (long)d * 256 * E;
    const float *b2 = W2 + (long)d * E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * E;
            float *dst = concat + j * E;
            memcpy(dst, src, sizeof(float) * E);
        }
        for (int c = 0; c < C; c++) scores[c] = b2[c];
        for (int k = 0; k < d * E; k++) {
            float v = concat[k];
            if (v == 0.0f) continue;
            for (int c = 0; c < C; c++) scores[c] += v * W2[(long)k * C + c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```