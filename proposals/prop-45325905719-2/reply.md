```hypothesis
Family: label-embed1h (position-specific embeddings + one ReLU layer).
Why: The 16-byte context has strong positional patterns; factorizing the first layer into 16 lookup tables (256×H each) reduces parameters vs. one-hot linear (4096×65) and captures interactions via the summed embedding + ReLU. Inference is 16×H adds + H×65 MACs (≈10k ops for H=128) vs. 266k for label-linear, so it should be faster and more accurate.
Failure: If dev accuracy ≤ 0.3375 or instructions/item ≥ label-linear, the factorization is not helping.
```

```python
"""label-embed1h: position-specific embeddings + one ReLU layer.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x454D4231, d, H, C; float32 E[d][256][H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
H_CHOICES = [64, 128, 256]  # tuned on last 20% of train


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_model(X, y, H, rng, epochs, verbose=False):
    d = X.shape[1]
    C = int(y.max()) + 1
    # embeddings: d x 256 x H
    E = (rng.normal(0, 0.01, (d, 256, H))).astype(np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [E, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            # forward
            h = np.zeros((len(yb), H), np.float32)
            for j in range(d):
                h += E[j, xb[:, j]]
            h_relu = np.maximum(h, 0)
            scores = h_relu @ W2 + b2
            # softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            probs /= len(yb)
            # backward
            dW2 = h_relu.T @ probs + L2 * W2
            db2 = probs.sum(0)
            dh = probs @ W2.T * (h > 0)
            dE = np.zeros_like(E)
            for j in range(d):
                np.add.at(dE[j], xb[:, j], dh)
            dE += L2 * E
            grads = [dE, dW2, db2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return E, W2, b2


def evaluate(X, y, E, W2, b2):
    d = X.shape[1]
    h = np.zeros((X.shape[0], W2.shape[0]), np.float32)
    for j in range(d):
        h += E[j, X[:, j]]
    h = np.maximum(h, 0)
    scores = h @ W2 + b2
    pred = scores.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)
    n = X.shape[0]
    tune_start = int(0.8 * n)
    X_tr, y_tr = X[:tune_start], y[:tune_start]
    X_va, y_va = X[tune_start:], y[tune_start:]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_tr)))
    best_H, best_acc = H_CHOICES[0], -1.0
    for H in H_CHOICES:
        rng_i = np.random.default_rng(SEED + H)  # deterministic per H
        E, W2, b2 = train_model(X_tr, y_tr, H, rng_i, epochs)
        acc = evaluate(X_va, y_va, E, W2, b2)
        if acc > best_acc:
            best_acc, best_H = acc, H
    # final training on all data
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    rng_final = np.random.default_rng(SEED + best_H)
    E, W2, b2 = train_model(X, y, best_H, rng_final, epochs_full)
    # export
    out = struct.pack("<4i", 0x454D4231, d, best_H, C)
    out += E.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embed1h infer.c: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic 0x454D4231, int32 d, H, C; float32 E[d][256][H], W2[H][C], b2[C]. */
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
    if (nw < 16 || hd[0] != 0x454D4231) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (d != 16 || C < 1 || C > 256 || H < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *E = (const float *)(hd + 4);
    const float *W2 = E + (long)d * 256 * H;
    const float *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(h, 0, sizeof(float) * H);
        for (int j = 0; j < d; j++) {
            const float *ej = E + ((long)j * 256 + xi[j]) * H;
            for (int u = 0; u < H; u++) h[u] += ej[u];
        }
        for (int u = 0; u < H; u++) if (h[u] < 0.0f) h[u] = 0.0f;
        for (int c = 0; c < C; c++) scores[c] = b2[c];
        for (int u = 0; u < H; u++) {
            float hu = h[u];
            if (hu == 0.0f) continue;
            const float *w2u = W2 + (long)u * C;
            for (int c = 0; c < C; c++) scores[c] += hu * w2u[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```