```hypothesis
Family: label-embed. Learns a dense embedding (size E) for each of the 256 values of every feature, then applies a linear layer. All 14 features are quantized to 256 bins, so embeddings capture non‑linear ordinal effects and categorical similarities that label‑linear’s one‑hot/z‑score cannot. Inference is a fast gather‑add + matrix‑vector multiply. Expect dev accuracy > 0.8524 (label‑linear) with Ir/item well below label‑mlp16 (2325). Failure if accuracy ≤ 0.8524 or Ir/item ≥ 2325.
```
```python
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
E_CANDIDATES = [2, 4, 8, 16]
MAGIC = 0x454D424C  # "EMBL" little‑endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_model(X, y, d, C, E, rng, n_epochs):
    embeddings = [rng.normal(0, 0.01, (256, E)).astype(np.float32) for _ in range(d)]
    W = rng.normal(0, 0.01, (d * E, C)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = embeddings + [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]
    for _ in range(n_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            emb_concat = np.concatenate([embeddings[j][xb[:, j]] for j in range(d)], axis=1)
            logits = emb_concat @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_logits = probs / len(yb)
            grad_W = emb_concat.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
            grad_emb_concat = grad_logits @ W.T
            grad_embeddings = []
            for j in range(d):
                grad_emb_j = np.zeros((256, E), np.float32)
                np.add.at(grad_emb_j, xb[:, j], grad_emb_concat[:, j * E:(j + 1) * E])
                grad_emb_j += L2 * embeddings[j]
                grad_embeddings.append(grad_emb_j)
            grads = grad_embeddings + [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9**t)
                v_hat = v / (1 - 0.999**t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return embeddings, W, b


def evaluate(X, y, embeddings, W, b, d, C, E):
    emb_concat = np.concatenate([embeddings[j][X[:, j]] for j in range(d)], axis=1)
    logits = emb_concat @ W + b
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    n = X.shape[0]
    split = n - n // 5
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]

    best_E, best_acc = None, -1.0
    for E in E_CANDIDATES:
        n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))
        embeddings, W, b = train_model(X_train, y_train, d, C, E, rng, n_epochs)
        acc = evaluate(X_tune, y_tune, embeddings, W, b, d, C, E)
        if acc > best_acc:
            best_acc, best_E = acc, E

    n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    embeddings, W, b = train_model(X, y, d, C, best_E, rng, n_epochs)

    out = struct.pack("<4i", MAGIC, d, best_E, C)
    for j in range(d):
        out += embeddings[j].astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
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
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1 || E < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)d * E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + (long)d * E * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * (size_t)d * E);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < d * E; w++) acc[w] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int u = 0; u < E; u++) acc[j * E + u] = e[u];
        }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int w = 0; w < d * E; w++) {
            float a = acc[w];
            if (a != 0.0f) {
                const float *wrow = W + (long)w * C;
                for (int c = 0; c < C; c++) scores[c] += a * wrow[c];
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```