```hypothesis
We propose a factorized embedding model (label-emb) that replaces the per-position one-hot linear weights with low-rank embeddings. Each of the 16 context positions has a 256×E embedding table; the 16 E-dimensional vectors are summed and multiplied by a shared E×C matrix to produce class scores. This reduces parameters (≈164k for E=32 vs 266k for linear) and captures character similarities, which should improve generalization on the small tinyshakespeare dataset. The model is trained with Adam on a tuning split (last 20% of train) to select E ∈ {16,32,64,128}, then retrained on all data. Inference cost is O(d·E + E·C) per item, comparable to label-linear. If dev accuracy does not exceed 0.34, the factorization is too restrictive for this task.
```
```python
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
CANDIDATE_E = [16, 32, 64, 128]
MAGIC = 0x4C42454D  # "L B E M" little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_model(X, y, C, E, epochs, rng):
    d = X.shape[1]
    embeddings = []
    for j in range(d):
        limit = math.sqrt(6.0 / 256)
        emb = rng.uniform(-limit, limit, (256, E)).astype(np.float32)
        embeddings.append(emb)
    limit = math.sqrt(6.0 / E)
    V = rng.uniform(-limit, limit, (E, C)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = embeddings + [V, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb = X[idx]
            yb = y[idx]
            h = np.zeros((len(yb), E), np.float32)
            for j in range(d):
                h += embeddings[j][Xb[:, j]]
            scores = h @ V + b
            scores_max = scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_scores = probs / len(yb)
            grad_b = grad_scores.sum(axis=0)
            grad_V = h.T @ grad_scores
            grad_h = grad_scores @ V.T
            grad_embeddings = []
            for j in range(d):
                grad_emb = np.zeros((256, E), np.float32)
                np.add.at(grad_emb, Xb[:, j], grad_h)
                grad_emb += L2 * embeddings[j]
                grad_embeddings.append(grad_emb)
            grad_V += L2 * V
            grads = grad_embeddings + [grad_V, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return embeddings, V, b


def evaluate(X, y, C, embeddings, V, b):
    d = X.shape[1]
    E = V.shape[0]
    h = np.zeros((X.shape[0], E), np.float32)
    for j in range(d):
        h += embeddings[j][X[:, j]]
    scores = h @ V + b
    preds = scores.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)
    n = X.shape[0]
    split = int(0.8 * n)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]
    epochs_tune = min(MAX_EPOCHS, math.ceil(SAMPLES / split))
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    best_E = None
    best_acc = -1.0
    for E in CANDIDATE_E:
        embeddings, V, b = train_model(X_train, y_train, C, E, epochs_tune, rng)
        acc = evaluate(X_tune, y_tune, C, embeddings, V, b)
        if acc > best_acc:
            best_acc = acc
            best_E = E
    embeddings, V, b = train_model(X, y, C, best_E, epochs_full, rng)
    out = struct.pack("<4i", MAGIC, d, best_E, C)
    for j in range(d):
        out += embeddings[j].astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x4C42454D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * E + (long)E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 4);
    const float *V = emb + (long)d * 256 * E;
    const float *b = V + (long)E * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * E);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int e = 0; e < E; e++) acc[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *evec = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) acc[e] += evec[e];
        }
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int e = 0; e < E; e++) sum += acc[e] * V[(long)e * C + c];
            scores[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```