```hypothesis
Family: label-embed. Learns per-position byte embeddings (16×256×D), sums them, applies ReLU and a linear output layer. The low-rank factorization regularizes the model and reduces inference cost (D≪4096). Hyperparameter D is chosen on the tuning split. Failure: dev accuracy ≤ 0.3375 or ns/item ≥ 665.8.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
LR = 0.01
L2 = 1e-5
BATCH = 64
MAX_EPOCHS = 20

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]

    n = X.shape[0]
    tune_size = n // 5
    train_X, tune_X = X[:-tune_size], X[-tune_size:]
    train_y, tune_y = y[:-tune_size], y[-tune_size:]

    best_D = 16
    best_acc = -1
    rng = np.random.default_rng(SEED)

    for D in [8, 16, 32, 64]:
        E = (rng.uniform(-1, 1, (d, 256, D)) * math.sqrt(6.0 / 256)).astype(np.float32)
        b1 = np.zeros(D, np.float32)
        W2 = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
        b2 = np.zeros(C, np.float32)

        params = [E, b1, W2, b2]
        state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

        for epoch in range(5):
            order = rng.permutation(len(train_X))
            for s in range(0, len(train_X), BATCH):
                idx = order[s:s+BATCH]
                x = train_X[idx]
                yy = train_y[idx]
                h = np.zeros((len(idx), D), np.float32)
                for j in range(d):
                    h += E[j, x[:, j]]
                h += b1
                h = np.maximum(h, 0)
                logits = h @ W2 + b2
                logits -= logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits)
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                probs[np.arange(len(yy)), yy] -= 1.0
                grad_logits = probs / len(yy)
                grad_h = grad_logits @ W2.T
                grad_h *= (h > 0)
                grad_W2 = h.T @ grad_logits + L2 * W2
                grad_b2 = grad_logits.sum(0)
                grad_b1 = grad_h.sum(0)
                grad_E = np.zeros_like(E)
                for j in range(d):
                    np.add.at(grad_E[j], x[:, j], grad_h)
                grad_E += L2 * E
                grads = [grad_E, grad_b1, grad_W2, grad_b2]
                t = epoch * (len(train_X)//BATCH) + s//BATCH + 1
                for i, (p, gr) in enumerate(zip(params, grads)):
                    m, v = state[i]
                    m *= 0.9; m += 0.1 * gr
                    v *= 0.999; v += 0.001 * gr * gr
                    m_hat = m / (1 - 0.9 ** t)
                    v_hat = v / (1 - 0.999 ** t)
                    p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        h = np.zeros((len(tune_X), D), np.float32)
        for j in range(d):
            h += E[j, tune_X[:, j]]
        h += b1
        h = np.maximum(h, 0)
        logits = h @ W2 + b2
        preds = logits.argmax(axis=1)
        acc = (preds == tune_y).mean()
        if acc > best_acc:
            best_acc = acc
            best_D = D

    D = best_D
    rng = np.random.default_rng(SEED)
    E = (rng.uniform(-1, 1, (d, 256, D)) * math.sqrt(6.0 / 256)).astype(np.float32)
    b1 = np.zeros(D, np.float32)
    W2 = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [E, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            x = X[idx]
            yy = y[idx]
            h = np.zeros((len(idx), D), np.float32)
            for j in range(d):
                h += E[j, x[:, j]]
            h += b1
            h = np.maximum(h, 0)
            logits = h @ W2 + b2
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad_logits = probs / len(yy)
            grad_h = grad_logits @ W2.T
            grad_h *= (h > 0)
            grad_W2 = h.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(0)
            grad_b1 = grad_h.sum(0)
            grad_E = np.zeros_like(E)
            for j in range(d):
                np.add.at(grad_E[j], x[:, j], grad_h)
            grad_E += L2 * E
            grads = [grad_E, grad_b1, grad_W2, grad_b2]
            t = epoch * (n//BATCH) + s//BATCH + 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    magic = 0x454D4244
    out = struct.pack("<4i", magic, d, D, C)
    out += E.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
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
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * D + D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *E = (const float *)(hd + 4);
    const float *b1 = E + (long)d * 256 * D;
    const float *W2 = b1 + D;
    const float *b2 = W2 + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int u = 0; u < D; u++) h[u] = b1[u];
        for (int j = 0; j < d; j++) {
            const float *emb = E + ((long)j * 256 + xi[j]) * D;
            for (int u = 0; u < D; u++) h[u] += emb[u];
        }
        for (int u = 0; u < D; u++) if (h[u] < 0.0f) h[u] = 0.0f;
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < D; u++) sum += h[u] * W2[(long)u * C + c];
            scores[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```