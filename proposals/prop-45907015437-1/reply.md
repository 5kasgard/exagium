```hypothesis
Family: label-embed (shallow embedding model). Each of the 64 pixel positions gets a learned embedding vector for each of the 17 possible values (0..16). The embeddings are summed and fed to a linear output layer. This captures position‑value interactions more efficiently than a linear model on ordinal features, while staying far cheaper than an MLP or GBDT. If dev accuracy ≤ 0.9613 (label‑linear) or inference cost exceeds label‑linear by >2×, the approach does not work.
```
```python
"""Shallow embedding model for optdigits: 64 positions × 17 values → E‑dim embeddings → linear 10‑way classifier.
python3 -s -P train.py card.json train.in train.key weights.bin  (deterministic, seed 1, single thread)
Weights format: int32 magic 0x454D4245, d, V, E, C; float32 emb[d][V][E], W_out[E][C], b_out[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_model(X, y, E, lr, l2, batch_size, epochs, seed):
    n, d = X.shape
    C = 10
    V = 17
    rng = np.random.default_rng(seed)
    emb = rng.normal(0, 0.01, size=(d, V, E)).astype(np.float32)
    W_out = rng.normal(0, 0.01, size=(E, C)).astype(np.float32)
    b_out = np.zeros(C, dtype=np.float32)
    params = [emb, W_out, b_out]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            x_batch = X[idx]
            y_batch = y[idx]
            B = len(idx)
            hidden = np.zeros((B, E), dtype=np.float32)
            for j in range(d):
                hidden += emb[j, x_batch[:, j]]
            logits = hidden @ W_out + b_out
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), y_batch] -= 1.0
            grad_logits = probs / B
            grad_W_out = hidden.T @ grad_logits + l2 * W_out
            grad_b_out = grad_logits.sum(axis=0)
            grad_hidden = grad_logits @ W_out.T
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], x_batch[:, j], grad_hidden)
            grad_emb += l2 * emb
            t += 1
            grads = [grad_emb, grad_W_out, grad_b_out]
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return params


def evaluate(X, y, params):
    emb, W_out, b_out = params
    n, d = X.shape
    E = emb.shape[2]
    hidden = np.zeros((n, E), dtype=np.float32)
    for j in range(d):
        hidden += emb[j, X[:, j]]
    logits = hidden @ W_out + b_out
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    V = 17
    n = len(X)
    val_size = n // 5
    train_X, train_y = X[:-val_size], y[:-val_size]
    val_X, val_y = X[-val_size:], y[-val_size:]

    best_acc = -1.0
    best_params = None
    best_hp = None
    for E in [8, 16, 32]:
        for lr in [1e-3, 3e-3]:
            for l2 in [1e-4, 1e-5]:
                for epochs in [30, 50]:
                    params = train_model(train_X, train_y, E, lr, l2, 64, epochs, SEED)
                    acc = evaluate(val_X, val_y, params)
                    if acc > best_acc:
                        best_acc = acc
                        best_params = params
                        best_hp = (E, lr, l2, epochs)
    print(f"Best hp: E={best_hp[0]}, lr={best_hp[1]}, l2={best_hp[2]}, epochs={best_hp[3]}, val_acc={best_acc:.4f}", file=sys.stderr)

    E, lr, l2, epochs = best_hp
    final_params = train_model(X, y, E, lr, l2, 64, epochs, SEED)
    emb, W_out, b_out = final_params

    magic = 0x454D4245
    out = struct.pack("<5i", magic, d, V, E, C)
    out += emb.astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    out += b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Shallow embedding model for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels 0..16). Output: one byte class (argmax). */
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
    if (nw < 20 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], V = hd[2], E = hd[3], C = hd[4];
    if (C < 1 || C > 256 || V != 17 || d != 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * V * E + (long)E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 5);
    const float *W_out = emb + (long)d * V * E;
    const float *b_out = W_out + (long)E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * E);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int e = 0; e < E; e++) hidden[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            if (v >= V) v = V - 1;
            const float *e = emb + ((long)j * V + v) * E;
            for (int u = 0; u < E; u++) hidden[u] += e[u];
        }
        for (int c = 0; c < C; c++) scores[c] = b_out[c];
        for (int u = 0; u < E; u++) {
            float hu = hidden[u];
            if (hu == 0.0f) continue;
            const float *w = W_out + (long)u * C;
            for (int c = 0; c < C; c++) scores[c] += hu * w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```