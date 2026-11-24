```hypothesis
Family: label-embed (factorized linear with shared byte embeddings).
Why: Sharing embeddings across 16 context positions reduces parameters (256*E + 16*E*C vs 16*256*C) and regularizes the model, which should improve generalization on the small tinyshakespeare training set.
Failure: If validation accuracy does not exceed label-linear (0.3375), the factorization constraint is too restrictive for this context length.
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


def records(path):
    data = open(path, "rb").read()
    out = []
    i = 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_model(X, y, C, E, rng, n_samples=SAMPLES, max_epochs=MAX_EPOCHS):
    n, d = X.shape
    emb = (rng.normal(0, 1.0 / math.sqrt(E), (256, E))).astype(np.float32)
    W = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, C))).astype(np.float32)
    bias = np.zeros(C, np.float32)
    params = [emb, W, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(max_epochs, math.ceil(n_samples / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x_batch = X[idx]
            y_batch = y[idx]
            emb_batch = emb[x_batch]
            flat = emb_batch.reshape(len(idx), -1)
            logits = flat @ W + bias
            max_logits = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            probs[np.arange(len(idx)), y_batch] -= 1.0
            grad_logits = probs / len(idx)
            grad_W = flat.T @ grad_logits + L2 * W
            grad_bias = np.sum(grad_logits, axis=0)
            grad_flat = grad_logits @ W.T
            grad_emb_batch = grad_flat.reshape(len(idx), d, E)
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb, x_batch[:, j], grad_emb_batch[:, j, :])
            grad_emb += L2 * emb
            grads = [grad_emb, grad_W, grad_bias]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9
                m += 0.1 * gr
                v *= 0.999
                v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9**t)
                v_hat = v / (1 - 0.999**t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return emb, W, bias


def evaluate(emb, W, bias, X, y):
    n, d = X.shape
    emb_batch = emb[X]
    flat = emb_batch.reshape(n, -1)
    logits = flat @ W + bias
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == y)


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_all = len(X_all)
    split = int(0.8 * n_all)
    X_train, X_val = X_all[:split], X_all[split:]
    y_train, y_val = y_all[:split], y_all[split:]
    rng = np.random.default_rng(SEED)
    best_E = 16
    best_val = -1.0
    for E in [8, 16, 32, 64]:
        emb, W, bias = train_model(X_train, y_train, C, E, rng)
        val_acc = evaluate(emb, W, bias, X_val, y_val)
        print(f"E={E}, val_acc={val_acc:.4f}", file=sys.stderr)
        if val_acc > best_val:
            best_val = val_acc
            best_E = E
    print(f"Best E={best_E}, val_acc={best_val:.4f}", file=sys.stderr)
    rng = np.random.default_rng(SEED)
    emb, W, bias = train_model(X_all, y_all, C, best_E, rng)
    T = np.zeros((d, 256, C), dtype=np.float32)
    for j in range(d):
        W_j = W[j * best_E:(j + 1) * best_E, :]
        T[j] = emb @ W_j
    magic = 0x454D4245
    out = struct.pack("<4i", magic, d, 0, C)
    out += T.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (H != 0 || C < 1 || C > 256) { fprintf(stderr, "bad weights: H must be 0\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *T = (const float *)(hd + 4);
    const float *bias = T + (long)d * 256 * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) acc[c] = bias[c];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) acc[c] += t[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```