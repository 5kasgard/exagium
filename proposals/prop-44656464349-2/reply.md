```hypothesis
We propose label-embedding-mlp: dense per-position byte embeddings (size E) followed by a ReLU hidden layer (size H) and a linear output. This captures nonlinear interactions while keeping inference fast via precomputed lookup tables (d x 256 x H). Hyperparameters (E,H) are chosen on the tuning split (last 20% of train). If dev accuracy ≤ 0.34 or inference cost > 15k Ir/item, the model fails.
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
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, d, C, E, H, rng, epochs):
    embeddings = [rng.uniform(-1, 1, (256, E)).astype(np.float32) * math.sqrt(6.0 / 256) for _ in range(d)]
    W1 = rng.uniform(-1, 1, (d * E, H)).astype(np.float32) * math.sqrt(6.0 / (d * E))
    b1 = np.zeros(H, np.float32)
    W2 = rng.uniform(-1, 1, (H, C)).astype(np.float32) * math.sqrt(6.0 / H)
    b2 = np.zeros(C, np.float32)

    params = [embeddings, W1, b1, W2, b2]
    state = []
    for p in params:
        if isinstance(p, list):
            state.append([(np.zeros_like(pe), np.zeros_like(pe)) for pe in p])
        else:
            state.append((np.zeros_like(p), np.zeros_like(p)))

    t = 0
    n = X.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            emb = np.concatenate([embeddings[j][xb[:, j]] for j in range(d)], axis=1)
            hidden = emb @ W1 + b1
            hidden_relu = np.maximum(hidden, 0)
            logits = hidden_relu @ W2 + b2
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(len(yb)), yb] -= 1.0
            grad_logits /= len(yb)

            grad_W2 = hidden_relu.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(0)
            grad_hidden_relu = grad_logits @ W2.T
            grad_hidden = grad_hidden_relu * (hidden > 0)
            grad_W1 = emb.T @ grad_hidden + L2 * W1
            grad_b1 = grad_hidden.sum(0)
            grad_emb = grad_hidden @ W1.T
            grad_emb = grad_emb.reshape(len(yb), d, E)
            grad_embeddings = [np.zeros((256, E), np.float32) for _ in range(d)]
            for j in range(d):
                np.add.at(grad_embeddings[j], xb[:, j], grad_emb[:, j])
            for j in range(d):
                grad_embeddings[j] += L2 * embeddings[j]

            grads = [grad_embeddings, grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                if isinstance(p, list):
                    for pe, gre, (m, v) in zip(p, gr, state[i]):
                        m *= 0.9; m += 0.1 * gre
                        v *= 0.999; v += 0.001 * gre * gre
                        m_hat = m / (1 - 0.9 ** t)
                        v_hat = v / (1 - 0.999 ** t)
                        pe -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
                else:
                    m, v = state[i]
                    m *= 0.9; m += 0.1 * gr
                    v *= 0.999; v += 0.001 * gr * gr
                    m_hat = m / (1 - 0.9 ** t)
                    v_hat = v / (1 - 0.999 ** t)
                    p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return embeddings, W1, b1, W2, b2

def evaluate(X, y, d, C, embeddings, W1, b1, W2, b2):
    emb = np.concatenate([embeddings[j][X[:, j]] for j in range(d)], axis=1)
    hidden = np.maximum(emb @ W1 + b1, 0)
    logits = hidden @ W2 + b2
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    val_size = n // 5
    train_X, val_X = X[:-val_size], X[-val_size:]
    train_y, val_y = y[:-val_size], y[-val_size:]

    grid = [(16, 64), (16, 128), (32, 64)]
    best_acc = -1.0
    best_params = None
    best_weights = None
    rng = np.random.default_rng(SEED)

    for E, H in grid:
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(train_X)))
        embeddings, W1, b1, W2, b2 = train_model(train_X, train_y, d, C, E, H, rng, epochs)
        acc = evaluate(val_X, val_y, d, C, embeddings, W1, b1, W2, b2)
        if acc > best_acc:
            best_acc = acc
            best_params = (E, H)
            T = np.stack([embeddings[j] @ W1[j * E:(j + 1) * E] for j in range(d)])
            best_weights = (T, b1, W2, b2)

    print(f"Best hyperparams: E={best_params[0]}, H={best_params[1]}, val_acc={best_acc:.4f}", file=sys.stderr)

    E, H = best_params
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    embeddings, W1, b1, W2, b2 = train_model(X, y, d, C, E, H, rng, epochs)
    T = np.stack([embeddings[j] @ W1[j * E:(j + 1) * E] for j in range(d)])

    magic = 0x4C424D45
    out = struct.pack("<4i", magic, d, H, C)
    out += T.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x4C424D45) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * H + H + (long)H * C + C)) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * H);
    float *s = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int h = 0; h < H; h++) acc[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) acc[h] += t[h];
        }
        for (int h = 0; h < H; h++) if (acc[h] < 0.0f) acc[h] = 0.0f;
        for (int c = 0; c < C; c++) s[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float ah = acc[h];
            for (int c = 0; c < C; c++) s[c] += ah * W2[(long)h * C + c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (s[c] > s[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```