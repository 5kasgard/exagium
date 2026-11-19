```hypothesis
Family: label-embed-mlp – per-position byte embeddings summed, then an optional ReLU hidden layer and linear output.  
Why: One-hot linear (label-linear) ignores interactions between positions; a dense embedding compresses the 256‑ary input, allowing a wider hidden layer for the same parameter budget and capturing nonlinear context.  
Failure sign: dev accuracy ≤ 0.3375 or instructions/item > 7360 (label-linear’s cost).
```

```python
"""label-embed-mlp: per-position embeddings + optional ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only.
Grid-searches (D, H) on last 20% of train.in, then retrains on all data.
Exports: magic 0x454D424C, d, D, H, C; embeddings[d][256][D]; W1[D][W]; b1[W]; if H: W2[H][C], b2[C] (W=H if H>0 else C)."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
GRID = [(16, 0), (16, 32), (16, 64), (32, 0), (32, 32), (32, 64), (64, 0), (64, 32), (64, 64)]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(logits, y):
    return (logits.argmax(1) == y).mean()


def train_step(Xb, yb, emb, W1, b1, W2, b2, H, C, L2):
    n = Xb.shape[0]
    # forward
    z = np.zeros((n, emb.shape[2]), np.float32)
    for j in range(emb.shape[0]):
        z += emb[j, Xb[:, j]]
    if H:
        a = z @ W1 + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
    else:
        logits = z @ W1 + b1
    # loss + grad
    logits -= logits.max(1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(1, keepdims=True)
    probs[np.arange(n), yb] -= 1.0
    probs /= n
    # gradients
    if H:
        gW2 = h.T @ probs + L2 * W2
        gb2 = probs.sum(0)
        gh = probs @ W2.T
        gh *= (a > 0)
        gW1 = z.T @ gh + L2 * W1
        gb1 = gh.sum(0)
        gz = gh @ W1.T
    else:
        gW1 = z.T @ probs + L2 * W1
        gb1 = probs.sum(0)
        gz = probs @ W1.T
    # embedding gradients
    gemb = np.zeros_like(emb)
    for j in range(emb.shape[0]):
        np.add.at(gemb[j], Xb[:, j], gz)
    return gemb, gW1, gb1, (gW2, gb2) if H else (None, None)


def fit(X, y, D, H, C, rng, epochs, batch, lr, l2):
    d = X.shape[1]
    # init
    emb = rng.normal(0, 0.01, (d, 256, D)).astype(np.float32)
    W = H if H else C
    W1 = rng.uniform(-1, 1, (D, W)).astype(np.float32) * math.sqrt(6.0 / D)
    b1 = np.zeros(W, np.float32)
    if H:
        W2 = rng.uniform(-1, 1, (H, C)).astype(np.float32) * math.sqrt(6.0 / H)
        b2 = np.zeros(C, np.float32)
    else:
        W2 = b2 = None
    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params if p is not None]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(len(X))
        for s in range(0, len(X), batch):
            idx = order[s:s + batch]
            grads = train_step(X[idx], y[idx], emb, W1, b1, W2, b2, H, C, l2)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                if gr is None:
                    continue
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return emb, W1, b1, W2, b2


def predict(X, emb, W1, b1, W2, b2, H):
    n = X.shape[0]
    z = np.zeros((n, emb.shape[2]), np.float32)
    for j in range(emb.shape[0]):
        z += emb[j, X[:, j]]
    if H:
        a = z @ W1 + b1
        h = np.maximum(a, 0)
        return h @ W2 + b2
    else:
        return z @ W1 + b1


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    n = len(X)
    split = int(0.8 * n)
    Xtr, ytr = X[:split], y[:split]
    Xval, yval = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc, best_cfg, best_weights = -1, None, None
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(Xtr)))
    for D, H in GRID:
        r = np.random.default_rng(SEED)  # fresh seed per config
        emb, W1, b1, W2, b2 = fit(Xtr, ytr, D, H, C, r, epochs, BATCH, LR, L2)
        logits = predict(Xval, emb, W1, b1, W2, b2, H)
        acc = accuracy(logits, yval)
        if acc > best_acc:
            best_acc, best_cfg = acc, (D, H)
            best_weights = (emb, W1, b1, W2, b2)
            print(f"  tuning: D={D} H={H} acc={acc:.4f}", file=sys.stderr)

    D, H = best_cfg
    print(f"selected: D={D} H={H} val_acc={best_acc:.4f}", file=sys.stderr)
    # retrain on all data
    rng = np.random.default_rng(SEED)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    emb, W1, b1, W2, b2 = fit(X, y, D, H, C, rng, epochs, BATCH, LR, L2)

    # pack weights
    W = H if H else C
    magic = 0x454D424C
    header = struct.pack("<5i", magic, d, D, H, C)
    emb_bytes = emb.astype("<f4").tobytes()
    W1_bytes = W1.astype("<f4").tobytes()
    b1_bytes = b1.astype("<f4").tobytes()
    out = header + emb_bytes + W1_bytes + b1_bytes
    if H:
        out += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C < 1 || C > 256 || D < 1) { fprintf(stderr, "bad header\n"); return 3; }
    size_t emb_sz = (size_t)d * 256 * D;
    size_t W1_sz = (size_t)D * (H ? H : C);
    size_t b1_sz = H ? H : C;
    size_t W2_sz = H ? (size_t)H * C : 0;
    size_t b2_sz = H ? C : 0;
    long expected = 20 + 4L * (emb_sz + W1_sz + b1_sz + W2_sz + b2_sz);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + emb_sz;
    const float *b1 = W1 + W1_sz;
    const float *W2 = b1 + b1_sz;
    const float *b2 = W2 + W2_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * D);
    float *h = H ? malloc(sizeof(float) * H) : NULL;
    float *s = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(z, 0, sizeof(float) * D);
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((size_t)j * 256 + xi[j]) * D;
            for (int v = 0; v < D; v++) z[v] += e[v];
        }
        if (H) {
            for (int u = 0; u < H; u++) {
                float sum = b1[u];
                for (int v = 0; v < D; v++) sum += z[v] * W1[(size_t)v * H + u];
                h[u] = sum > 0.0f ? sum : 0.0f;
            }
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int u = 0; u < H; u++) sum += h[u] * W2[(size_t)u * C + c];
                s[c] = sum;
            }
        } else {
            for (int c = 0; c < C; c++) {
                float sum = b1[c];
                for (int v = 0; v < D; v++) sum += z[v] * W1[(size_t)v * C + c];
                s[c] = sum;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (s[c] > s[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```