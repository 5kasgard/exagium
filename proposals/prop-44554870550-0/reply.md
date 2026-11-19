```hypothesis
Family: label-emb (embedding-based linear model). Learns a dense embedding vector for each of the 14×256 feature values, sums the 14 embeddings per example, and applies a linear classifier. Captures feature interactions via the shared embedding space while keeping inference fast (sum + small matmul). Should beat label-linear on dev accuracy because ordinal bins and categorical indices both benefit from learned similarities. If dev accuracy ≤ label-linear (0.8524) or inference cost exceeds label-linear, the hypothesis fails.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 256
MAX_EPOCHS = 50
SAMPLES = 600000
D_CANDIDATES = [8, 16, 32, 64]  # embedding dimensions to try on tuning split

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, D, C, rng, epochs, batch, lr, l2):
    n, d = X.shape
    # embeddings: (d, 256, D)
    scale = math.sqrt(6.0 / (256 + D))
    E = rng.uniform(-scale, scale, (d, 256, D)).astype(np.float32)
    W = rng.uniform(-scale, scale, (D, C)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [E, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], y[idx]
            # forward
            h = np.zeros((len(xb), D), np.float32)
            for j in range(d):
                h += E[j, xb[:, j]]
            logits = h @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_logits = probs / len(yb)  # (batch, C)
            # backward
            grad_W = h.T @ grad_logits + l2 * W
            grad_b = grad_logits.sum(axis=0)
            grad_h = grad_logits @ W.T  # (batch, D)
            grad_E = np.zeros_like(E)
            for j in range(d):
                # accumulate gradients for each feature value
                np.add.at(grad_E[j], xb[:, j], grad_h)
            grad_E += l2 * E
            grads = [grad_E, grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params

def eval_model(X, y, params):
    E, W, b = params
    n, d = X.shape
    h = np.zeros((n, W.shape[0]), np.float32)
    for j in range(d):
        h += E[j, X[:, j]]
    logits = h @ W + b
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n_all = X_all.shape[0]
    # tuning split: last 20% in file order
    split = int(0.8 * n_all)
    X_train, X_val = X_all[:split], X_all[split:]
    y_train, y_val = y_all[:split], y_all[split:]

    best_D = D_CANDIDATES[0]
    best_acc = -1.0
    for D in D_CANDIDATES:
        rng = np.random.default_rng(SEED)
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))
        params = train_model(X_train, y_train, D, C, rng, epochs, BATCH, LR, L2)
        acc = eval_model(X_val, y_val, params)
        if acc > best_acc:
            best_acc = acc
            best_D = D

    # final training on all data
    rng = np.random.default_rng(SEED)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_all))
    E, W, b = train_model(X_all, y_all, best_D, C, rng, epochs, BATCH, LR, L2)

    # write weights.bin
    # header: magic, d, D, C (all int32 little-endian)
    magic = 0x454D424C  # 'L B M E' in little-endian? Actually 0x4C424D45 is 'EMBL' but we'll use this.
    hdr = struct.pack("<4i", magic, d, best_D, C)
    # embeddings: (d, 256, D) -> flatten as d*256*D float32 little-endian
    emb_bytes = E.astype("<f4").tobytes()
    W_bytes = W.astype("<f4").tobytes()
    b_bytes = b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(hdr + emb_bytes + W_bytes + b_bytes)

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
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 14 || C != 2 || D <= 0 || D > 1024) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *E = (const float *)(hd + 4);
    const float *W = E + (long)d * 256 * D;
    const float *b = W + (long)D * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * D);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(h, 0, sizeof(float) * D);
        for (int j = 0; j < d; j++) {
            const float *emb = E + ((long)j * 256 + xi[j]) * D;
            for (int u = 0; u < D; u++) h[u] += emb[u];
        }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int u = 0; u < D; u++) {
            float hu = h[u];
            if (hu != 0.0f) {
                const float *wu = W + (long)u * C;
                for (int c = 0; c < C; c++) scores[c] += hu * wu[c];
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```