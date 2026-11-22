```hypothesis
Propose "label-emb": dense per-position byte embeddings summed and fed to a linear classifier.  
Captures character similarities (e.g., 'a'≈'b') and reduces parameters vs. one-hot linear, improving generalization on 24k samples.  
If dev accuracy ≤ 0.3375 (label-linear) or inference cost ≥ label-linear, the hypothesis fails.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
MAX_EPOCHS_TUNE = 5
MAX_EPOCHS_FULL = 20
EMB_DIMS = [16, 32, 64]
MAGIC = 0x454D424C  # "LBME"

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def evaluate(X, y, emb, W, b):
    n = X.shape[0]
    correct = 0
    for s in range(0, n, BATCH):
        xb = X[s:s+BATCH]
        yb = y[s:s+BATCH]
        h = np.zeros((len(xb), emb.shape[2]), np.float32)
        for pos in range(X.shape[1]):
            h += emb[pos, xb[:, pos]]
        logits = h @ W + b
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n

def train_loop(X, y, emb, W, b, epochs, rng):
    n, d, D, C = X.shape[0], X.shape[1], emb.shape[2], W.shape[1]
    params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yy = X[idx], y[idx]
            bs = len(yy)
            # forward
            h = np.zeros((bs, D), np.float32)
            for pos in range(d):
                h += emb[pos, xb[:, pos]]
            logits = h @ W + b
            # softmax gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yy] -= 1.0
            g_logits = (probs / bs).astype(np.float32)
            # gradients
            grad_W = (h.T @ g_logits).astype(np.float32) + L2 * W
            grad_b = g_logits.sum(axis=0).astype(np.float32)
            grad_h = (g_logits @ W.T).astype(np.float32)
            grad_emb = np.zeros_like(emb)
            for pos in range(d):
                # accumulate grad_h for each byte value
                bytes_pos = xb[:, pos]
                for byte_val in np.unique(bytes_pos):
                    mask = (bytes_pos == byte_val)
                    grad_emb[pos, byte_val] = grad_h[mask].sum(axis=0) / bs
                grad_emb[pos] += L2 * emb[pos]
            grads = [grad_emb, grad_W, grad_b]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    n = X.shape[0]
    val_size = n // 5
    train_X, train_y = X[:-val_size], y[:-val_size]
    val_X, val_y = X[-val_size:], y[-val_size:]

    best_acc = -1.0
    best_D = EMB_DIMS[0]
    best_emb = best_W = best_b = None

    for D in EMB_DIMS:
        rng = np.random.default_rng(SEED)
        emb = rng.normal(0, 0.01, (d, 256, D)).astype(np.float32)
        W = rng.normal(0, 0.01, (D, C)).astype(np.float32)
        b = np.zeros(C, np.float32)
        train_loop(train_X, train_y, emb, W, b, MAX_EPOCHS_TUNE, rng)
        acc = evaluate(val_X, val_y, emb, W, b)
        if acc > best_acc:
            best_acc = acc
            best_D = D
            best_emb, best_W, best_b = emb.copy(), W.copy(), b.copy()

    # final training on all data
    rng = np.random.default_rng(SEED)
    emb = rng.normal(0, 0.01, (d, 256, best_D)).astype(np.float32)
    W = rng.normal(0, 0.01, (best_D, C)).astype(np.float32)
    b = np.zeros(C, np.float32)
    train_loop(X, y, emb, W, b, MAX_EPOCHS_FULL, rng)

    # write weights.bin
    out = struct.pack("<4i", MAGIC, d, best_D, C)
    out += best_emb.astype("<f4").tobytes()
    out += best_W.astype("<f4").tobytes()
    out += best_b.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * D;
    const float *b = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(h, 0, sizeof(float) * D);
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            for (int u = 0; u < D; u++) h[u] += e[u];
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
    free(h); free(scores);
    return 0;
}
```