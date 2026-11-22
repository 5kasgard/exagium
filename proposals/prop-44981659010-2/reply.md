```hypothesis
The label-emblinear family uses position-specific embeddings (dimension D) followed by a linear output layer. This factorizes the per-position weight matrices (16×256×65 → 16×256×D + 16×D×65), reducing overfitting on 24k examples while preserving position information. It should exceed label-linear's 0.3375 dev accuracy at lower CPU cost (fewer multiply-adds). If dev accuracy ≤ 0.3375 or inference cost ≥ label-linear, the hypothesis fails.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
LR, BETA1, BETA2, EPS = 1e-2, 0.9, 0.999, 1e-8
BATCH = 64
TUNE_EPOCHS, FINAL_EPOCHS = 10, 50
D_CANDIDATES = [4, 8, 16, 32]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_step(X, y, emb, W, b, m_emb, v_emb, m_W, v_W, m_b, v_b, t, rng, l2=1e-5):
    n = X.shape[0]
    order = rng.permutation(n)
    for s in range(0, n, BATCH):
        idx = order[s:s + BATCH]
        xb, yb = X[idx], y[idx]
        bsz = xb.shape[0]
        # forward
        hid = np.zeros((bsz, emb.shape[0] * emb.shape[2]), np.float32)
        for j in range(emb.shape[0]):
            hid[:, j*emb.shape[2]:(j+1)*emb.shape[2]] = emb[j, xb[:, j]]
        logits = hid @ W + b
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(bsz), yb] -= 1.0
        grad_logits = probs / bsz
        # backward
        grad_hid = grad_logits @ W.T
        grad_W = hid.T @ grad_logits + l2 * W
        grad_b = grad_logits.sum(axis=0)
        grad_emb = np.zeros_like(emb)
        for j in range(emb.shape[0]):
            d = emb.shape[2]
            gj = grad_hid[:, j*d:(j+1)*d]
            np.add.at(grad_emb[j], xb[:, j], gj)
        grad_emb += l2 * emb
        # Adam update
        t += 1
        for param, grad, m, v in [(emb, grad_emb, m_emb, v_emb),
                                  (W, grad_W, m_W, v_W),
                                  (b, grad_b, m_b, v_b)]:
            m *= BETA1; m += (1 - BETA1) * grad
            v *= BETA2; v += (1 - BETA2) * grad * grad
            m_hat = m / (1 - BETA1 ** t)
            v_hat = v / (1 - BETA2 ** t)
            param -= LR * m_hat / (np.sqrt(v_hat) + EPS)
    return t

def evaluate(X, y, emb, W, b):
    hid = np.zeros((X.shape[0], emb.shape[0] * emb.shape[2]), np.float32)
    for j in range(emb.shape[0]):
        hid[:, j*emb.shape[2]:(j+1)*emb.shape[2]] = emb[j, X[:, j]]
    logits = hid @ W + b
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    n = X.shape[0]
    val_size = n // 5
    train_idx = np.arange(n - val_size)
    val_idx = np.arange(n - val_size, n)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    best_D, best_acc = None, -1.0
    for D in D_CANDIDATES:
        rng = np.random.default_rng(SEED)
        emb = rng.normal(0, 0.01, (d, 256, D)).astype(np.float32)
        W = rng.normal(0, 0.01, (d * D, C)).astype(np.float32)
        b = np.zeros(C, np.float32)
        m_emb = np.zeros_like(emb); v_emb = np.zeros_like(emb)
        m_W = np.zeros_like(W); v_W = np.zeros_like(W)
        m_b = np.zeros_like(b); v_b = np.zeros_like(b)
        t = 0
        for _ in range(TUNE_EPOCHS):
            t = train_step(X_train, y_train, emb, W, b,
                           m_emb, v_emb, m_W, v_W, m_b, v_b, t, rng)
        acc = evaluate(X_val, y_val, emb, W, b)
        if acc > best_acc:
            best_acc, best_D = acc, D

    # final training on all data
    rng = np.random.default_rng(SEED)
    emb = rng.normal(0, 0.01, (d, 256, best_D)).astype(np.float32)
    W = rng.normal(0, 0.01, (d * best_D, C)).astype(np.float32)
    b = np.zeros(C, np.float32)
    m_emb = np.zeros_like(emb); v_emb = np.zeros_like(emb)
    m_W = np.zeros_like(W); v_W = np.zeros_like(W)
    m_b = np.zeros_like(b); v_b = np.zeros_like(b)
    t = 0
    for _ in range(FINAL_EPOCHS):
        t = train_step(X, y, emb, W, b,
                       m_emb, v_emb, m_W, v_W, m_b, v_b, t, rng)

    # write weights: magic, d, D, C; emb[d][256][D]; W[d*D][C]; b[C]
    magic = 0x454D424C  # "EMBL"
    out = struct.pack("<4i", magic, d, best_D, C)
    out += emb.astype("<f4").tobytes()
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
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * D + (long)d * D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * D;
    const float *b = W + (long)d * D * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hid = malloc(sizeof(float) * d * D);
    float *sc = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            float *h = hid + j * D;
            for (int u = 0; u < D; u++) h[u] = e[u];
        }
        for (int c = 0; c < C; c++) sc[c] = b[c];
        for (int u = 0; u < d * D; u++) {
            float hu = hid[u];
            if (hu == 0.0f) continue;
            const float *wu = W + (long)u * C;
            for (int c = 0; c < C; c++) sc[c] += hu * wu[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```