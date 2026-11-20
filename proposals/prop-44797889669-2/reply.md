```hypothesis
The low-rank linear model (label-lowrank) factorizes the per-position weight matrices through a shared byte embedding of dimension D. This reduces parameters (from ~266k to ~41k for D=32) and acts as structured regularization by forcing similar characters to have similar weights. It should beat label-linear (0.3375) on dev accuracy by capturing character similarities, while being faster due to smaller weight matrices. If dev accuracy does not exceed 0.3375, the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np

# hyperparameters
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
EPOCHS_SEARCH = 15
EPOCHS_FINAL = 50
CANDIDATE_D = [8, 16, 32, 64]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, C, D, epochs, rng):
    n, d = X.shape
    # shared embedding for all 256 byte values
    E = rng.uniform(-0.1, 0.1, (256, D)).astype(np.float32)
    # per-position weight matrices: [d, D, C]
    W = rng.uniform(-0.1, 0.1, (d, D, C)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [E, W, b]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)
            # forward
            emb = E[xb]          # (B, d, D)
            logits = np.einsum('bjd,jdc->bc', emb, W) + b  # (B, C)
            # softmax
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            # gradient
            dlogits = probs
            dlogits[np.arange(B), yb] -= 1.0
            dlogits /= B
            # gradients
            db = dlogits.sum(axis=0)
            dW = np.einsum('bjd,bc->jdc', emb, dlogits)  # (d, D, C)
            dE = np.zeros_like(E)
            for j in range(d):
                np.add.at(dE, xb[:, j], dlogits @ W[j].T)  # (B, D)
            # L2
            dW += L2 * W
            dE += L2 * E
            grads = [dE, dW, db]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return E, W, b

def eval_model(X, y, C, E, W, b):
    n = X.shape[0]
    correct = 0
    for s in range(0, n, BATCH):
        xb = X[s:s+BATCH]
        yb = y[s:s+BATCH]
        emb = E[xb]
        logits = np.einsum('bjd,jdc->bc', emb, W) + b
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]
    # split last 20% for tuning
    n_train = int(0.8 * n)
    X_tr, X_val = X[:n_train], X[n_train:]
    y_tr, y_val = y[:n_train], y[n_train:]
    rng = np.random.default_rng(SEED)
    best_D = CANDIDATE_D[0]
    best_acc = -1.0
    for D in CANDIDATE_D:
        rng_d = np.random.default_rng(SEED + D)
        E, W, b = train_model(X_tr, y_tr, C, D, EPOCHS_SEARCH, rng_d)
        acc = eval_model(X_val, y_val, C, E, W, b)
        if acc > best_acc:
            best_acc = acc
            best_D = D
    # final training on all data
    rng_final = np.random.default_rng(SEED + 1000 + best_D)
    E, W, b = train_model(X, y, C, best_D, EPOCHS_FINAL, rng_final)
    # write weights
    # magic: 0x4C4F5752 ("LOWR")
    magic = 0x4C4F5752
    out = struct.pack("<4i", magic, d, best_D, C)
    out += E.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x4C4F5752) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * (256L * D + 16L * D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *E = (const float *)(hd + 4);
    const float *W = E + 256L * D;
    const float *b = W + 16L * D * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(scores, b, C * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *e = E + (long)xi[j] * D;
            const float *w = W + (long)j * D * C;
            for (int u = 0; u < D; u++) {
                float eu = e[u];
                if (eu != 0.0f) {
                    const float *wu = w + (long)u * C;
                    for (int c = 0; c < C; c++) scores[c] += eu * wu[c];
                }
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```