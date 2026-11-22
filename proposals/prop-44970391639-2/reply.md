```hypothesis
Family: label-hash2 (linear + hashed bigrams). Adds pairwise interactions between adjacent context bytes via a fixed hash table (512 buckets per position), capturing local n-gram statistics that a pure linear model misses. Trained with Adam on 80% of data, early-stopped on the last 20% tuning split, then refit on all data. If dev accuracy does not exceed label-linear's 0.3375, the hashed bigrams are not helping or hash collisions hurt.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 50
B = 512          # bigram hash buckets (power of two)
PATIENCE = 5     # early stopping patience


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]          # 16

    n = len(X_all)
    split = int(0.8 * n)                # last 20% in file order is tuning
    train_idx = np.arange(split)
    tune_idx = np.arange(split, n)

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_tune,  y_tune  = X_all[tune_idx],  y_all[tune_idx]

    rng = np.random.default_rng(SEED)

    def init_params():
        U = np.zeros((d, 256, C), np.float32)       # unigram [pos, byte, class]
        V = np.zeros((d - 1, B, C), np.float32)     # bigram  [pos, bucket, class]
        b = np.zeros(C, np.float32)                 # bias
        return U, V, b

    def predict(X, U, V, b):
        n = len(X)
        logits = np.tile(b, (n, 1))
        for j in range(d):
            logits += U[j, X[:, j]]
        for j in range(d - 1):
            bidx = (X[:, j].astype(np.uint32) * 257 + X[:, j + 1]) & (B - 1)
            logits += V[j, bidx]
        return logits.argmax(axis=1)

    def evaluate(X, y, U, V, b):
        return (predict(X, U, V, b) == y).mean()

    def train_epoch(X, y, U, V, b, states, lr, t):
        n = len(X)
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            bs = len(Xb)

            # forward
            logits = np.tile(b, (bs, 1))
            for j in range(d):
                logits += U[j, Xb[:, j]]
            for j in range(d - 1):
                bidx = (Xb[:, j].astype(np.uint32) * 257 + Xb[:, j + 1]) & (B - 1)
                logits += V[j, bidx]

            # softmax gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs

            # gradients
            g_b = probs.sum(axis=0)
            g_U = np.zeros_like(U)
            g_V = np.zeros_like(V)
            for j in range(d):
                np.add.at(g_U[j], Xb[:, j], probs)
            for j in range(d - 1):
                bidx = (Xb[:, j].astype(np.uint32) * 257 + Xb[:, j + 1]) & (B - 1)
                np.add.at(g_V[j], bidx, probs)

            # L2
            g_b += L2 * b
            g_U += L2 * U
            g_V += L2 * V

            # Adam update
            params = [U, V, b]
            grads  = [g_U, g_V, g_b]
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = states[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            t += 1
        return t

    # ---- tuning phase ----
    best_acc = -1.0
    best_epoch = 0
    best_params = None

    U, V, b = init_params()
    states = [(np.zeros_like(U), np.zeros_like(U)),
              (np.zeros_like(V), np.zeros_like(V)),
              (np.zeros_like(b), np.zeros_like(b))]
    t = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        t = train_epoch(X_train, y_train, U, V, b, states, LR, t)
        acc = evaluate(X_tune, y_tune, U, V, b)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            best_params = (U.copy(), V.copy(), b.copy())
        if epoch - best_epoch >= PATIENCE:
            break

    # ---- final training on all data ----
    U, V, b = init_params()
    states = [(np.zeros_like(U), np.zeros_like(U)),
              (np.zeros_like(V), np.zeros_like(V)),
              (np.zeros_like(b), np.zeros_like(b))]
    t = 0
    for _ in range(best_epoch):
        t = train_epoch(X_all, y_all, U, V, b, states, LR, t)

    # ---- save weights ----
    # magic "HAS2" = 0x48415332, d, C, B, then U, V, b (all float32 little-endian)
    magic = 0x48415332
    out = struct.pack("<4i", magic, d, C, B)
    out += U.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x48415332) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], B = hd[3];
    if (d != 16 || C != 65 || B != 512) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * C + (long)(d - 1) * B * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *U = (const float *)(hd + 4);
    const float *V = U + (long)d * 256 * C;
    const float *b = V + (long)(d - 1) * B * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) logits[c] = b[c];

        for (int j = 0; j < d; j++) {
            const float *u = U + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) logits[c] += u[c];
        }
        for (int j = 0; j < d - 1; j++) {
            int bidx = (xi[j] * 257 + xi[j + 1]) & (B - 1);
            const float *v = V + ((long)j * B + bidx) * C;
            for (int c = 0; c < C; c++) logits[c] += v[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```