```hypothesis
Family: embedding-linear (position-specific character embeddings + linear classifier).
Why: Character-level sequences exhibit strong similarity structure; embeddings capture this with far fewer parameters than one-hot linear (16*256*D + 16*D*65 vs 16*256*65), reducing overfitting and compute. Position-specific embeddings respect the fixed-context order.
Failure: Dev accuracy ≤ 0.3375 (label-linear) or inference instructions/item not lower than ~7000.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters (tuned on last 20% of train.in)
SEED = 1
LR = 0.01
L2 = 1e-5
BATCH = 128
SAMPLES = 600000
MAX_EPOCHS = 100
D_CANDIDATES = [4, 8, 16, 32]  # embedding dimensions to try


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_eval(X_tr, y_tr, X_val, y_val, D, C, rng, epochs):
    n_tr, d = X_tr.shape
    # position-specific embeddings: (d, 256, D)
    emb = (rng.uniform(-1, 1, (d, 256, D)) * math.sqrt(6.0 / (256 * D))).astype(np.float32)
    # linear weights: (d*D, C)
    W = (rng.uniform(-1, 1, (d * D, C)) * math.sqrt(6.0 / (d * D))).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(X):
        # X: (N, d) uint8
        N = X.shape[0]
        # embed: (N, d, D)
        e = emb[X, np.arange(d)]  # advanced indexing: (N, d, D)
        e = e.reshape(N, d * D)   # (N, d*D)
        scores = e @ W + b
        return scores, e

    def sg(scores, yy):
        # scores: (N, C), yy: (N,)
        scores = scores - scores.max(axis=1, keepdims=True)
        P = np.exp(scores)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(epochs):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            scores, e = forward(Xb)
            g = sg(scores, yb)  # (B, C)
            # gradients
            grad_W = e.T @ g + L2 * W
            grad_b = g.sum(0)
            # emb gradient: (d, 256, D)
            grad_emb = np.zeros_like(emb)
            # backprop through embedding lookup
            g_e = g @ W.T  # (B, d*D)
            g_e = g_e.reshape(len(idx), d, D)  # (B, d, D)
            # accumulate per position and byte value
            for j in range(d):
                bytes_j = Xb[:, j]  # (B,)
                np.add.at(grad_emb[j], bytes_j, g_e[:, j, :])
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # validation accuracy
    scores_val, _ = forward(X_val)
    pred = scores_val.argmax(axis=1)
    acc = (pred == y_val).mean()
    return acc, params


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]

    # tuning split: last 20%
    split = int(0.8 * n)
    X_tr_full, y_tr_full = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_D, best_acc, best_params = None, -1, None
    tune_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_tr_full)))

    for D in D_CANDIDATES:
        rng_tune = np.random.default_rng(SEED + D)  # deterministic per D
        acc, params = train_eval(X_tr_full, y_tr_full, X_val, y_val, D, C, rng_tune, tune_epochs)
        if acc > best_acc:
            best_acc, best_D, best_params = acc, D, params

    # retrain on all data with best D
    final_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    rng_final = np.random.default_rng(SEED + best_D + 1000)
    _, final_params = train_eval(X, y, X_val, y_val, best_D, C, rng_final, final_epochs)
    emb, W, b = final_params

    # write weights.bin
    # magic: 0x44424D45 ("EMBD" little-endian), D, d, C
    header = struct.pack("<4I", 0x44424D45, best_D, d, C)
    out = header
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* embedding-linear inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte class index. */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k>>8 & 255, k>>16 & 255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wbuf = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hd = (uint32_t *)wbuf;
    if (hd[0] != 0x44424D45) { fprintf(stderr, "bad magic\n"); return 3; }
    uint32_t D = hd[1], d = hd[2], C = hd[3];
    if (d != 16 || C != 65) { fprintf(stderr, "unexpected dims\n"); return 3; }
    size_t emb_sz = (size_t)d * 256 * D;
    size_t W_sz = (size_t)d * D * C;
    size_t b_sz = C;
    long expected = 16 + 4L * (emb_sz + W_sz + b_sz);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(wbuf + 16);
    const float *W = emb + emb_sz;
    const float *b = W + W_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * d * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != d) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, d); return 3; }

        // accumulate embeddings
        memset(acc, 0, sizeof(float) * d * D);
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((size_t)j * 256 + xi[j]) * D;
            for (int kk = 0; kk < (int)D; kk++) acc[j * D + kk] += e[kk];
        }

        // linear layer
        for (int c = 0; c < (int)C; c++) scores[c] = b[c];
        for (int f = 0; f < (int)(d * D); f++) {
            float a = acc[f];
            if (a != 0.0f) {
                const float *w = W + (size_t)f * C;
                for (int c = 0; c < (int)C; c++) scores[c] += a * w[c];
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < (int)C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```