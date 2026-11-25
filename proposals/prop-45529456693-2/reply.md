```hypothesis
We propose a factorized linear model (label-embed) where each of the 16 context positions has a learned embedding of dimension E, summed and passed to a shared linear classifier. This reduces parameters vs label-linear (which uses 65-dim one-hot per position) while preserving positional information, and acts as regularization. We expect it to match or beat label-linear's 0.3375 dev accuracy at lower inference cost. If dev accuracy is below 0.33 or inference cost exceeds label-linear, the hypothesis fails.
```

```python
"""label-embed: factorized linear model with per-position embeddings.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Chooses embedding dim E on the last 20% tuning split, then trains on all data.
Exports: magic 0x454D4245, d=16, E, C=65; float32 embeddings[16][256][E], W[E][65], b[65]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
CANDIDATE_E = [32, 64, 128]  # tried on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_eval(X_tr, y_tr, X_val, y_val, E, C, rng, epochs):
    """Train on tr, return validation accuracy."""
    n, d = X_tr.shape
    # embeddings: d x 256 x E
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    W = (rng.uniform(-1, 1, (E, C)) * math.sqrt(6.0 / E)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(X):
        # X: (n, d) uint8
        # sum embeddings: (n, E)
        s = np.zeros((len(X), E), np.float32)
        for j in range(d):
            s += emb[j, X[:, j]]
        logits = s @ W + b
        return logits

    def loss_grad(logits, y):
        # logits: (n, C), y: (n,)
        max_logits = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(y)), y] -= 1.0
        probs /= len(y)
        return probs.astype(np.float32)

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            # forward
            s_emb = np.zeros((len(Xb), E), np.float32)
            for j in range(d):
                s_emb += emb[j, Xb[:, j]]
            logits = s_emb @ W + b
            g = loss_grad(logits, yb)  # (b, C)
            # gradients
            grad_W = s_emb.T @ g + L2 * W
            grad_b = g.sum(0)
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                # grad_emb[j, Xb[:, j]] += g @ W.T  (b, E) -> need to scatter
                gw = g @ W.T  # (b, E)
                np.add.at(grad_emb[j], Xb[:, j], gw)
            grad_emb += L2 * emb
            grads = [grad_emb, grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # validation accuracy
    logits_val = forward(X_val)
    pred = logits_val.argmax(axis=1)
    return (pred == y_val).mean(), params


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    n = len(X)
    # split: last 20% for tuning
    n_tr = int(0.8 * n)
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_val, y_val = X[n_tr:], y[n_tr:]
    rng = np.random.default_rng(SEED)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))

    best_E, best_acc, best_params = None, -1.0, None
    for E in CANDIDATE_E:
        acc, params = train_eval(X_tr, y_tr, X_val, y_val, E, C, np.random.default_rng(SEED), epochs)
        if acc > best_acc:
            best_acc, best_E, best_params = acc, E, params
    # retrain on full data with best_E
    rng_full = np.random.default_rng(SEED)
    _, final_params = train_eval(X, y, X[:1], y[:1], best_E, C, rng_full, epochs)  # dummy val
    emb, W, b = final_params

    # write weights.bin
    magic = 0x454D4245  # 'EMBE' little-endian
    out = struct.pack("<4i", magic, d, best_E, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input: 16 bytes per record. Output: one byte class index per record. */
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
    if (nw < 16 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 16 || C != 65 || E <= 0 || E > 256) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + (long)E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum = malloc(sizeof(float) * E);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(sum, 0, E * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int u = 0; u < E; u++) sum[u] += e[u];
        }
        for (int c = 0; c < C; c++) {
            float acc = b[c];
            for (int u = 0; u < E; u++) acc += sum[u] * W[(long)u * C + c];
            logits[c] = acc;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```