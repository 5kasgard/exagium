```hypothesis
Family: label-emb-linear. Position-specific embeddings (dim E) for each of 16 context bytes, concatenated and fed to a linear output layer (65 classes). Embeddings share statistical strength across characters, improving over one-hot linear while adding only 16*65*E parameters. Fast inference: 16*E lookups + one matrix-vector multiply. If dev accuracy ≤ label-linear (0.3375) or CPU cost exceeds label-linear significantly, the approach fails.
```

```python
"""label-emb-linear: position-specific embeddings + linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy. Tunes emb_dim on last 20% of train, then retrains on all.
Exports: magic 0x454D424C, int32 d, E, C; float32 embeddings[d][65][E], W[d*E][C], b[C]."""
import json, math, struct, sys

import numpy as np

# Tuning grid
EMB_DIMS = [4, 8, 16]
LRS = [1e-3, 3e-3]
L2S = [1e-4, 1e-5]
SEED = 1
BATCH = 64
MAX_EPOCHS = 100
SAMPLES = 600000


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(logits, y):
    return (logits.argmax(axis=1) == y).mean()


def train_eval(X, y, Xv, yv, d, C, E, lr, l2, rng):
    # embeddings: (d, 65, E)
    emb = (rng.normal(0, 0.01, (d, 65, E))).astype(np.float32)
    W = (rng.normal(0, 0.01, (d * E, C))).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    n = X.shape[0]
    steps_per_epoch = max(1, n // BATCH)
    total_steps = min(MAX_EPOCHS * steps_per_epoch, math.ceil(SAMPLES / BATCH))
    t = 0
    for step in range(total_steps):
        idx = rng.integers(0, n, BATCH)
        xb, yb = X[idx], y[idx]
        # forward
        x_emb = emb[np.arange(d)[:, None], xb.T].transpose(1, 0, 2).reshape(BATCH, d * E)
        logits = x_emb @ W + b
        # softmax cross-entropy gradient
        max_logits = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(BATCH), yb] -= 1.0
        grad_logits = probs / BATCH
        # gradients
        grad_W = x_emb.T @ grad_logits + l2 * W
        grad_b = grad_logits.sum(axis=0)
        grad_x_emb = grad_logits @ W.T
        grad_emb = np.zeros_like(emb)
        for j in range(d):
            for i in range(BATCH):
                v = xb[i, j]
                grad_emb[j, v] += grad_x_emb[i, j * E:(j + 1) * E]
        grad_emb += l2 * emb
        grads = [grad_emb, grad_W, grad_b]
        # Adam update
        t += 1
        for p, gr, (m, v) in zip(params, grads, state):
            m *= 0.9; m += 0.1 * gr
            v *= 0.999; v += 0.001 * gr * gr
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    # validation accuracy
    xv_emb = emb[np.arange(d)[:, None], Xv.T].transpose(1, 0, 2).reshape(Xv.shape[0], d * E)
    logits_v = xv_emb @ W + b
    return accuracy(logits_v, yv), (emb, W, b)


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc, best_cfg, best_params = -1, None, None
    for E in EMB_DIMS:
        for lr in LRS:
            for l2 in L2S:
                np.random.seed(SEED)
                rng = np.random.default_rng(SEED)
                acc, params = train_eval(X_tr, y_tr, X_val, y_val, d, C, E, lr, l2, rng)
                if acc > best_acc:
                    best_acc, best_cfg, best_params = acc, (E, lr, l2), params
    E, lr, l2 = best_cfg
    # retrain on full data
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    _, final_params = train_eval(X, y, X[:1], y[:1], d, C, E, lr, l2, rng)  # dummy val
    emb, W, b = final_params

    # write weights
    magic = 0x454D424C  # 'L','B','M','E' little-endian
    out = struct.pack("<4i", magic, d, E, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emb-linear inference: ./infer weights.bin in.rec N
 * Input: d bytes per record (d=16). Output: 1 byte class index.
 * Weights format: magic 0x454D424C, int32 d, E, C; float32 emb[d][65][E], W[d*E][C], b[C]. */
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
    int d = hd[1], E = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad header\n"); return 3; }
    long emb_sz = (long)d * 65 * E;
    long W_sz = (long)d * E * C;
    long b_sz = C;
    long expected = 16 + 4L * (emb_sz + W_sz + b_sz);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 4);
    const float *W = emb + emb_sz;
    const float *b = W + W_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_emb = malloc(sizeof(float) * d * E);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 65 + xi[j]) * E;
            memcpy(x_emb + j * E, e, E * sizeof(float));
        }
        // linear layer
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int w = 0; w < d * E; w++) sum += x_emb[w] * W[(long)w * C + c];
            logits[c] = sum;
        }
        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```