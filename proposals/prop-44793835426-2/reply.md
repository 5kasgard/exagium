```hypothesis
Family: label-embedding-linear. Learns a shared 256×E embedding table for byte values, concatenates the 16 embeddings (16E features), then a linear layer to 65 classes. Embeddings capture character similarities, acting as regularization on the small dataset (24k train). The reduced feature dimension (16E vs 4096 one-hot) speeds up inference. Hyperparameter E is chosen on the last 20% of train.in (tuning split). Failure: dev accuracy ≤ label-linear (0.3375) or inference cost not lower.
```

```python
"""label-embedding-linear: shared byte embeddings + linear classifier.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# hyperparameters to tune on the validation split (last 20% of train.in)
E_CANDIDATES = [8, 16, 32, 64]
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_eval(X_tr, y_tr, X_val, y_val, E, C, rng):
    d = X_tr.shape[1]
    # embeddings: 256 x E
    emb = (rng.normal(0, 0.01, (256, E))).astype(np.float32)
    # linear: (d*E) x C
    W = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, C))).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    n_tr = X_tr.shape[0]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            # forward
            emb_x = emb[xb]                 # (B, d, E)
            flat = emb_x.reshape(len(yb), -1)  # (B, d*E)
            logits = flat @ W + b
            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_logits = (probs / len(yb)).astype(np.float32)
            # gradients
            grad_W = flat.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(0)
            grad_flat = grad_logits @ W.T
            grad_emb = np.zeros_like(emb)
            np.add.at(grad_emb, xb.reshape(-1), grad_flat.reshape(-1, E))
            grad_emb += L2 * emb
            grads = [grad_emb, grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    # validation accuracy
    emb_x = emb[X_val]
    flat = emb_x.reshape(X_val.shape[0], -1)
    logits = flat @ W + b
    preds = logits.argmax(axis=1)
    acc = (preds == y_val).mean()
    return acc, params


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]
    # split: last 20% for validation (file order)
    val_size = n // 5
    X_tr, y_tr = X[:-val_size], y[:-val_size]
    X_val, y_val = X[-val_size:], y[-val_size:]

    rng = np.random.default_rng(SEED)
    best_E, best_acc, best_params = None, -1, None
    for E in E_CANDIDATES:
        acc, params = train_eval(X_tr, y_tr, X_val, y_val, E, C, rng)
        if acc > best_acc:
            best_acc, best_E, best_params = acc, E, params
    # retrain on full data with best_E
    rng = np.random.default_rng(SEED)
    _, best_params = train_eval(X, y, X[:1], y[:1], best_E, C, rng)  # dummy val, we just need the trained params

    emb, W, b = best_params
    # write weights.bin
    # magic 0x454D424C ('L' 'B' 'M' 'E' little-endian? we'll use 0x454D424C)
    magic = 0x454D424C  # 'EMBL' in ASCII? Actually 'L','B','M','E' -> 0x454D424C? Let's use a distinct magic.
    # We'll use 0x4C424D45 ('L','B','M','E') as little-endian: 0x454D424C. But label-linear used 0x54424C31.
    # We'll use 0x454D424C (E M B L). Doesn't matter as long as infer.c matches.
    out = struct.pack("<4i", 0x454D424C, d, best_E, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embedding-linear infer.c
 * ./infer weights.bin in.rec N
 * weights format: magic(4), d(4), E(4), C(4), emb[256*E](f4), W[d*E*C](f4), b[C](f4) */
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
    int d = hd[1], E = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (256L * E + (long)d * E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 4);
    const float *W = emb + 256L * E;
    const float *b = W + (long)d * E * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *logits = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + (long)xi[j] * E;
            memcpy(concat + j * E, e, E * sizeof(float));
        }
        // linear layer
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int f = 0; f < d * E; f++) sum += concat[f] * W[(long)f * C + c];
            logits[c] = sum;
        }
        // argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(concat); free(logits);
    return 0;
}
```