```hypothesis
Propose label-factorized: a factorized linear model with per-position byte embeddings (dimension D) followed by a linear classifier. This reduces parameters from 16*256*65 to 16*256*D + 65*16*D, which should generalize better on this small dataset while keeping inference fast (O(16*D*65) operations). Hyperparameters (D, lr, L2) are chosen on the tuning split (last 20% of train). If dev accuracy does not exceed label-linear's 0.3375, the family is not effective.
```

```python
"""label-factorized: per-position byte embeddings + linear classifier.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid searched on tuning split
D_CHOICES = [4, 8, 16]
LR_CHOICES = [1e-3, 3e-3]
L2_CHOICES = [1e-4, 1e-3]
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
SEED = 1


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def evaluate(X, y, emb, W, b, d, D, C):
    """Compute accuracy on given data."""
    n = X.shape[0]
    correct = 0
    for s in range(0, n, 1024):
        xb = X[s:s+1024]
        yb = y[s:s+1024]
        # embed
        z = np.zeros((xb.shape[0], d*D), dtype=np.float32)
        for j in range(d):
            z[:, j*D:(j+1)*D] = emb[j, xb[:, j], :]
        logits = z @ W.T + b
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n


def train_model(X_tr, y_tr, X_val, y_val, d, D, C, lr, l2, rng):
    """Train one configuration, return (emb, W, b, val_acc)."""
    n, _ = X_tr.shape
    # init embeddings: d x 256 x D
    emb = (rng.uniform(-1, 1, (d, 256, D)) * math.sqrt(6.0 / 256)).astype(np.float32)
    W = (rng.uniform(-1, 1, (C, d*D)) * math.sqrt(6.0 / (d*D))).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    params = [emb, W, b]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            # forward
            z = np.zeros((xb.shape[0], d*D), dtype=np.float32)
            for j in range(d):
                z[:, j*D:(j+1)*D] = emb[j, xb[:, j], :]
            logits = z @ W.T + b
            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            probs /= len(yb)
            # gradients
            gW = probs.T @ z + l2 * W
            gb = probs.sum(axis=0)
            gemb = np.zeros_like(emb)
            for j in range(d):
                # gradient wrt embeddings at position j
                gz = probs @ W[:, j*D:(j+1)*D]
                np.add.at(gemb[j], xb[:, j], gz)
            gemb += l2 * emb
            grads = [gemb, gW, gb]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9**t)
                v_hat = v / (1 - 0.999**t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    val_acc = evaluate(X_val, y_val, emb, W, b, d, D, C)
    return emb, W, b, val_acc


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # load all training data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # tuning split: last 20%
    split = int(0.8 * n)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_cfg = None
    best_params = None

    for D in D_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                # use a sub-seed for each config to keep deterministic
                cfg_rng = np.random.default_rng(SEED + hash((D, lr, l2)) % 2**32)
                emb, W, b, val_acc = train_model(X_tr, y_tr, X_val, y_val, d, D, C, lr, l2, cfg_rng)
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_cfg = (D, lr, l2)
                    best_params = (emb, W, b)
    print(f"Best config: D={best_cfg[0]}, lr={best_cfg[1]}, l2={best_cfg[2]}, val_acc={best_acc:.4f}", file=sys.stderr)

    # retrain on full data with best config
    D, lr, l2 = best_cfg
    full_rng = np.random.default_rng(SEED + 12345)
    emb, W, b, _ = train_model(X, y, X_val, y_val, d, D, C, lr, l2, full_rng)

    # save weights
    # magic 0x46414354 ('FACT' little-endian), d, D, C
    magic = 0x46414354
    out = struct.pack("<4i", magic, d, D, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-factorized infer: ./infer weights.bin in.rec N
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
    uint8_t *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x46414354) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C != 65 || D <= 0 || D > 64) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16
        + (long)d * 256 * D * 4
        + (long)C * d * D * 4
        + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(wb + 16);
    const float *W = emb + (long)d * 256 * D;
    const float *b = W + (long)C * d * D;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d * D);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        /* embed: concatenate embeddings for each position */
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            memcpy(z + j * D, e, D * sizeof(float));
        }
        /* linear layer */
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            const float *wc = W + (long)c * d * D;
            for (int w = 0; w < d * D; w++) sum += z[w] * wc[w];
            logits[c] = sum;
        }
        /* argmax */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```