```hypothesis
Family: label-embmlp (embedding MLP with precomputed first-layer tables).
Why: The task is character-level language modeling with 16-byte context. A linear model on one-hot features (label-linear) captures only independent per-position effects. An embedding-based MLP learns distributed character representations and their interactions through a hidden layer, capturing n-gram statistics and longer-range dependencies. By precomputing the embedding→hidden transformation into per-position/byte tables, inference remains fast (sum 16 vectors, ReLU, then small dense layer). Tuning embedding dim E and hidden size H on the tuning split should yield higher accuracy than label-linear at comparable cost.
Failure: If dev accuracy ≤ 0.3375 (label-linear) or instructions per item exceed label-linear's 7360 by >20% without accuracy gain.
```

```python
"""label-embmlp: Embedding MLP for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Architecture: per-position byte embeddings (dim E) -> hidden layer (size H, ReLU) -> output (C classes).
First layer folded into tables T1[d][256][H] for fast inference.
Hyperparameters (E, H, lr, epochs) chosen on last 20% of train.in (file order), then retrained on all.
Weight format: magic 0x4D4C5031, int32 d, E, H, C; float32 T1[d*256*H], b1[H], W2[H*C], b2[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyperparameter grid for tuning split
GRID = [
    {"E": 8, "H": 32, "lr": 1e-3, "epochs": 30, "batch": 64},
    {"E": 16, "H": 32, "lr": 1e-3, "epochs": 30, "batch": 64},
    {"E": 8, "H": 64, "lr": 1e-3, "epochs": 30, "batch": 64},
    {"E": 16, "H": 64, "lr": 1e-3, "epochs": 30, "batch": 64},
    {"E": 8, "H": 32, "lr": 3e-3, "epochs": 30, "batch": 64},
    {"E": 16, "H": 32, "lr": 3e-3, "epochs": 30, "batch": 64},
]
L2 = 1e-4
MAGIC = 0x4D4C5031  # 'MLP1'


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_model(X, y, C, cfg, rng, val_idx=None):
    n, d = X.shape
    E, H, lr, epochs, batch = cfg["E"], cfg["H"], cfg["lr"], cfg["epochs"], cfg["batch"]
    # Embeddings: [d, 256, E]
    emb = (rng.normal(0, 1/math.sqrt(E), (d, 256, E))).astype(np.float32)
    # W1: [d*E, H]
    W1 = (rng.normal(0, math.sqrt(2.0/(d*E)), (d*E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # W2: [H, C]
    W2 = (rng.normal(0, math.sqrt(2.0/H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [emb, W1, b1, W2, b2]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(idx):
        # idx: batch indices
        x = X[idx]  # [B, d]
        B = len(idx)
        # Embedding lookup: [B, d, E]
        e = emb[np.arange(d), x.T].transpose(1, 0, 2)  # [B, d, E]
        flat = e.reshape(B, d*E)  # [B, d*E]
        h = flat @ W1 + b1  # [B, H]
        h = np.maximum(h, 0)  # ReLU
        logits = h @ W2 + b2  # [B, C]
        return logits, h, flat

    def loss_grad(logits, yy):
        B = logits.shape[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(B), yy] -= 1.0
        probs /= B
        return probs.astype(np.float32)

    def backward(idx, yy, probs, h, flat):
        B = len(idx)
        # grad W2, b2
        gW2 = h.T @ probs + L2 * W2
        gb2 = probs.sum(0)
        # grad hidden
        gh = probs @ W2.T
        gh[h <= 0] = 0  # ReLU grad
        # grad W1, b1
        gW1 = flat.T @ gh + L2 * W1
        gb1 = gh.sum(0)
        # grad embeddings
        gflat = gh @ W1.T  # [B, d*E]
        gemb = np.zeros_like(emb)
        gflat = gflat.reshape(B, d, E)
        x = X[idx]
        for j in range(d):
            np.add.at(gemb[j], x[:, j], gflat[:, j])
        return [gemb, gW1, gb1, gW2, gb2]

    def adam_step(grads):
        nonlocal t
        t += 1
        for i, (p, gr) in enumerate(zip(params, grads)):
            m, v = state[i]
            m *= 0.9; m += 0.1 * gr
            v *= 0.999; v += 0.001 * gr * gr
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Training loop
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            logits, h, flat = forward(idx)
            probs = loss_grad(logits, y[idx])
            grads = backward(idx, y[idx], probs, h, flat)
            adam_step(grads)

    # Validation accuracy if needed
    if val_idx is not None:
        logits, _, _ = forward(val_idx)
        preds = logits.argmax(1)
        acc = (preds == y[val_idx]).mean()
        return acc, params
    return None, params


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    val_size = n // 5
    train_idx = np.arange(n - val_size)
    val_idx = np.arange(n - val_size, n)

    rng = np.random.default_rng(SEED)
    best_acc, best_cfg, best_params = -1, None, None
    for cfg in GRID:
        # Use a fresh RNG for each config to keep deterministic across configs? We'll just use the same rng but training is stochastic.
        # To be deterministic, we can create a new rng per config seeded from SEED and config hash.
        cfg_seed = SEED + hash((cfg["E"], cfg["H"], cfg["lr"])) % 2**32
        cfg_rng = np.random.default_rng(cfg_seed)
        acc, params = train_model(X[train_idx], y[train_idx], C, cfg, cfg_rng, val_idx)
        if acc > best_acc:
            best_acc, best_cfg, best_params = acc, cfg, params
    # Retrain on full data with best config
    full_rng = np.random.default_rng(SEED + 12345)
    _, final_params = train_model(X, y, C, best_cfg, full_rng, val_idx=None)
    emb, W1, b1, W2, b2 = final_params

    # Build T1 = emb @ W1 per position: [d, 256, H]
    E, H = best_cfg["E"], best_cfg["H"]
    T1 = np.zeros((d, 256, H), np.float32)
    for j in range(d):
        T1[j] = emb[j] @ W1[j*E:(j+1)*E]

    # Write weights.bin
    header = struct.pack("<5i", MAGIC, d, E, H, C)
    out = header
    out += T1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weight format: magic 0x4D4C5031, int32 d, E, H, C;
 *   float32 T1[d*256*H], b1[H], W2[H*C], b2[C] (all little-endian). */
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
    if (nw < 20 || hd[0] != 0x4D4C5031) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *T1 = (const float *)(hd + 5);
    const float *b1 = T1 + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // hidden = b1
        memcpy(hidden, b1, H * sizeof(float));
        // add T1 contributions
        for (int j = 0; j < d; j++) {
            const float *t = T1 + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) hidden[h] += t[h];
        }
        // ReLU
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;
        // scores = hidden @ W2 + b2
        memcpy(scores, b2, C * sizeof(float));
        for (int h = 0; h < H; h++) {
            float hv = hidden[h];
            if (hv == 0.0f) continue;
            const float *w = W2 + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += hv * w[c];
        }
        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```