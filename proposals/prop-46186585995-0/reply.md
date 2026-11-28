```hypothesis
We propose a new family "label-emb-linear": each of the 14 features (both ordinal and categorical) is mapped to a learned embedding of size E (e.g., 8), all embeddings are concatenated, and a linear layer predicts the class. This captures non-linear feature effects via dense embeddings and allows the linear layer to weigh each feature's embedding dimensions independently, unlike label-linear which either one-hot encodes (high-dim sparse) or sums embeddings (losing feature identity). On this task with mixed ordinal/categorical features, embeddings should better represent categorical similarities and ordinal non-linearities. If dev accuracy does not exceed label-linear (0.8524) or inference cost exceeds label-mlp16 (2325 Ir/item), the hypothesis fails.
```

```python
"""label-emb-linear: per-feature embeddings concatenated + linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameter search space (tuned on last 20% of train.in)
CONFIGS = [
    {"E": 4, "lr": 0.03, "epochs": 30, "batch": 128},
    {"E": 8, "lr": 0.03, "epochs": 30, "batch": 128},
    {"E": 16, "lr": 0.01, "epochs": 30, "batch": 128},
]
SEED = 1
L2 = 1e-5
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(X, y, emb, W, b):
    n, d = X.shape
    E = emb.shape[2]
    C = W.shape[1]
    correct = 0
    for i in range(0, n, 1024):
        xb = X[i:i+1024]
        yb = y[i:i+1024]
        B = xb.shape[0]
        concat = np.empty((B, d * E), np.float32)
        for j in range(d):
            concat[:, j*E:(j+1)*E] = emb[j, xb[:, j]]
        scores = concat @ W + b
        pred = scores.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n


def train_config(X, y, config, seed):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    E, lr, epochs, batch = config["E"], config["lr"], config["epochs"], config["batch"]
    C = 2

    # Initialize parameters
    emb = rng.normal(0, 1/math.sqrt(E), (d, 256, E)).astype(np.float32)
    W = rng.normal(0, 1/math.sqrt(d*E), (d*E, C)).astype(np.float32)
    b = np.zeros(C, np.float32)

    # Adam state
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    t = 0

    idx = np.arange(n)
    for _ in range(epochs):
        rng.shuffle(idx)
        for s in range(0, n, batch):
            sel = idx[s:s+batch]
            xb = X[sel]
            yb = y[sel]
            B = len(sel)

            # Forward
            concat = np.empty((B, d*E), np.float32)
            for j in range(d):
                concat[:, j*E:(j+1)*E] = emb[j, xb[:, j]]
            scores = concat @ W + b
            # Softmax cross-entropy
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            probs /= B

            # Gradients
            grad_concat = probs @ W.T  # (B, d*E)
            grad_W = concat.T @ probs + L2 * W
            grad_b = probs.sum(axis=0)

            # Backprop to embeddings
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                slice_ = grad_concat[:, j*E:(j+1)*E]
                np.add.at(grad_emb[j], xb[:, j], slice_)
            grad_emb += L2 * emb

            # Adam update
            t += 1
            for param, grad, m, v in [
                (emb, grad_emb, m_emb, v_emb),
                (W, grad_W, m_W, v_W),
                (b, grad_b, m_b, v_b)
            ]:
                m *= BETA1; m += (1 - BETA1) * grad
                v *= BETA2; v += (1 - BETA2) * grad * grad
                m_hat = m / (1 - BETA1 ** t)
                v_hat = v / (1 - BETA2 ** t)
                param -= lr * m_hat / (np.sqrt(v_hat) + EPS)

    return emb, W, b


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)

    n = X.shape[0]
    tune_size = n // 5
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)

    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    best_acc = -1.0
    best_params = None
    best_config = None

    for i, cfg in enumerate(CONFIGS):
        print(f"Trying config {i}: {cfg}", file=sys.stderr)
        emb, W, b = train_config(X_train, y_train, cfg, SEED + i)
        acc = accuracy(X_tune, y_tune, emb, W, b)
        print(f"  tune accuracy: {acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc = acc
            best_params = (emb, W, b)
            best_config = cfg

    print(f"Best config: {best_config} (acc={best_acc:.4f})", file=sys.stderr)
    # Retrain on full data
    emb, W, b = train_config(X, y, best_config, SEED + 999)

    # Save weights: magic, d, E, C, then embeddings[d][256][E], W[d*E][C], b[C]
    E = best_config["E"]
    magic = 0x454D424C  # 'L''B''M''E' in little-endian
    header = struct.pack("<4i", magic, d, E, C)
    emb_bytes = emb.astype("<f4").tobytes()
    W_bytes = W.astype("<f4").tobytes()
    b_bytes = b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + emb_bytes + W_bytes + b_bytes)


if __name__ == "__main__":
    main()
```

```c
/* label-emb-linear inference: ./infer weights.bin in.rec N
 * Input record: d bytes (one per feature). Output record: 1 byte class index. */
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
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)d * E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + (long)d * E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * E;
            float *dst = concat + j * E;
            memcpy(dst, src, E * sizeof(float));
        }

        // Linear layer
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int k = 0; k < d * E; k++) {
            float v = concat[k];
            if (v != 0.0f) {
                const float *w = W + (long)k * C;
                for (int c = 0; c < C; c++) scores[c] += v * w[c];
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```