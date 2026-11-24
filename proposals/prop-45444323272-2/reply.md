```hypothesis
Family: label-embed (per-position byte embeddings summed + linear classifier).
Why: Character-level language modeling benefits from distributed representations; embeddings capture similarity between characters and reduce parameters vs one-hot linear models. Summing embeddings across 16 positions yields a fixed-size context vector, followed by a linear layer to 65 classes. This should exceed label-linear accuracy (0.3375) with similar inference cost (16*D adds + D*C multiply-adds, D=32).
Failure: If dev accuracy ≤ 0.3375 or inference instructions/item exceed label-linear by >2×, the approach does not justify its complexity.
```

```python
"""label-embed: per-position byte embeddings summed, then linear classifier.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning split (last 20% of train.in)
GRID = [
    {"D": 16, "lr": 0.03, "epochs": 10},
    {"D": 32, "lr": 0.03, "epochs": 10},
    {"D": 48, "lr": 0.03, "epochs": 10},
    {"D": 64, "lr": 0.03, "epochs": 10},
    {"D": 32, "lr": 0.01, "epochs": 10},
    {"D": 32, "lr": 0.1, "epochs": 10},
]
SEED = 1
BATCH = 64
L2 = 1e-5
MAGIC = 0x454D4245  # 'EMBE' little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(logits, y):
    return np.mean(np.argmax(logits, axis=1) == y)


def forward(X, E, W_out, b_out):
    # X: (N, d) uint8
    N, d = X.shape
    D = E.shape[2]
    h = np.zeros((N, D), dtype=np.float32)
    for j in range(d):
        h += E[j, X[:, j]]  # (N, D)
    logits = h @ W_out + b_out  # (N, C)
    return logits, h


def train_model(X_tr, y_tr, X_val, y_val, D, lr, epochs, C, d, rng):
    # Initialize parameters
    E = rng.normal(0, 0.01, size=(d, 256, D)).astype(np.float32)
    W_out = rng.normal(0, 0.01, size=(D, C)).astype(np.float32)
    b_out = np.zeros(C, dtype=np.float32)

    # Adam state
    mE, vE = np.zeros_like(E), np.zeros_like(E)
    mW, vW = np.zeros_like(W_out), np.zeros_like(W_out)
    mb, vb = np.zeros_like(b_out), np.zeros_like(b_out)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    N = X_tr.shape[0]
    for _ in range(epochs):
        order = rng.permutation(N)
        for s in range(0, N, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            Nb = len(yb)

            # Forward
            logits, h = forward(Xb, E, W_out, b_out)
            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(Nb), yb] -= 1.0
            probs /= Nb  # (Nb, C)

            # Gradients
            gW = h.T @ probs + L2 * W_out  # (D, C)
            gb = probs.sum(axis=0)         # (C,)
            gh = probs @ W_out.T           # (Nb, D)

            # Gradient for embeddings
            gE = np.zeros_like(E)
            for j in range(d):
                # scatter-add: for each example, add gh[i] to gE[j, Xb[i, j]]
                np.add.at(gE[j], Xb[:, j], gh)

            # Adam updates
            t += 1
            for param, grad, m, v in [
                (E, gE, mE, vE),
                (W_out, gW, mW, vW),
                (b_out, gb, mb, vb),
            ]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # Validation accuracy
    logits_val, _ = forward(X_val, E, W_out, b_out)
    val_acc = accuracy(logits_val, y_val)
    return E, W_out, b_out, val_acc


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    N = X.shape[0]

    # Tuning split: last 20% in file order
    split = int(0.8 * N)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)

    best_acc = -1.0
    best_params = None
    best_cfg = None

    for cfg in GRID:
        D = cfg["D"]
        lr = cfg["lr"]
        epochs = cfg["epochs"]
        # Train on 80%
        E, W_out, b_out, val_acc = train_model(X_tr, y_tr, X_val, y_val, D, lr, epochs, C, d, rng)
        if val_acc > best_acc:
            best_acc = val_acc
            best_params = (E, W_out, b_out)
            best_cfg = cfg
        print(f"cfg {cfg} val_acc={val_acc:.4f}", file=sys.stderr)

    print(f"Best cfg {best_cfg} val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best config
    D, lr, epochs = best_cfg["D"], best_cfg["lr"], best_cfg["epochs"]
    rng_full = np.random.default_rng(SEED)  # fresh seed for final training
    E, W_out, b_out, _ = train_model(X, y, X_val[:1], y_val[:1], D, lr, epochs, C, d, rng_full)

    # Write weights.bin
    # Header: magic, d, D, C (int32)
    header = struct.pack("<4i", MAGIC, d, D, C)
    # Embeddings: d * 256 * D float32 (row-major: position, byte, dim)
    emb_bytes = E.astype("<f4").tobytes()
    # Output weights: D * C float32
    w_bytes = W_out.astype("<f4").tobytes()
    # Output bias: C float32
    b_bytes = b_out.astype("<f4").tobytes()

    with open(sys.argv[4], "wb") as f:
        f.write(header + emb_bytes + w_bytes + b_bytes)


if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes (d=16). Output: one byte class index (0..64). */
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
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C != 65 || D <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }

    long expected = 16 + 4L * ((long)d * 256 * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *E = (const float *)(hd + 4);                 // [d][256][D]
    const float *W_out = E + (long)d * 256 * D;               // [D][C]
    const float *b_out = W_out + (long)D * C;                 // [C]

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // h = sum_j E[j][xi[j]]
        memset(h, 0, sizeof(float) * D);
        for (int j = 0; j < d; j++) {
            const float *ej = E + ((long)j * 256 + xi[j]) * D;
            for (int m = 0; m < D; m++) h[m] += ej[m];
        }

        // scores = h @ W_out + b_out
        for (int c = 0; c < C; c++) scores[c] = b_out[c];
        for (int m = 0; m < D; m++) {
            float hm = h[m];
            if (hm == 0.0f) continue;
            const float *wm = W_out + (long)m * C;
            for (int c = 0; c < C; c++) scores[c] += hm * wm[c];
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```