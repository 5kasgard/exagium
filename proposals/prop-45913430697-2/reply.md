```hypothesis
Family: label-posmlp (position-aware embedding MLP). The linear baseline sums position-specific embeddings, losing order information critical for next-character prediction. By concatenating embeddings (preserving position) and adding a small ReLU hidden layer, the model captures pairwise interactions between context bytes. We expect >0.34 dev accuracy (beating label-linear's 0.3375) with ~2x the inference cost (still <15k Ir/item). If dev accuracy ≤0.3375 or cost >2x label-linear, the hypothesis fails.
```

```python
"""Position-aware embedding MLP for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Architecture: per-position byte embeddings (size E) concatenated -> hidden layer (size H, ReLU) -> output (65 classes).
Hyperparameters (E, H) chosen on last 20% of train.in (tuning split), then retrained on all data.
Exported weights: magic 0x4D4C5031, d, E, H, C; float32 embeddings[d][256][E], W1[D][H], b1[H], W2[H][C], b2[C] (D=d*E)."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 1e-3
BATCH = 256
EPOCHS_TUNE = 5
EPOCHS_FULL = 25
L2 = 1e-5
# Grid for tuning split (E, H)
GRID = [(4, 32), (8, 64), (16, 128)]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def init_params(d, E, H, C, rng):
    D = d * E
    # embeddings: per position, per byte value, vector of size E
    emb = (rng.normal(0, 1.0 / math.sqrt(E), (d, 256, E))).astype(np.float32)
    # W1: (D, H), He init for ReLU
    W1 = (rng.normal(0, math.sqrt(2.0 / D), (D, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # W2: (H, C), Xavier init
    W2 = (rng.normal(0, math.sqrt(2.0 / (H + C)), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return emb, W1, b1, W2, b2


def forward(emb, W1, b1, W2, b2, X_idx):
    """X_idx: (batch, d) uint8. Returns logits (batch, C)."""
    batch, d = X_idx.shape
    E = emb.shape[2]
    D = d * E
    # concatenate embeddings
    cat = np.empty((batch, D), np.float32)
    for j in range(d):
        cat[:, j*E:(j+1)*E] = emb[j, X_idx[:, j]]
    # hidden
    h = cat @ W1 + b1
    np.maximum(h, 0, out=h)  # ReLU in-place
    # output
    logits = h @ W2 + b2
    return logits, cat, h


def loss_grad(logits, y):
    """Cross-entropy loss and gradient wrt logits."""
    batch = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits, dtype=np.float32)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(batch), y] -= 1.0
    probs /= batch
    loss = -np.log(np.maximum(probs[np.arange(batch), y] + 1.0, 1e-12)).mean()
    return loss, probs.astype(np.float32)


def backward(cat, h, probs, W2, emb, W1, b1, W2_grad, b2_grad, X_idx, y, batch, d, E, H, C, L2):
    # probs is gradient of loss wrt logits (batch, C)
    # W2 grad: h.T @ probs
    W2_grad[:] = h.T @ probs
    b2_grad[:] = probs.sum(axis=0)
    # grad wrt h
    grad_h = probs @ W2.T
    # ReLU mask
    grad_h[h == 0] = 0.0
    # W1 grad: cat.T @ grad_h
    W1_grad = cat.T @ grad_h
    b1_grad = grad_h.sum(axis=0)
    # embedding grads
    emb_grad = np.zeros_like(emb)
    for j in range(d):
        grad_emb_j = grad_h[:, j*E:(j+1)*E].T @ np.eye(batch, dtype=np.float32)[X_idx[:, j]]
        # The above is inefficient; we can accumulate directly:
        # Actually, we can do: for each sample, add grad_h[i, j*E:(j+1)*E] to emb_grad[j, X_idx[i, j]]
        # Let's do a loop over batch (batch is small, 256)
        for i in range(batch):
            b = X_idx[i, j]
            emb_grad[j, b] += grad_h[i, j*E:(j+1)*E]
    # Add L2 regularization gradients
    W2_grad += L2 * W2
    W1_grad += L2 * W1
    emb_grad += L2 * emb
    return emb_grad, W1_grad, b1_grad


def evaluate(emb, W1, b1, W2, b2, X_idx, y):
    logits, _, _ = forward(emb, W1, b1, W2, b2, X_idx)
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def train_split(emb, W1, b1, W2, b2, X_idx, y, rng, epochs, batch_size, L2):
    n = X_idx.shape[0]
    # Adam state
    params = [emb, W1, b1, W2, b2]
    shapes = [p.shape for p in params]
    m = [np.zeros(s, np.float32) for s in shapes]
    v = [np.zeros(s, np.float32) for s in shapes]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s+batch_size]
            batch_idx = X_idx[idx]
            batch_y = y[idx]
            logits, cat, h = forward(emb, W1, b1, W2, b2, batch_idx)
            loss, probs = loss_grad(logits, batch_y)
            # backward
            emb_grad, W1_grad, b1_grad = backward(cat, h, probs, W2, emb, W1, b1, None, None, batch_idx, batch_y,
                                                  len(batch_idx), emb.shape[0], emb.shape[2], W1.shape[1], W2.shape[1], L2)
            W2_grad = h.T @ probs + L2 * W2
            b2_grad = probs.sum(axis=0) + L2 * b2
            grads = [emb_grad, W1_grad, b1_grad, W2_grad, b2_grad]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)
    return emb, W1, b1, W2, b2


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # load data
    X_recs = records(sys.argv[2])
    y_recs = records(sys.argv[3])
    X = np.frombuffer(b"".join(X_recs), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in y_recs], dtype=np.int64)
    n = X.shape[0]
    # split tuning (last 20%)
    tune_size = n // 5
    X_train, X_tune = X[:-tune_size], X[-tune_size:]
    y_train, y_tune = y[:-tune_size], y[-tune_size:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_EH = None

    for E, H in GRID:
        emb, W1, b1, W2, b2 = init_params(d, E, H, C, rng)
        emb, W1, b1, W2, b2 = train_split(emb, W1, b1, W2, b2, X_train, y_train, rng, EPOCHS_TUNE, BATCH, L2)
        acc = evaluate(emb, W1, b1, W2, b2, X_tune, y_tune)
        print(f"Tune E={E} H={H} acc={acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc = acc
            best_params = (emb.copy(), W1.copy(), b1.copy(), W2.copy(), b2.copy())
            best_EH = (E, H)

    E, H = best_EH
    print(f"Best: E={E} H={H} acc={best_acc:.4f}", file=sys.stderr)
    # retrain on full data
    emb, W1, b1, W2, b2 = init_params(d, E, H, C, rng)
    emb, W1, b1, W2, b2 = train_split(emb, W1, b1, W2, b2, X, y, rng, EPOCHS_FULL, BATCH, L2)

    # save weights
    D = d * E
    magic = 0x4D4C5031  # "MLP1"
    header = struct.pack("<5i", magic, d, E, H, C)
    # order: embeddings (d,256,E), W1 (D,H), b1 (H), W2 (H,C), b2 (C)
    out = header
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Position-aware embedding MLP (label-posmlp) for next-byte prediction.
 * ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic 0x4D4C5031, int32 d, E, H, C; float32 embeddings[d][256][E], W1[D][H], b1[H], W2[H][C], b2[C] (D=d*E). */
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
    if (nw < 20 || hd[0] != 0x4D4C5031) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    int D = d * E;
    long expected = 20L + 4L * ((long)d * 256 * E + (long)D * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld vs %ld\n", nw, expected); return 3; }
    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *cat = malloc(sizeof(float) * D);
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(cat + j * E, e, E * sizeof(float));
        }
        // hidden layer: h = ReLU(cat @ W1 + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < D; v++) sum += cat[v] * W1[(long)v * H + u];
            h[u] = sum > 0.0f ? sum : 0.0f;
        }
        // output layer: logits = h @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[(long)u * C + c];
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