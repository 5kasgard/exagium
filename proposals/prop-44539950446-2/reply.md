```hypothesis
Family: label-embedmlp. Learns per-position byte embeddings (dense vectors) followed by a one-hidden-layer ReLU MLP. Embeddings capture character similarities (e.g., letters cluster) and reduce input dimension vs one-hot, improving generalization on small data. The MLP models non-linear context interactions. If dev accuracy ≤ 0.3375 (label-linear), the embedding+MLP advantage does not materialize.
```

```python
"""label-embedmlp: per-position byte embeddings + one-hidden-layer ReLU MLP.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters chosen on last 20% tuning split, then trained on all data.
Weights format: magic 0x454D4244, d, D, H, C; float32 embed[d][256][D], W1[(d*D)][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 1e-3
L2 = 1e-5
BATCH = 64
MAX_EPOCHS = 100
SAMPLES = 600000
# Grid for tuning split (last 20% of train)
GRID_D = [8, 16, 32]
GRID_H = [32, 64, 128]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_model(d, D, H, C, rng):
    embed = (rng.normal(0, 0.01, (d, 256, D))).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * D)), (d * D, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return embed, W1, b1, W2, b2

def forward(embed, W1, b1, W2, b2, X_idx):
    # X_idx: (N, d) uint8
    N, d = X_idx.shape
    D = embed.shape[2]
    # Embedding lookup: (N, d, D)
    emb = embed[np.arange(d), X_idx.T].transpose(1, 0, 2)  # (N, d, D)
    x = emb.reshape(N, d * D)  # (N, d*D)
    h = x @ W1 + b1
    h = np.maximum(h, 0)
    logits = h @ W2 + b2
    return logits, h, x

def loss_grad(logits, y):
    N = logits.shape[0]
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(N), y] -= 1.0
    return probs / N  # dL/dlogits

def train_model(X, y, d, D, H, C, rng, epochs, batch, lr, l2, verbose=False):
    embed, W1, b1, W2, b2 = build_model(d, D, H, C, rng)
    params = [embed, W1, b1, W2, b2]
    shapes = [p.shape for p in params]
    # Adam state
    m = [np.zeros(s, np.float32) for s in shapes]
    v = [np.zeros(s, np.float32) for s in shapes]
    t = 0
    N = X.shape[0]
    for ep in range(epochs):
        order = rng.permutation(N)
        for s in range(0, N, batch):
            idx = order[s:s + batch]
            Xb, yb = X[idx], y[idx]
            logits, h, x = forward(embed, W1, b1, W2, b2, Xb)
            g = loss_grad(logits, yb)  # (B, C)
            # Backprop
            dW2 = h.T @ g + l2 * W2
            db2 = g.sum(0)
            dh = g @ W2.T
            dh[h <= 0] = 0
            dW1 = x.T @ dh + l2 * W1
            db1 = dh.sum(0)
            dx = dh @ W1.T  # (B, d*D)
            # Embedding gradients
            dembed = np.zeros_like(embed)
            dx_reshaped = dx.reshape(-1, d, D)  # (B, d, D)
            for j in range(d):
                np.add.at(dembed[j], Xb[:, j], dx_reshaped[:, j, :])
            dembed += l2 * embed
            grads = [dembed, dW1, db1, dW2, db2]
            t += 1
            beta1, beta2 = 0.9, 0.999
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        if verbose:
            logits, _, _ = forward(embed, W1, b1, W2, b2, X)
            acc = (logits.argmax(1) == y).mean()
            print(f"epoch {ep+1}: train acc {acc:.4f}")
    return embed, W1, b1, W2, b2

def eval_acc(embed, W1, b1, W2, b2, X, y):
    logits, _, _ = forward(embed, W1, b1, W2, b2, X)
    return (logits.argmax(1) == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    N = X_bytes.shape[0]
    # Tuning split: last 20%
    split = int(0.8 * N)
    X_train, X_tune = X_bytes[:split], X_bytes[split:]
    y_train, y_tune = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_cfg = None
    best_params = None
    print("Tuning on last 20%...", file=sys.stderr)
    for D in GRID_D:
        for H in GRID_H:
            cfg_rng = np.random.default_rng(SEED + D * 1000 + H)
            embed, W1, b1, W2, b2 = train_model(X_train, y_train, d, D, H, C, cfg_rng,
                                                epochs=min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train))),
                                                batch=BATCH, lr=LR, l2=L2, verbose=False)
            acc = eval_acc(embed, W1, b1, W2, b2, X_tune, y_tune)
            print(f"  D={D}, H={H}: tune acc={acc:.4f}", file=sys.stderr)
            if acc > best_acc:
                best_acc = acc
                best_cfg = (D, H)
                best_params = (embed, W1, b1, W2, b2)
    D, H = best_cfg
    print(f"Best: D={D}, H={H} (tune acc={best_acc:.4f}), retraining on all data", file=sys.stderr)
    # Retrain on full data
    rng = np.random.default_rng(SEED)
    embed, W1, b1, W2, b2 = train_model(X_bytes, y, d, D, H, C, rng,
                                        epochs=min(MAX_EPOCHS, math.ceil(SAMPLES / N)),
                                        batch=BATCH, lr=LR, l2=L2, verbose=True)

    # Save weights
    # Magic: 0x454D4244 ("EMBD" in little-endian)
    magic = 0x454D4244
    out = struct.pack("<5i", magic, d, D, H, C)
    out += embed.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)
    print(f"Saved weights: {len(out)} bytes", file=sys.stderr)

if __name__ == "__main__":
    main()
```

```c
/* label-embedmlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index (argmax).
 * Weights format: magic 0x454D4244, d, D, H, C; float32 embed[d][256][D], W1[(d*D)][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d <= 0 || D <= 0 || H <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 20L
        + (long)d * 256 * D
        + (long)d * D * H
        + H
        + (long)H * C
        + C;
    expected *= 4; /* float32 bytes */
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *embed = (const float *)(hd + 5);
    const float *W1 = embed + (long)d * 256 * D;
    const float *b1 = W1 + (long)d * D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);
    float *x_flat = malloc(sizeof(float) * d * D);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* Embedding lookup and concatenation */
        for (int j = 0; j < d; j++) {
            const float *e = embed + ((long)j * 256 + xi[j]) * D;
            memcpy(x_flat + (long)j * D, e, D * sizeof(float));
        }

        /* Hidden layer: ReLU(W1^T x + b1) */
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            const float *w = W1 + h; /* column-major? W1 is (d*D) x H stored row-major? In train.py W1 shape (d*D, H), row-major: W1[i, h] at i*H + h. So column h starts at h, stride H. */
            for (int idx = 0; idx < d * D; idx++) {
                sum += x_flat[idx] * w[idx * H];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        /* Output layer: W2^T hidden + b2 */
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            const float *w = W2 + c; /* W2 shape (H, C) row-major: W2[h, c] at h*C + c. Column c starts at c, stride C. */
            for (int h = 0; h < H; h++) {
                sum += hidden[h] * w[h * C];
            }
            scores[c] = sum;
        }

        /* Argmax (first max wins) */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(hidden); free(scores); free(x_flat);
    if (x) free(x);
    return 0;
}
```