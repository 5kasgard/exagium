```hypothesis
Family: label-embedmlp (per-position byte embeddings + one ReLU hidden layer).
Why: The task predicts the next character from 16 bytes of context. One-hot linear models (label-linear) use 16×256=4096 sparse features, which overparameterize the small training set (24k items). Learned embeddings (e.g., 8–16 dims per position) drastically reduce parameters while capturing positional character statistics; a small ReLU layer then models non-linear interactions. This should improve accuracy over label-linear (0.3375) at similar or lower inference cost.
Falsification: If dev accuracy ≤ 0.3375 or instructions/item exceed label-linear (7359) by >20%, the hypothesis fails.
```

```python
"""label-embedmlp: per-position byte embeddings + one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (tuning split), then retrained on all train.
Weights format (little-endian):
  uint32 magic 0x454D424C ('L','B','M','E')
  uint32 version (1)
  uint32 d (16), emb_dim, hidden_dim, C (65)
  float32 embeddings[d][256][emb_dim]
  float32 W_hidden[emb_dim][hidden_dim], b_hidden[hidden_dim]
  float32 W_out[hidden_dim][C], b_out[C]
"""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameter grid for tuning
EMB_DIMS = [4, 8, 16]
HIDDEN_DIMS = [16, 32, 64]
LRS = [1e-3, 3e-3]
L2S = [1e-4, 1e-5]
SEED = 1
BATCH = 64
MAX_EPOCHS_TUNE = 20
MAX_EPOCHS_FULL = 50
SAMPLES_PER_EPOCH = 600000  # same as label-linear


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def softmax_cross_entropy_grad(logits, y):
    # logits: (N, C), y: (N,) int
    N = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(N), y] -= 1.0
    return probs / N  # (N, C)


def accuracy(logits, y):
    return (logits.argmax(axis=1) == y).mean()


def forward(X, emb, W_hid, b_hid, W_out, b_out):
    # X: (N, d) uint8
    N, d = X.shape
    emb_dim = emb.shape[2]
    # Sum embeddings
    h = np.zeros((N, emb_dim), dtype=np.float32)
    for j in range(d):
        h += emb[j, X[:, j]]  # emb[j] is (256, emb_dim)
    # Hidden layer
    h = h @ W_hid + b_hid
    h = np.maximum(h, 0)  # ReLU
    # Output
    logits = h @ W_out + b_out
    return logits, h


def train_model(X_tr, y_tr, X_val, y_val, emb_dim, hidden_dim, lr, l2, max_epochs, rng):
    n, d = X_tr.shape
    C = 65
    # Initialize embeddings
    emb = rng.normal(0, 0.01, size=(d, 256, emb_dim)).astype(np.float32)
    W_hid = rng.normal(0, math.sqrt(2.0 / emb_dim), size=(emb_dim, hidden_dim)).astype(np.float32)
    b_hid = np.zeros(hidden_dim, dtype=np.float32)
    W_out = rng.normal(0, math.sqrt(2.0 / hidden_dim), size=(hidden_dim, C)).astype(np.float32)
    b_out = np.zeros(C, dtype=np.float32)

    params = [emb, W_hid, b_hid, W_out, b_out]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    best_val_acc = -1.0
    best_params = None

    steps_per_epoch = max(1, n // BATCH)
    total_steps = max_epochs * steps_per_epoch

    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            # Forward
            logits, h = forward(Xb, emb, W_hid, b_hid, W_out, b_out)
            # Loss gradient
            g_logits = softmax_cross_entropy_grad(logits, yb)  # (B, C)
            # Backprop
            g_W_out = h.T @ g_logits + l2 * W_out
            g_b_out = g_logits.sum(axis=0)
            g_h = g_logits @ W_out.T
            g_h[h <= 0] = 0  # ReLU grad
            g_W_hid = (emb_sum_grad(Xb, emb, g_h)).transpose(1, 0) + l2 * W_hid
            # emb_sum_grad returns (B, d, emb_dim) -> sum over batch gives (d, emb_dim)
            # Actually we need gradient wrt emb: for each position j, for each byte value, sum g_h over batch where Xb[:,j]==v
            # We'll compute efficiently with scatter-add.
            g_b_hid = g_h.sum(axis=0)

            # Gradient for embeddings
            g_emb = np.zeros_like(emb)
            for j in range(d):
                # g_h is (B, emb_dim), Xb[:,j] is (B,)
                # For each sample, add g_h[i] to g_emb[j, Xb[i,j]]
                # We can use np.add.at
                np.add.at(g_emb[j], Xb[:, j], g_h)
            g_emb += l2 * emb

            grads = [g_emb, g_W_hid, g_b_hid, g_W_out, g_b_out]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)

        # Validation
        val_logits, _ = forward(X_val, emb, W_hid, b_hid, W_out, b_out)
        val_acc = accuracy(val_logits, y_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = [p.copy() for p in params]

    return best_params, best_val_acc


def emb_sum_grad(Xb, emb, g_h):
    # Helper: compute gradient for W_hid: sum over batch of (emb_sum)^T * g_h
    # emb_sum for each sample is sum_j emb[j, Xb[i,j]] -> (B, emb_dim)
    # We already have that from forward? Let's recompute.
    # Actually we need the sum of embeddings for each sample to compute W_hid grad.
    # We'll compute it here.
    B, d = Xb.shape
    emb_dim = emb.shape[2]
    emb_sum = np.zeros((B, emb_dim), dtype=np.float32)
    for j in range(d):
        emb_sum += emb[j, Xb[:, j]]
    # Gradient for W_hid: emb_sum.T @ g_h  (emb_dim, hidden_dim)
    return emb_sum.T @ g_h


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert d == 16 and C == 65

    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(X_all)
    split = int(0.8 * n)
    X_tr, y_tr = X_all[:split], y_all[:split]
    X_val, y_val = X_all[split:], y_all[split:]

    rng = np.random.default_rng(SEED)

    best_config = None
    best_val_acc = -1.0
    best_params = None

    # Grid search
    for emb_dim in EMB_DIMS:
        for hidden_dim in HIDDEN_DIMS:
            for lr in LRS:
                for l2 in L2S:
                    params, val_acc = train_model(
                        X_tr, y_tr, X_val, y_val, emb_dim, hidden_dim, lr, l2, MAX_EPOCHS_TUNE, rng
                    )
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_config = (emb_dim, hidden_dim, lr, l2)
                        best_params = params
                    print(f"Tune: emb={emb_dim} hid={hidden_dim} lr={lr} l2={l2} -> val_acc={val_acc:.4f}")

    print(f"Best config: {best_config}, val_acc={best_val_acc:.4f}")

    # Retrain on full data with best config
    emb_dim, hidden_dim, lr, l2 = best_config
    params, _ = train_model(
        X_all, y_all, X_val, y_val, emb_dim, hidden_dim, lr, l2, MAX_EPOCHS_FULL, rng
    )
    emb, W_hid, b_hid, W_out, b_out = params

    # Write weights
    magic = 0x454D424C  # 'L','B','M','E'
    version = 1
    header = struct.pack("<4I", magic, version, d, emb_dim, hidden_dim, C)
    # Flatten embeddings: d * 256 * emb_dim floats
    emb_flat = emb.astype("<f4").tobytes()
    W_hid_flat = W_hid.astype("<f4").tobytes()
    b_hid_flat = b_hid.astype("<f4").tobytes()
    W_out_flat = W_out.astype("<f4").tobytes()
    b_out_flat = b_out.astype("<f4").tobytes()
    out = header + emb_flat + W_hid_flat + b_hid_flat + W_out_flat + b_out_flat
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embedmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format (little-endian):
 *   uint32 magic (0x454D424C), version (1)
 *   uint32 d, emb_dim, hidden_dim, C
 *   float32 embeddings[d][256][emb_dim]
 *   float32 W_hidden[emb_dim][hidden_dim], b_hidden[hidden_dim]
 *   float32 W_out[hidden_dim][C], b_out[C]
 */
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 24) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x454D424C || hd[1] != 1) { fprintf(stderr, "bad magic/version\n"); return 3; }
    int d = hd[2], emb_dim = hd[3], hidden_dim = hd[4], C = hd[5];
    if (d != 16 || C != 65 || emb_dim <= 0 || hidden_dim <= 0) { fprintf(stderr, "bad dims\n"); return 3; }

    // Expected size: header(24) + d*256*emb_dim*4 + emb_dim*hidden_dim*4 + hidden_dim*4 + hidden_dim*C*4 + C*4
    long expected = 24L
        + (long)d * 256 * emb_dim * 4
        + (long)emb_dim * hidden_dim * 4
        + (long)hidden_dim * 4
        + (long)hidden_dim * C * 4
        + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    float *emb = (float *)(wb + 24);
    float *W_hid = emb + (long)d * 256 * emb_dim;
    float *b_hid = W_hid + (long)emb_dim * hidden_dim;
    float *W_out = b_hid + hidden_dim;
    float *b_out = W_out + (long)hidden_dim * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *h = malloc(sizeof(float) * emb_dim);
    float *hid = malloc(sizeof(float) * hidden_dim);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Sum embeddings
        for (int e = 0; e < emb_dim; e++) h[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((long)j * 256 + xi[j]) * emb_dim;
            for (int e = 0; e < emb_dim; e++) h[e] += eptr[e];
        }

        // Hidden layer: hid = ReLU(h @ W_hid + b_hid)
        for (int u = 0; u < hidden_dim; u++) {
            float sum = b_hid[u];
            for (int e = 0; e < emb_dim; e++) sum += h[e] * W_hid[(long)e * hidden_dim + u];
            hid[u] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b_out[c];
            for (int u = 0; u < hidden_dim; u++) sum += hid[u] * W_out[(long)u * C + c];
            logits[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    return 0;
}
```