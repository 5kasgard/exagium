```hypothesis
Family: label-embmlp (embedding + optional one-hidden-layer MLP).
Why: The task is next-character prediction from 16 bytes. A linear model on one-hot bytes (label-linear) ignores interactions between positions. Learned per-position embeddings (dimension E) compress the 256 categories into a dense space, capturing similarity between characters. A small hidden layer (size H) on top of concatenated embeddings can model position interactions. At inference, the embedding+first-layer is precomputed into a lookup table T[16][256][W] (W=H or C), so inference cost is similar to label-linear (16 table lookups + small matvec). If this does not beat label-linear on dev accuracy, the character-level structure may require deeper context or attention beyond a shallow MLP.
```

```python
"""label-embmlp: per-position embeddings + optional one-hidden-layer MLP.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy.
Exports: int32 magic 0x454D424C, d, E, H, C; float32 T[d][256][W], b1[W]; if H: W2[H][C], b2[C] (W=H if H>0 else C)."""
import json, math, struct, sys

import numpy as np

# Hyperparameter search space (embedding dim E, hidden size H; H=0 means linear on embeddings)
CONFIGS = [
    (8, 0), (16, 0), (32, 0),
    (8, 32), (16, 64), (32, 128),
]
SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def train_eval(config, X_train, y_train, X_tune, y_tune, rng_seed, epochs):
    """Train on X_train, return (model_params, tune_acc)."""
    E, H = config
    d = X_train.shape[1]
    C = 65
    W = H if H > 0 else C
    rng = np.random.default_rng(rng_seed)

    # Initialize embeddings: (d, 256, E)
    emb = (rng.normal(0, 1.0 / math.sqrt(E), (d, 256, E))).astype(np.float32)
    # First layer weights: (d*E, W)
    W1 = (rng.uniform(-1, 1, (d * E, W)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(W, np.float32)
    if H > 0:
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
    else:
        W2, b2 = None, None

    params = [emb, W1, b1]
    if H > 0:
        params += [W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X_train.shape[0]
    steps_per_epoch = (n + BATCH - 1) // BATCH
    total_steps = epochs * steps_per_epoch
    t = 0

    def forward(X_batch):
        # X_batch: (bs, d) uint8
        bs = X_batch.shape[0]
        # Embedding lookup: (bs, d, E)
        x_emb = emb[X_batch]  # (bs, d, E)
        x_flat = x_emb.reshape(bs, d * E)  # (bs, d*E)
        # First layer
        pre = x_flat @ W1 + b1  # (bs, W)
        if H > 0:
            hidden = np.maximum(pre, 0)
            logits = hidden @ W2 + b2
        else:
            logits = pre
        return logits, x_emb, x_flat, pre if H > 0 else None

    def softmax_cross_entropy_grad(logits, y):
        # logits: (bs, C), y: (bs,)
        bs = logits.shape[0]
        max_logits = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(bs), y] -= 1.0
        return probs / bs  # gradient w.r.t logits

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_train[idx], y_train[idx]
            logits, x_emb, x_flat, pre = forward(Xb)
            g_logits = softmax_cross_entropy_grad(logits, yb).astype(np.float32)

            grads = [None] * len(params)
            if H > 0:
                # g_logits: (bs, C)
                # W2 grad: (H, C)
                grad_W2 = x_flat.reshape(bs, H) @ g_logits  # Wait, hidden = relu(pre) shape (bs, H)
                hidden = np.maximum(pre, 0)
                grad_W2 = hidden.T @ g_logits
                grad_b2 = g_logits.sum(axis=0)
                # backprop to hidden
                g_hidden = g_logits @ W2.T  # (bs, H)
                g_pre = g_hidden * (pre > 0)  # (bs, H)
                # backprop to first layer
                grad_W1 = x_flat.T @ g_pre  # (d*E, H)
                grad_b1 = g_pre.sum(axis=0)
                # backprop to embeddings
                # g_pre is (bs, H), need to multiply by W1^T? Actually x_flat @ W1 = pre.
                # So gradient w.r.t x_flat is g_pre @ W1^T (bs, d*E)
                g_x_flat = g_pre @ W1.T  # (bs, d*E)
                g_x_emb = g_x_flat.reshape(bs, d, E)  # (bs, d, E)
                # Now accumulate gradient for emb: for each position j, byte value b, add g_x_emb[:, j, :]
                grad_emb = np.zeros_like(emb)
                for j in range(d):
                    bytes_j = Xb[:, j]
                    # Use np.add.at for unsorted indices
                    np.add.at(grad_emb[j], bytes_j, g_x_emb[:, j, :])
                grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            else:
                # Linear case: logits = x_flat @ W1 + b1
                grad_W1 = x_flat.T @ g_logits  # (d*E, C)
                grad_b1 = g_logits.sum(axis=0)
                g_x_flat = g_logits @ W1.T  # (bs, d*E)
                g_x_emb = g_x_flat.reshape(bs, d, E)
                grad_emb = np.zeros_like(emb)
                for j in range(d):
                    bytes_j = Xb[:, j]
                    np.add.at(grad_emb[j], bytes_j, g_x_emb[:, j, :])
                grads = [grad_emb, grad_W1, grad_b1]

            # Adam update with L2 on weights (not biases, not embeddings? We'll apply L2 to W1 and W2 only)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                if i == 1:  # W1
                    gr = gr + L2 * p
                elif H > 0 and i == 3:  # W2
                    gr = gr + L2 * p
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Evaluate on tuning set
    logits_tune, _, _, _ = forward(X_tune)
    y_pred = logits_tune.argmax(axis=1)
    acc = accuracy(y_tune, y_pred)
    return params, acc


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n_total = X_all.shape[0]
    n_tune = n_total // 5
    n_train = n_total - n_tune
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_tune, y_tune = X_all[n_train:], y_all[n_train:]

    # Hyperparameter selection on tuning split
    best_config = None
    best_acc = -1.0
    best_params = None
    rng_seed_base = SEED
    for i, config in enumerate(CONFIGS):
        # Train for a few epochs for selection
        params, acc = train_eval(config, X_train, y_train, X_tune, y_tune, rng_seed_base + i, epochs=10)
        if acc > best_acc:
            best_acc = acc
            best_config = config
            best_params = params
    print(f"Selected config: E={best_config[0]}, H={best_config[1]}, tune_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with more epochs
    E, H = best_config
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n_total))
    final_params, _ = train_eval(best_config, X_all, y_all, X_all[:1], y_all[:1], SEED + 100, epochs=epochs_full)

    # Unpack final parameters
    emb, W1, b1 = final_params[0], final_params[1], final_params[2]
    if H > 0:
        W2, b2 = final_params[3], final_params[4]
    else:
        W2, b2 = None, None

    # Precompute T: for each position j (0..d-1), each byte b (0..255), vector of size W
    W = H if H > 0 else C
    T = np.zeros((d, 256, W), dtype=np.float32)
    for j in range(d):
        # emb[j] is (256, E), W1 slice for position j: (E, W)
        W1_j = W1[j * E:(j + 1) * E, :]  # (E, W)
        T[j] = emb[j] @ W1_j  # (256, W)

    # Write weights.bin
    magic = 0x454D424C  # 'L','B','M','E' little-endian
    out = struct.pack("<5i", magic, d, E, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H > 0:
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte (class index 0..64).
 * Weights format: int32 magic(0x454D424C), d, E, H, C; float32 T[d][256][W], b1[W]; if H>0: W2[H][C], b2[C] (W=H if H>0 else C). */
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    int W = H ? H : C;
    if (d != 16 || C != 65 || E <= 0 || H < 0 || W <= 0) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0));
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *T = (const float *)(hd + 5);
    const float *b1 = T + (long)d * 256 * W;
    const float *W2 = b1 + W;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W);
    float *s = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        const float *sc = acc;
        if (H) {
            for (int c = 0; c < C; c++) s[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) s[c] += h * W2[(long)u * C + c];
            }
            sc = s;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```