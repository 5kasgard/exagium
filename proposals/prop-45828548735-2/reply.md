```hypothesis
We propose a position-wise embedding model with one hidden ReLU layer (emb-mlp). Each of the 16 context bytes gets a learned dense embedding (size E), concatenated and passed through a hidden layer (size H) then to 65 output classes. This subsumes label-linear (E=65, H=0) but can use smaller E and non-linearity to capture byte interactions. Tuning E,H on the last 20% of train should yield >33.75% dev accuracy with similar or lower inference cost (smaller lookup tables). If the best config has H=0 and E≥65, the family offers no advantage.
```

```python
"""emb-mlp: position-wise embeddings + one ReLU layer for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weight format: magic 0x454D4231, int32 d, H, C; float32 T[d][256][W], b1[W];
if H>0: float32 W2[H][C], b2[C]  (W = H if H>0 else C)."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
# Hyperparameter grid for tuning split (last 20% of train)
E_CHOICES = [8, 16, 32, 64]
H_CHOICES = [0, 32, 64, 128]   # 0 means linear on embeddings


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_model(X, y, d, C, E, H, rng, n_iter):
    """Train on given data for n_iter minibatches, return parameters."""
    n = X.shape[0]
    W = H if H else C
    # Initialize embeddings: (d, 256, E)
    emb = rng.normal(0, 0.01, (d, 256, E)).astype(np.float32)
    # First layer weights: (d*E, W)
    W1 = rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, W)).astype(np.float32)
    b1 = np.zeros(W, np.float32)
    if H:
        W2 = rng.normal(0, math.sqrt(2.0 / H), (H, C)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W1, b1, W2, b2]
    else:
        params = [emb, W1, b1]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(z):
        # z: (batch, d*E)
        a = z @ W1 + b1
        if H:
            h = np.maximum(a, 0)
            return h @ W2 + b2, (z, a, h)
        return a, (z, a)

    def backward(cache, grad_out):
        if H:
            z, a, h = cache
            grad_W2 = h.T @ grad_out
            grad_b2 = grad_out.sum(0)
            grad_h = grad_out @ W2.T
            grad_a = grad_h * (a > 0)
            grad_W1 = z.T @ grad_a
            grad_b1 = grad_a.sum(0)
            grad_z = grad_a @ W1.T
            # grad_emb: need to scatter grad_z back to embeddings
            grad_emb = np.zeros_like(emb)
            grad_z_reshaped = grad_z.reshape(-1, d, E)
            for j in range(d):
                bytes_j = X_batch[:, j]
                np.add.at(grad_emb[j], bytes_j, grad_z_reshaped[:, j])
            return [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
        else:
            z, a = cache
            grad_W1 = z.T @ grad_out
            grad_b1 = grad_out.sum(0)
            grad_z = grad_out @ W1.T
            grad_emb = np.zeros_like(emb)
            grad_z_reshaped = grad_z.reshape(-1, d, E)
            for j in range(d):
                bytes_j = X_batch[:, j]
                np.add.at(grad_emb[j], bytes_j, grad_z_reshaped[:, j])
            return [grad_emb, grad_W1, grad_b1]

    for _ in range(n_iter):
        idx = rng.choice(n, BATCH, replace=False)
        X_batch = X[idx]
        y_batch = y[idx]
        # Build concatenated embeddings for batch
        z = np.zeros((BATCH, d * E), np.float32)
        for j in range(d):
            z[:, j*E:(j+1)*E] = emb[j, X_batch[:, j]]
        logits, cache = forward(z)
        # Softmax cross-entropy gradient
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(BATCH), y_batch] -= 1.0
        grad_out = (probs / BATCH).astype(np.float32)
        grads = backward(cache, grad_out)
        # Adam update
        t += 1
        for i, (p, gr) in enumerate(zip(params, grads)):
            if H == 0 and i == 0:
                gr += L2 * p  # L2 on embeddings only when linear? Actually we apply L2 to all weights.
            else:
                gr += L2 * p
            m, v = state[i]
            m *= 0.9; m += 0.1 * gr
            v *= 0.999; v += 0.001 * gr * gr
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params


def eval_acc(X, y, params, d, C, E, H):
    n = X.shape[0]
    W = H if H else C
    emb, W1, b1 = params[0], params[1], params[2]
    if H:
        W2, b2 = params[3], params[4]
    correct = 0
    for i in range(0, n, 1024):
        batch = X[i:i+1024]
        z = np.zeros((len(batch), d * E), np.float32)
        for j in range(d):
            z[:, j*E:(j+1)*E] = emb[j, batch[:, j]]
        a = z @ W1 + b1
        if H:
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
        else:
            logits = a
        pred = logits.argmax(axis=1)
        correct += (pred == y[i:i+len(batch)]).sum()
    return correct / n


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = len(X_all)
    split = int(0.8 * n)
    X_train, y_train = X_all[:split], y_all[:split]
    X_tune, y_tune = X_all[split:], y_all[split:]

    rng = np.random.default_rng(SEED)
    best_acc, best_params, best_E, best_H = -1, None, None, None

    for E in E_CHOICES:
        for H in H_CHOICES:
            # Train on tuning split for a few epochs to evaluate
            n_iter = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train))) * (len(X_train) // BATCH)
            # Actually we need to train for a fixed number of epochs on the tuning split? We'll train for a small fixed number of epochs.
            # To save time, we train for 5 epochs on the tuning split.
            epochs = 5
            n_iter_tune = epochs * (len(X_train) // BATCH)
            params = train_model(X_train, y_train, d, C, E, H, rng, n_iter_tune)
            acc = eval_acc(X_tune, y_tune, params, d, C, E, H)
            if acc > best_acc:
                best_acc, best_params, best_E, best_H = acc, params, E, H
                print(f"New best: E={E}, H={H}, acc={acc:.4f}", file=sys.stderr)

    # Retrain on full data with best hyperparameters
    print(f"Final training with E={best_E}, H={best_H}", file=sys.stderr)
    n_iter_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n)) * (n // BATCH)
    final_params = train_model(X_all, y_all, d, C, best_E, best_H, rng, n_iter_full)

    # Prepare weights for export: compute T = emb @ W1 per position
    emb, W1, b1 = final_params[0], final_params[1], final_params[2]
    W = best_H if best_H else C
    T = np.zeros((d, 256, W), np.float32)
    for j in range(d):
        T[j] = emb[j] @ W1[j*best_E:(j+1)*best_E]
    # Pack weights
    magic = 0x454D4231  # "EMB1"
    out = struct.pack("<4i", magic, d, best_H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if best_H:
        W2, b2 = final_params[3], final_params[4]
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: one byte class index.
 * Weight format: magic 0x454D4231, int32 d, H, C; float32 T[d][256][W], b1[W];
 * if H>0: float32 W2[H][C], b2[C]  (W = H if H>0 else C). */
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
    if (nw < 16 || hd[0] != 0x454D4231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256) { fprintf(stderr, "bad class count\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0));
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }
    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * W;
    const float *W2 = b1 + W;
    const float *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W);
    float *sc = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        if (H) {
            for (int c = 0; c < C; c++) sc[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) sc[c] += h * W2[(long)u * C + c];
            }
        } else {
            for (int c = 0; c < C; c++) sc[c] = acc[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```