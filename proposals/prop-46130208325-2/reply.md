```hypothesis
Family: label-embed. Uses byte embeddings (dim E) + one hidden ReLU layer (dim H). Embeddings capture character similarities, reducing input dimensionality from 4096 (one-hot) to 16*E. Should beat linear (0.3375) with fewer parameters and faster inference than GBRT. If dev accuracy ≤ 0.3375 or inference cost > label-linear, the hypothesis fails.
```

```python
"""label-embed: byte embeddings + one hidden ReLU layer for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (file order), then retrain on all.
Weights format: magic 0x454D4244, d, E, H, C; embeddings[256*E], W1[(d*E)*H], b1[H], W2[H*C], b2[C] (all float32 le)."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 128
SAMPLES = 600000
MAX_EPOCHS = 100

# Hyperparameter grid for tuning (E, H)
GRID = [(8, 32), (16, 64), (16, 32), (8, 64), (4, 32)]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()


def train_model(X, y, C, E, H, rng, epochs, batch_size):
    n, d = X.shape
    # Initialize parameters
    emb = (rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            xb, yb = X[idx], y[idx]
            # Forward
            emb_x = emb[xb]  # (batch, d, E)
            flat = emb_x.reshape(len(xb), -1)  # (batch, d*E)
            h_pre = flat @ W1 + b1
            h = np.maximum(h_pre, 0, dtype=np.float32)
            logits = h @ W2 + b2

            # Loss gradient (cross-entropy)
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            probs /= len(yb)

            # Backward
            grad_W2 = h.T @ probs + L2 * W2
            grad_b2 = probs.sum(axis=0)
            grad_h = probs @ W2.T
            grad_h[h_pre <= 0] = 0.0
            grad_W1 = flat.T @ grad_h + L2 * W1
            grad_b1 = grad_h.sum(axis=0)
            grad_flat = grad_h @ W1.T  # (batch, d*E)
            grad_emb_x = grad_flat.reshape(len(xb), d, E)

            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb, xb[:, j], grad_emb_x[:, j, :])
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    return params


def predict(X, params, C):
    emb, W1, b1, W2, b2 = params
    n, d = X.shape
    emb_x = emb[X]
    flat = emb_x.reshape(n, -1)
    h = np.maximum(flat @ W1 + b1, 0, dtype=np.float32)
    logits = h @ W2 + b2
    return logits.argmax(axis=1).astype(np.int64)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]

    # Tuning split: last 20% in file order
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / max(1, len(X_train))))

    best_acc, best_params, best_cfg = -1, None, None
    for E, H in GRID:
        print(f"Trying E={E}, H={H}", file=sys.stderr)
        params = train_model(X_train, y_train, C, E, H, rng, epochs, BATCH)
        pred = predict(X_val, params, C)
        acc = accuracy(y_val, pred)
        print(f"  val acc={acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc, best_params, best_cfg = acc, params, (E, H)

    print(f"Best cfg: E={best_cfg[0]}, H={best_cfg[1]}, val acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on all data
    rng = np.random.default_rng(SEED)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    final_params = train_model(X, y, C, best_cfg[0], best_cfg[1], rng, epochs, BATCH)

    # Pack weights
    E, H = best_cfg
    emb, W1, b1, W2, b2 = final_params
    magic = 0x454D4244  # "EMBD"
    header = struct.pack("<5i", magic, d, E, H, C)
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
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output record: 1 byte class index.
 * Weights format: magic 0x454D4244, d, E, H, C; embeddings[256*E], W1[(d*E)*H], b1[H], W2[H*C], b2[C] (float32 le). */
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
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 20
        + 4L * 256 * E
        + 4L * d * E * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);
    float *flat = malloc(sizeof(float) * d * E);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Gather embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + (long)xi[j] * E;
            memcpy(flat + (long)j * E, e, E * sizeof(float));
        }

        // Hidden layer: ReLU(flat * W1 + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < d * E; v++) sum += flat[v] * W1[(long)v * H + u];
            h[u] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[(long)u * C + c];
            scores[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(h); free(scores); free(flat);
    if (x) free(x);
    return 0;
}
```