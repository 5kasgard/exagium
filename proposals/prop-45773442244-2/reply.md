```hypothesis
Propose label-embed-mlp: position-specific byte embeddings fed into a one-hidden-layer ReLU MLP.
Character-level language modeling benefits from distributed representations that capture similarity between characters
and from nonlinear combination of context positions. The embedding layer reduces the one-hot dimensionality and shares
statistical strength across positions. If the best validation accuracy on the tuning split does not exceed label-linear
by at least 1 percentage point, the added complexity is not justified.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameter grid for tuning split
GRID = [
    {"E": 8, "H": 64, "lr": 1e-3, "wd": 1e-4, "epochs": 50, "batch": 64},
    {"E": 16, "H": 128, "lr": 1e-3, "wd": 1e-4, "epochs": 50, "batch": 64},
    {"E": 16, "H": 256, "lr": 5e-4, "wd": 1e-4, "epochs": 50, "batch": 64},
    {"E": 32, "H": 128, "lr": 1e-3, "wd": 1e-4, "epochs": 50, "batch": 64},
]

SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    # Last 20% for validation (tuning split)
    val_size = n // 5
    train_size = n - val_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    best_val_acc = -1.0
    best_config = None
    best_params = None

    rng = np.random.default_rng(SEED)

    for config in GRID:
        E = config["E"]
        H = config["H"]
        lr = config["lr"]
        wd = config["wd"]
        epochs = config["epochs"]
        batch = config["batch"]

        # Initialize parameters
        emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
        W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
        b2 = np.zeros(C, np.float32)

        params = [emb, W1, b1, W2, b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t = 0

        for epoch in range(epochs):
            order = rng.permutation(train_size)
            for s in range(0, train_size, batch):
                idx = order[s:s + batch]
                x_batch = X_train[idx]
                y_batch = y_train[idx]
                B = x_batch.shape[0]

                # Forward
                emb_concat = np.concatenate([emb[j, x_batch[:, j], :] for j in range(d)], axis=1)
                hidden_pre = emb_concat @ W1 + b1
                hidden = np.maximum(hidden_pre, 0)
                logits = hidden @ W2 + b2

                # Loss gradient
                logits_max = logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits - logits_max)
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                grad_logits = probs.copy()
                grad_logits[np.arange(B), y_batch] -= 1.0
                grad_logits /= B

                # Backprop
                grad_W2 = hidden.T @ grad_logits + wd * W2
                grad_b2 = grad_logits.sum(axis=0)
                grad_hidden = grad_logits @ W2.T
                grad_hidden_pre = grad_hidden * (hidden_pre > 0)
                grad_W1 = emb_concat.T @ grad_hidden_pre + wd * W1
                grad_b1 = grad_hidden_pre.sum(axis=0)

                grad_emb = np.zeros_like(emb)
                grad_emb_concat = grad_hidden_pre @ W1.T
                for j in range(d):
                    start = j * E
                    end = (j + 1) * E
                    grad_emb_j = grad_emb_concat[:, start:end]
                    indices = x_batch[:, j]
                    np.add.at(grad_emb[j], indices, grad_emb_j)
                grad_emb += wd * emb

                grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

                # Adam update
                t += 1
                beta1, beta2 = 0.9, 0.999
                for i, (p, gr) in enumerate(zip(params, grads)):
                    m[i] = beta1 * m[i] + (1 - beta1) * gr
                    v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                    m_hat = m[i] / (1 - beta1 ** t)
                    v_hat = v[i] / (1 - beta2 ** t)
                    p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Validate
        emb_concat_val = np.concatenate([emb[j, X_val[:, j], :] for j in range(d)], axis=1)
        hidden_val = np.maximum(emb_concat_val @ W1 + b1, 0)
        logits_val = hidden_val @ W2 + b2
        preds = logits_val.argmax(axis=1)
        val_acc = (preds == y_val).mean()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_config = config
            best_params = [emb.copy(), W1.copy(), b1.copy(), W2.copy(), b2.copy()]

    print(f"Best config: {best_config}, val acc: {best_val_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best config
    config = best_config
    E = config["E"]; H = config["H"]; lr = config["lr"]; wd = config["wd"]; epochs = config["epochs"]; batch = config["batch"]
    emb, W1, b1, W2, b2 = best_params
    params = [emb, W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            x_batch = X[idx]
            y_batch = y[idx]
            B = x_batch.shape[0]

            emb_concat = np.concatenate([emb[j, x_batch[:, j], :] for j in range(d)], axis=1)
            hidden_pre = emb_concat @ W1 + b1
            hidden = np.maximum(hidden_pre, 0)
            logits = hidden @ W2 + b2

            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs.copy()
            grad_logits[np.arange(B), y_batch] -= 1.0
            grad_logits /= B

            grad_W2 = hidden.T @ grad_logits + wd * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_hidden = grad_logits @ W2.T
            grad_hidden_pre = grad_hidden * (hidden_pre > 0)
            grad_W1 = emb_concat.T @ grad_hidden_pre + wd * W1
            grad_b1 = grad_hidden_pre.sum(axis=0)

            grad_emb = np.zeros_like(emb)
            grad_emb_concat = grad_hidden_pre @ W1.T
            for j in range(d):
                start = j * E
                end = (j + 1) * E
                grad_emb_j = grad_emb_concat[:, start:end]
                indices = x_batch[:, j]
                np.add.at(grad_emb[j], indices, grad_emb_j)
            grad_emb += wd * emb

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            beta1, beta2 = 0.9, 0.999
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Save weights: magic 0x44424D45 ('EMBD' little-endian), d, E, H, C
    magic = 0x44424D45
    out = struct.pack("<5i", magic, d, E, H, C)
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
    if (nw < 20 || hd[0] != 0x44424D45) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C < 1 || C > 256 || E <= 0 || H <= 0) { fprintf(stderr, "bad header values\n"); return 3; }

    long expected = 20L + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden_pre = malloc(sizeof(float) * H);
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // hidden_pre = b1
        for (int h = 0; h < H; h++) hidden_pre[h] = b1[h];

        // For each position, add embedding * corresponding slice of W1
        for (int j = 0; j < d; j++) {
            const float *emb_vec = emb + ((long)j * 256 + xi[j]) * E;
            const float *W1_slice = W1 + (long)j * E * H;
            for (int e = 0; e < E; e++) {
                float ev = emb_vec[e];
                const float *w = W1_slice + (long)e * H;
                for (int h = 0; h < H; h++) hidden_pre[h] += ev * w[h];
            }
        }

        // ReLU
        for (int h = 0; h < H; h++) hidden[h] = hidden_pre[h] > 0.0f ? hidden_pre[h] : 0.0f;

        // Output layer
        for (int c = 0; c < C; c++) scores[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float hv = hidden[h];
            const float *w = W2 + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += hv * w[c];
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(hidden_pre); free(hidden); free(scores);
    if (x) free(x);
    return 0;
}
```