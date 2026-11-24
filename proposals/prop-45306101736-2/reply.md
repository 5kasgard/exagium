```hypothesis
Family: label-emb-mlp. Learns position-specific byte embeddings (dim E) followed by a one-hidden-layer ReLU MLP (size H). Captures non-linear interactions between context bytes more efficiently than linear or bottleneck MLPs. Embedding reduces 4096 one-hot dims to 16*E dense features, enabling a wider hidden layer. If dev accuracy ≤ 0.3375 (label-linear), the family fails to improve on this task.
```

```python
import json, math, struct, sys

import numpy as np

# Fixed seed for determinism
SEED = 12345
np.random.seed(SEED)

# Hyperparameter grid for tuning on the last 20% of train.in
EMBED_DIMS = [16, 32, 64]
HIDDEN_SIZES = [64, 128, 256]
LEARNING_RATES = [1e-3, 3e-3]
L2 = 1e-5
BATCH_SIZE = 256
MAX_EPOCHS_TUNE = 20
MAX_EPOCHS_FULL = 50
PATIENCE = 5


def load_records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def evaluate(X, y, emb, W1, b1, W2, b2):
    d, _, E = emb.shape
    batch_size = 1024
    correct = 0
    total = 0
    for s in range(0, len(X), batch_size):
        xb = X[s:s + batch_size]
        yb = y[s:s + batch_size]
        batch_emb = np.zeros((len(xb), d, E), np.float32)
        for j in range(d):
            batch_emb[:, j, :] = emb[j, xb[:, j]]
        flat = batch_emb.reshape(len(xb), -1)
        h = flat @ W1 + b1
        h = np.maximum(h, 0)
        logits = h @ W2 + b2
        preds = logits.argmax(axis=1)
        correct += (preds == yb).sum()
        total += len(yb)
    return correct / total


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]  # 16
    C = card["classes"]         # 65

    X_recs = load_records(sys.argv[2])
    y_recs = load_records(sys.argv[3])
    X = np.frombuffer(b"".join(X_recs), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in y_recs], dtype=np.int64)
    n = X.shape[0]

    # Split: last 20% for validation (tuning split)
    val_size = n // 5
    train_size = n - val_size
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    best_acc = -1.0
    best_params = None

    for E in EMBED_DIMS:
        for H in HIDDEN_SIZES:
            for lr in LEARNING_RATES:
                rng = np.random.default_rng(SEED)
                emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
                W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
                b1 = np.zeros(H, np.float32)
                W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
                b2 = np.zeros(C, np.float32)

                params = [emb, W1, b1, W2, b2]
                m = [np.zeros_like(p) for p in params]
                v = [np.zeros_like(p) for p in params]
                t = 0
                beta1, beta2, eps = 0.9, 0.999, 1e-8

                best_val_acc = -1.0
                wait = 0
                for epoch in range(MAX_EPOCHS_TUNE):
                    perm = np.random.permutation(train_size)
                    for s in range(0, train_size, BATCH_SIZE):
                        idx = perm[s:s + BATCH_SIZE]
                        xb = X_train[idx]
                        yb = y_train[idx]

                        batch_emb = np.zeros((len(xb), d, E), np.float32)
                        for j in range(d):
                            batch_emb[:, j, :] = emb[j, xb[:, j]]
                        flat = batch_emb.reshape(len(xb), -1)

                        h = flat @ W1 + b1
                        h = np.maximum(h, 0)
                        logits = h @ W2 + b2

                        logits_max = logits.max(axis=1, keepdims=True)
                        exp_logits = np.exp(logits - logits_max)
                        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

                        grad_logits = probs
                        grad_logits[np.arange(len(yb)), yb] -= 1.0
                        grad_logits /= len(yb)

                        grad_W2 = h.T @ grad_logits + L2 * W2
                        grad_b2 = grad_logits.sum(axis=0)

                        grad_h = grad_logits @ W2.T
                        grad_h[h <= 0] = 0

                        grad_flat = grad_h
                        grad_W1 = flat.T @ grad_flat + L2 * W1
                        grad_b1 = grad_flat.sum(axis=0)

                        grad_emb = np.zeros_like(emb)
                        grad_flat_reshaped = grad_flat.reshape(len(xb), d, E)
                        for j in range(d):
                            np.add.at(grad_emb[j], xb[:, j], grad_flat_reshaped[:, j, :])
                        grad_emb += L2 * emb

                        grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
                        t += 1
                        for i, (p, gr) in enumerate(zip(params, grads)):
                            m[i] = beta1 * m[i] + (1 - beta1) * gr
                            v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                            m_hat = m[i] / (1 - beta1 ** t)
                            v_hat = v[i] / (1 - beta2 ** t)
                            p -= lr * m_hat / (np.sqrt(v_hat) + eps)

                    val_acc = evaluate(X_val, y_val, emb, W1, b1, W2, b2)
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        wait = 0
                    else:
                        wait += 1
                        if wait >= PATIENCE:
                            break

                if best_val_acc > best_acc:
                    best_acc = best_val_acc
                    best_params = (E, H, lr)

    # Retrain best config on full data
    E, H, lr = best_params
    rng = np.random.default_rng(SEED)
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for epoch in range(MAX_EPOCHS_FULL):
        perm = np.random.permutation(n)
        for s in range(0, n, BATCH_SIZE):
            idx = perm[s:s + BATCH_SIZE]
            xb = X[idx]
            yb = y[idx]

            batch_emb = np.zeros((len(xb), d, E), np.float32)
            for j in range(d):
                batch_emb[:, j, :] = emb[j, xb[:, j]]
            flat = batch_emb.reshape(len(xb), -1)

            h = flat @ W1 + b1
            h = np.maximum(h, 0)
            logits = h @ W2 + b2

            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            grad_logits = probs
            grad_logits[np.arange(len(yb)), yb] -= 1.0
            grad_logits /= len(yb)

            grad_W2 = h.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)

            grad_h = grad_logits @ W2.T
            grad_h[h <= 0] = 0

            grad_flat = grad_h
            grad_W1 = flat.T @ grad_flat + L2 * W1
            grad_b1 = grad_flat.sum(axis=0)

            grad_emb = np.zeros_like(emb)
            grad_flat_reshaped = grad_flat.reshape(len(xb), d, E)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_flat_reshaped[:, j, :])
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # Save weights: magic, d, E, H, C, then emb, W1, b1, W2, b2 (all float32 little-endian)
    magic = 0x454D424D  # 'EMBM'
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
    uint32_t k = buf[*pos] | (buf[*pos+1]<<8) | (buf[*pos+2]<<16) | ((uint32_t)buf[*pos+3]<<24);
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *r = buf + *pos + 4;
    *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, (k>>8)&255, (k>>16)&255, (k>>24)&255};
    fwrite(h, 1, 4, f);
    fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 20
        + (long)d * 256 * E * 4
        + (long)d * E * H * 4
        + (long)H * 4
        + (long)H * C * 4
        + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    long N = atol(argv[3]);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    long pos = 0;

    float *flat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *output = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k;
        const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(flat + j * E, e, E * sizeof(float));
        }

        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int f = 0; f < d * E; f++) {
                sum += flat[f] * W1[(long)f * H + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) {
                sum += hidden[h] * W2[(long)h * C + c];
            }
            output[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (output[c] > output[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```