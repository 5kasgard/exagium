```hypothesis
Family: label-embed-mlp – per-position byte embeddings (size 32) fed into a single ReLU hidden layer (size tuned on last 20% of train). The embeddings capture positional character statistics, and the hidden layer models non-linear interactions that a linear model misses. If dev accuracy does not exceed label-linear (0.3375) or inference cost per item grows more than ~2x, the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
E = 32                     # embedding size
H_CANDIDATES = [32, 64, 128]  # hidden sizes to try on tuning split


def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]          # 16

    n = len(X_bytes)
    split = int(0.8 * n)                # tuning split = last 20%
    X_train, X_val = X_bytes[:split], X_bytes[split:]
    y_train, y_val = y[:split], y[split:]

    best_H = None
    best_acc = -1.0
    rng = np.random.default_rng(SEED)

    # --- grid search on tuning split ---
    for H in H_CANDIDATES:
        # parameters
        emb = (rng.normal(0, 1 / np.sqrt(E), size=(d, 256, E))).astype(np.float32)
        W1 = (rng.normal(0, np.sqrt(2.0 / (d * E)), size=(d * E, H))).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.normal(0, np.sqrt(2.0 / H), size=(H, C))).astype(np.float32)
        b2 = np.zeros(C, np.float32)

        params = [emb, W1, b1, W2, b2]
        state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
        t = 0

        n_train = len(X_train)
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_train))
        for _ in range(epochs):
            order = rng.permutation(n_train)
            for s in range(0, n_train, BATCH):
                idx = order[s:s + BATCH]
                xb = X_train[idx]
                yb = y_train[idx]

                # forward
                x_emb_list = [emb[j, xb[:, j], :] for j in range(d)]
                x_flat = np.concatenate(x_emb_list, axis=1)          # (B, d*E)
                hidden = x_flat @ W1 + b1
                hidden_relu = np.maximum(hidden, 0)
                logits = hidden_relu @ W2 + b2

                # softmax + cross-entropy gradient
                logits -= logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits)
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

                dlogits = probs
                dlogits[np.arange(len(yb)), yb] -= 1.0
                dlogits /= len(yb)

                # backward
                dW2 = hidden_relu.T @ dlogits
                db2 = dlogits.sum(axis=0)
                dhidden_relu = dlogits @ W2.T
                dhidden = dhidden_relu * (hidden > 0)
                dW1 = x_flat.T @ dhidden
                db1 = dhidden.sum(axis=0)
                dx_flat = dhidden @ W1.T

                demb = np.zeros_like(emb)
                for j in range(d):
                    dx_emb_j = dx_flat[:, j * E:(j + 1) * E]
                    np.add.at(demb[j], xb[:, j], dx_emb_j)

                dW1 += L2 * W1
                dW2 += L2 * W2

                grads = [demb, dW1, db1, dW2, db2]
                t += 1
                for i, (p, gr) in enumerate(zip(params, grads)):
                    m, v = state[i]
                    m *= 0.9; m += 0.1 * gr
                    v *= 0.999; v += 0.001 * gr * gr
                    m_hat = m / (1 - 0.9 ** t)
                    v_hat = v / (1 - 0.999 ** t)
                    p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        # validation accuracy
        x_val_emb_list = [emb[j, X_val[:, j], :] for j in range(d)]
        x_val_flat = np.concatenate(x_val_emb_list, axis=1)
        hidden_val = x_val_flat @ W1 + b1
        hidden_relu_val = np.maximum(hidden_val, 0)
        logits_val = hidden_relu_val @ W2 + b2
        preds = logits_val.argmax(axis=1)
        acc = (preds == y_val).mean()
        if acc > best_acc:
            best_acc = acc
            best_H = H

    # --- retrain on full training set with best_H ---
    H = best_H
    rng = np.random.default_rng(SEED)
    emb = (rng.normal(0, 1 / np.sqrt(E), size=(d, 256, E))).astype(np.float32)
    W1 = (rng.normal(0, np.sqrt(2.0 / (d * E)), size=(d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, np.sqrt(2.0 / H), size=(H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X_bytes[idx]
            yb = y[idx]

            x_emb_list = [emb[j, xb[:, j], :] for j in range(d)]
            x_flat = np.concatenate(x_emb_list, axis=1)
            hidden = x_flat @ W1 + b1
            hidden_relu = np.maximum(hidden, 0)
            logits = hidden_relu @ W2 + b2

            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            dlogits = probs
            dlogits[np.arange(len(yb)), yb] -= 1.0
            dlogits /= len(yb)

            dW2 = hidden_relu.T @ dlogits
            db2 = dlogits.sum(axis=0)
            dhidden_relu = dlogits @ W2.T
            dhidden = dhidden_relu * (hidden > 0)
            dW1 = x_flat.T @ dhidden
            db1 = dhidden.sum(axis=0)
            dx_flat = dhidden @ W1.T

            demb = np.zeros_like(emb)
            for j in range(d):
                dx_emb_j = dx_flat[:, j * E:(j + 1) * E]
                np.add.at(demb[j], xb[:, j], dx_emb_j)

            dW1 += L2 * W1
            dW2 += L2 * W2

            grads = [demb, dW1, db1, dW2, db2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # combine embeddings and first layer into lookup table T[j][byte][h]
    T = np.zeros((d, 256, H), dtype=np.float32)
    for j in range(d):
        T[j] = emb[j] @ W1[j * E:(j + 1) * E, :]

    # write weights.bin
    magic = 0x454D424C
    out = struct.pack("<5i", magic, d, E, H, C)
    out += T.astype("<f4").tobytes()
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
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights params\n"); return 3; }
    long expected = 20L + (long)d * 256 * H * 4 + H * 4 + (long)H * C * 4 + C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hd + 5);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int u = 0; u < H; u++) acc[u] = b1[u];

        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H;
            for (int u = 0; u < H; u++) acc[u] += t[u];
        }

        for (int u = 0; u < H; u++) if (acc[u] < 0.0f) acc[u] = 0.0f;

        for (int c = 0; c < C; c++) scores[c] = b2[c];

        for (int u = 0; u < H; u++) {
            float hu = acc[u];
            if (hu != 0.0f) {
                const float *w = W2 + (long)u * C;
                for (int c = 0; c < C; c++) scores[c] += hu * w[c];
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;

        put_rec(stdout, &best, 1);
    }
    return 0;
}
```