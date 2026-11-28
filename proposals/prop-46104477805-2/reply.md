```hypothesis
The label-emb family uses per-position character embeddings (dimension 8) followed by a linear classifier. This captures inter-character similarities (e.g., vowels cluster) and reduces overfitting compared to the one-hot linear model, while adding only ~40k parameters. On the tuning split we select embedding size and learning rate; final model trained on all data. Expected to beat label-linear (0.3375) on dev accuracy with similar or lower inference cost (Ir/item ~10k). If dev accuracy ≤0.33, the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = records(sys.argv[2])
    y_bytes = records(sys.argv[3])
    X = np.array([list(b) for b in X_bytes], dtype=np.uint8)  # (n, 16)
    y = np.array([b[0] for b in y_bytes], dtype=np.int64)   # (n,)
    n, d = X.shape
    C = card["classes"]
    assert d == 16

    # tuning split: last 20% in file order
    val_start = int(0.8 * n)
    train_X, train_y = X[:val_start], y[:val_start]
    val_X, val_y = X[val_start:], y[val_start:]
    n_train = train_X.shape[0]

    best_acc = -1.0
    best_params = None
    best_emb = None
    best_W = None
    best_b = None

    # hyperparameter grid
    for D in [4, 8, 16]:
        for lr in [0.01, 0.03, 0.1]:
            for L2 in [1e-4, 1e-3]:
                rng = np.random.default_rng(12345 + D * 100 + int(lr * 1000) + int(L2 * 10000))
                embeddings = rng.normal(0, 0.1, size=(d, 256, D)).astype(np.float32)
                W = rng.normal(0, 0.1, size=(d * D, C)).astype(np.float32)
                b = np.zeros(C, dtype=np.float32)

                m_emb = np.zeros_like(embeddings); v_emb = np.zeros_like(embeddings)
                m_W = np.zeros_like(W); v_W = np.zeros_like(W)
                m_b = np.zeros_like(b); v_b = np.zeros_like(b)
                t = 0
                epochs = 10
                batch_size = 64
                for _ in range(epochs):
                    order = rng.permutation(n_train)
                    for s in range(0, n_train, batch_size):
                        idx = order[s:s + batch_size]
                        xb = train_X[idx]
                        yb = train_y[idx]
                        B = len(idx)
                        # forward
                        emb = embeddings[:, xb.T, :].transpose(1, 0, 2)  # (B, 16, D)
                        flat = emb.reshape(B, -1)
                        logits = flat @ W + b
                        # softmax
                        logits_max = logits.max(axis=1, keepdims=True)
                        exp_logits = np.exp(logits - logits_max)
                        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                        # gradient
                        grad_logits = probs
                        grad_logits[np.arange(B), yb] -= 1.0
                        grad_logits /= B
                        grad_W = flat.T @ grad_logits + L2 * W
                        grad_b = grad_logits.sum(axis=0)
                        grad_flat = grad_logits @ W.T
                        grad_emb = grad_flat.reshape(B, 16, D).transpose(1, 0, 2)  # (16, B, D)
                        grad_embeddings = np.zeros_like(embeddings)
                        for j in range(16):
                            np.add.at(grad_embeddings[j], xb[:, j], grad_emb[j])
                        grad_embeddings += L2 * embeddings

                        t += 1
                        # Adam updates
                        m_emb = 0.9 * m_emb + 0.1 * grad_embeddings
                        v_emb = 0.999 * v_emb + 0.001 * grad_embeddings * grad_embeddings
                        m_hat_emb = m_emb / (1 - 0.9 ** t)
                        v_hat_emb = v_emb / (1 - 0.999 ** t)
                        embeddings -= lr * m_hat_emb / (np.sqrt(v_hat_emb) + 1e-8)

                        m_W = 0.9 * m_W + 0.1 * grad_W
                        v_W = 0.999 * v_W + 0.001 * grad_W * grad_W
                        m_hat_W = m_W / (1 - 0.9 ** t)
                        v_hat_W = v_W / (1 - 0.999 ** t)
                        W -= lr * m_hat_W / (np.sqrt(v_hat_W) + 1e-8)

                        m_b = 0.9 * m_b + 0.1 * grad_b
                        v_b = 0.999 * v_b + 0.001 * grad_b * grad_b
                        m_hat_b = m_b / (1 - 0.9 ** t)
                        v_hat_b = v_b / (1 - 0.999 ** t)
                        b -= lr * m_hat_b / (np.sqrt(v_hat_b) + 1e-8)

                # evaluate on validation
                B_val = val_X.shape[0]
                emb_val = embeddings[:, val_X.T, :].transpose(1, 0, 2)
                flat_val = emb_val.reshape(B_val, -1)
                logits_val = flat_val @ W + b
                preds = logits_val.argmax(axis=1)
                acc = (preds == val_y).mean()
                if acc > best_acc:
                    best_acc = acc
                    best_params = (D, lr, L2)
                    best_emb = embeddings.copy()
                    best_W = W.copy()
                    best_b = b.copy()

    # retrain on full data with best hyperparameters
    D, lr, L2 = best_params
    rng = np.random.default_rng(12345 + D * 100 + int(lr * 1000) + int(L2 * 10000))
    embeddings = rng.normal(0, 0.1, size=(d, 256, D)).astype(np.float32)
    W = rng.normal(0, 0.1, size=(d * D, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    m_emb = np.zeros_like(embeddings); v_emb = np.zeros_like(embeddings)
    m_W = np.zeros_like(W); v_W = np.zeros_like(W)
    m_b = np.zeros_like(b); v_b = np.zeros_like(b)
    t = 0
    epochs = 30
    batch_size = 64
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            xb = X[idx]
            yb = y[idx]
            B = len(idx)
            emb = embeddings[:, xb.T, :].transpose(1, 0, 2)
            flat = emb.reshape(B, -1)
            logits = flat @ W + b
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(B), yb] -= 1.0
            grad_logits /= B
            grad_W = flat.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
            grad_flat = grad_logits @ W.T
            grad_emb = grad_flat.reshape(B, 16, D).transpose(1, 0, 2)
            grad_embeddings = np.zeros_like(embeddings)
            for j in range(16):
                np.add.at(grad_embeddings[j], xb[:, j], grad_emb[j])
            grad_embeddings += L2 * embeddings

            t += 1
            m_emb = 0.9 * m_emb + 0.1 * grad_embeddings
            v_emb = 0.999 * v_emb + 0.001 * grad_embeddings * grad_embeddings
            m_hat_emb = m_emb / (1 - 0.9 ** t)
            v_hat_emb = v_emb / (1 - 0.999 ** t)
            embeddings -= lr * m_hat_emb / (np.sqrt(v_hat_emb) + 1e-8)

            m_W = 0.9 * m_W + 0.1 * grad_W
            v_W = 0.999 * v_W + 0.001 * grad_W * grad_W
            m_hat_W = m_W / (1 - 0.9 ** t)
            v_hat_W = v_W / (1 - 0.999 ** t)
            W -= lr * m_hat_W / (np.sqrt(v_hat_W) + 1e-8)

            m_b = 0.9 * m_b + 0.1 * grad_b
            v_b = 0.999 * v_b + 0.001 * grad_b * grad_b
            m_hat_b = m_b / (1 - 0.9 ** t)
            v_hat_b = v_b / (1 - 0.999 ** t)
            b -= lr * m_hat_b / (np.sqrt(v_hat_b) + 1e-8)

    # save weights
    magic = 0x4C424D45  # 'L','B','M','E' little-endian
    out = struct.pack("<4i", magic, d, D, C)
    out += embeddings.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x4C424D45) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C != 65 || D < 1 || D > 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * D + (long)d * D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * D;
    const float *b = W + (long)d * D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = malloc(sizeof(float) * d * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // build feature vector: concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            for (int kk = 0; kk < D; kk++) {
                feat[j * D + kk] = e[kk];
            }
        }

        // compute scores = feat * W + b
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int f = 0; f < d * D; f++) {
            float fv = feat[f];
            const float *w = W + (long)f * C;
            for (int c = 0; c < C; c++) scores[c] += fv * w[c];
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(feat); free(scores);
    if (x) free(x);
    return 0;
}
```