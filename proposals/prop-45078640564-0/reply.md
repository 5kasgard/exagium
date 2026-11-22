```hypothesis
Family: label-fm (factorization machine with per-feature-value embeddings).
Why: Adult data has strong feature interactions (e.g., education × occupation). A factorization machine captures pairwise interactions via low-rank embeddings, adding only O(d·k) parameters per feature value. With k=8 it should beat label-linear (0.8524) while staying far cheaper than label-mlp16.
Failure: dev accuracy ≤ 0.8524 or inference instructions/item > 2× label-linear.
```
```python
import json, math, struct, sys

import numpy as np

SEED = 1
GRID_EPOCHS = 10
FINAL_EPOCHS = 30
BATCH = 256

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
    val_split = int(0.8 * n)
    X_train, y_train = X[:val_split], y[:val_split]
    X_val, y_val = X[val_split:], y[val_split:]

    # hyperparameter grid
    best_acc = -1
    best_params = None
    best_state = None

    for k in (4, 8, 16):
        for lr in (0.01, 0.03):
            for l2 in (1e-4, 1e-3):
                rng = np.random.default_rng(SEED)
                E = rng.normal(0, 0.01, (d, 256, k)).astype(np.float32)
                W = np.zeros((d, 256), np.float32)
                mE, vE = np.zeros_like(E), np.zeros_like(E)
                mW, vW = np.zeros_like(W), np.zeros_like(W)
                t = 0

                for _ in range(GRID_EPOCHS):
                    order = rng.permutation(len(X_train))
                    for s in range(0, len(X_train), BATCH):
                        idx = order[s:s+BATCH]
                        xb, yb = X_train[idx], y_train[idx]
                        B = len(yb)
                        sum_emb = np.zeros((B, k), np.float32)
                        sum_emb_sq = np.zeros((B, k), np.float32)
                        linear = np.zeros(B, np.float32)
                        for f in range(d):
                            vals = xb[:, f]
                            emb_f = E[f, vals]
                            sum_emb += emb_f
                            sum_emb_sq += emb_f * emb_f
                            linear += W[f, vals]
                        interaction = 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=1)
                        logits = linear + interaction
                        probs = 1.0 / (1.0 + np.exp(-logits))
                        dlogits = probs - yb

                        # gradients
                        for f in range(d):
                            vals = xb[:, f]
                            # linear
                            np.add.at(W, (f, vals), dlogits[:, None] / B)
                            # embedding
                            grad_emb = (sum_emb - E[f, vals]) * dlogits[:, None]
                            np.add.at(E, (f, vals), grad_emb / B)

                        # Adam update (simplified: apply L2 after gradient)
                        t += 1
                        for param, m, v, grad in [(W, mW, vW, None), (E, mE, vE, None)]:
                            pass  # we'll do manual Adam below

                # validation accuracy
                sum_emb = np.zeros((len(X_val), k), np.float32)
                sum_emb_sq = np.zeros((len(X_val), k), np.float32)
                linear = np.zeros(len(X_val), np.float32)
                for f in range(d):
                    vals = X_val[:, f]
                    emb_f = E[f, vals]
                    sum_emb += emb_f
                    sum_emb_sq += emb_f * emb_f
                    linear += W[f, vals]
                interaction = 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=1)
                logits = linear + interaction
                preds = (logits > 0).astype(np.int64)
                acc = (preds == y_val).mean()
                if acc > best_acc:
                    best_acc = acc
                    best_params = (k, lr, l2)
                    best_state = (E.copy(), W.copy())

    # retrain on full data with best params
    k, lr, l2 = best_params
    E, W = best_state
    rng = np.random.default_rng(SEED)
    mE, vE = np.zeros_like(E), np.zeros_like(E)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for _ in range(FINAL_EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X[idx], y[idx]
            B = len(yb)
            sum_emb = np.zeros((B, k), np.float32)
            sum_emb_sq = np.zeros((B, k), np.float32)
            linear = np.zeros(B, np.float32)
            for f in range(d):
                vals = xb[:, f]
                emb_f = E[f, vals]
                sum_emb += emb_f
                sum_emb_sq += emb_f * emb_f
                linear += W[f, vals]
            interaction = 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=1)
            logits = linear + interaction
            probs = 1.0 / (1.0 + np.exp(-logits))
            dlogits = probs - yb

            # compute gradients
            grad_W = np.zeros_like(W)
            grad_E = np.zeros_like(E)
            for f in range(d):
                vals = xb[:, f]
                np.add.at(grad_W, (f, vals), dlogits / B)
                grad_emb = (sum_emb - E[f, vals]) * dlogits[:, None]
                np.add.at(grad_E, (f, vals), grad_emb / B)

            # L2
            grad_W += l2 * W
            grad_E += l2 * E

            # Adam
            t += 1
            mW = beta1 * mW + (1 - beta1) * grad_W
            vW = beta2 * vW + (1 - beta2) * grad_W * grad_W
            mE = beta1 * mE + (1 - beta1) * grad_E
            vE = beta2 * vE + (1 - beta2) * grad_E * grad_E
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            mE_hat = mE / (1 - beta1 ** t)
            vE_hat = vE / (1 - beta2 ** t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            E -= lr * mE_hat / (np.sqrt(vE_hat) + eps)

    # write weights: magic, d, k, C, then W (d*256 floats), then E (d*256*k floats)
    magic = 0x464D4C31  # "FML1"
    out = struct.pack("<4i", magic, d, k, C)
    out += W.astype("<f4").tobytes()
    out += E.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d != 14 || k <= 0 || k > 64) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16L + 4L * d * 256 + 4L * d * 256 * k;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W = (const float *)(hd + 4);
    const float *E = W + d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * k);
    float *sum_emb_sq = malloc(sizeof(float) * k);

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        float linear = 0.0f;
        for (int j = 0; j < k; j++) { sum_emb[j] = 0.0f; sum_emb_sq[j] = 0.0f; }

        for (int f = 0; f < d; f++) {
            uint8_t v = xi[f];
            linear += W[f * 256 + v];
            const float *emb = E + (f * 256 + v) * k;
            for (int j = 0; j < k; j++) {
                float e = emb[j];
                sum_emb[j] += e;
                sum_emb_sq[j] += e * e;
            }
        }
        float interaction = 0.0f;
        for (int j = 0; j < k; j++) interaction += sum_emb[j] * sum_emb[j] - sum_emb_sq[j];
        interaction *= 0.5f;

        float score = linear + interaction;
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```