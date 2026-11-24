```hypothesis
Family: label-elm (Extreme Learning Machine). A single hidden layer with random fixed weights (deterministic seed) and ridge-regressed output weights. The 8x8 optdigits pixels have non-linear structure; a random projection + ReLU + linear readout should capture it better than pure linear at similar inference cost (one extra matrix multiply). If dev accuracy does not exceed label-linear's 0.9613, the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np

# hyperparameter grid (chosen on tuning split)
H_CHOICES = [16, 32, 64, 128]
LAMBDA_CHOICES = [1e-4, 1e-3, 1e-2]
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
    C = card["classes"]
    d = X.shape[1]
    n = X.shape[0]

    # split: last 20% for tuning
    n_tune = n // 5
    idx_train = slice(0, n - n_tune)
    idx_tune = slice(n - n_tune, n)

    X_tr, y_tr = X[idx_train], y[idx_train]
    X_tu, y_tu = X[idx_tune], y[idx_tune]

    best_acc = -1.0
    best_H = H_CHOICES[0]
    best_lambda = LAMBDA_CHOICES[0]
    best_W1_scaled = None
    best_b1_scaled = None
    best_W2 = None
    best_b2 = None

    for H in H_CHOICES:
        # deterministic random weights for this H
        rng = np.random.default_rng(SEED + H * 1000)
        # He initialization for ReLU
        W1 = (rng.uniform(-1, 1, (d, H)) * math.sqrt(6.0 / d)).astype(np.float64)
        b1 = rng.uniform(-1, 1, H).astype(np.float64)

        for lam in LAMBDA_CHOICES:
            # standardize on training portion
            mu = X_tr.mean(axis=0).astype(np.float64)
            sd = X_tr.std(axis=0).astype(np.float64)
            sd[sd == 0] = 1.0

            # fold standardization into first layer
            W1_scaled = W1 / sd[:, np.newaxis]          # (d, H)
            b1_scaled = b1 - mu @ W1_scaled              # (H,)

            # hidden activations for training set
            Z_tr = (X_tr.astype(np.float64) @ W1_scaled) + b1_scaled
            H_tr = np.maximum(Z_tr, 0)                   # ReLU

            # add bias column
            H_tr_b = np.hstack([H_tr, np.ones((H_tr.shape[0], 1), dtype=np.float64)])  # (N_tr, H+1)
            Y_tr = np.eye(C, dtype=np.float64)[y_tr]     # (N_tr, C)

            # ridge regression: (H^T H + lambda I) W = H^T Y, no regularization on bias
            A = H_tr_b.T @ H_tr_b
            reg = lam * np.eye(H + 1, dtype=np.float64)
            reg[-1, -1] = 0.0
            B = H_tr_b.T @ Y_tr
            W_combined = np.linalg.solve(A + reg, B)     # (H+1, C)
            W2 = W_combined[:H].astype(np.float32)
            b2 = W_combined[H].astype(np.float32)

            # evaluate on tuning set
            Z_tu = (X_tu.astype(np.float64) @ W1_scaled) + b1_scaled
            H_tu = np.maximum(Z_tu, 0)
            scores = H_tu @ W2 + b2
            pred = scores.argmax(axis=1)
            acc = (pred == y_tu).mean()

            if acc > best_acc:
                best_acc = acc
                best_H = H
                best_lambda = lam
                best_W1_scaled = W1_scaled.astype(np.float32)
                best_b1_scaled = b1_scaled.astype(np.float32)
                best_W2 = W2
                best_b2 = b2

    # retrain on full training set with best hyperparameters
    rng = np.random.default_rng(SEED + best_H * 1000)
    W1 = (rng.uniform(-1, 1, (d, best_H)) * math.sqrt(6.0 / d)).astype(np.float64)
    b1 = rng.uniform(-1, 1, best_H).astype(np.float64)

    mu = X.mean(axis=0).astype(np.float64)
    sd = X.std(axis=0).astype(np.float64)
    sd[sd == 0] = 1.0

    W1_scaled = W1 / sd[:, np.newaxis]
    b1_scaled = b1 - mu @ W1_scaled

    Z = (X.astype(np.float64) @ W1_scaled) + b1_scaled
    H_full = np.maximum(Z, 0)
    H_full_b = np.hstack([H_full, np.ones((n, 1), dtype=np.float64)])
    Y = np.eye(C, dtype=np.float64)[y]

    A = H_full_b.T @ H_full_b
    reg = best_lambda * np.eye(best_H + 1, dtype=np.float64)
    reg[-1, -1] = 0.0
    B = H_full_b.T @ Y
    W_combined = np.linalg.solve(A + reg, B)
    W2 = W_combined[:best_H].astype(np.float32)
    b2 = W_combined[best_H].astype(np.float32)

    # write weights.bin
    # magic 0x454C4D01 ('ELM\x01'), d, H, C
    header = struct.pack("<4i", 0x454C4D01, d, best_H, C)
    # W1_scaled: d * H floats (row-major: j*H + h)
    # b1_scaled: H floats
    # W2: H * C floats (row-major: h*C + c)
    # b2: C floats
    out = header
    out += best_W1_scaled.astype("<f4").tobytes()
    out += best_b1_scaled.astype("<f4").tobytes()
    out += best_W2.astype("<f4").tobytes()
    out += best_b2.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-elm inference: ./infer weights.bin in.rec N */
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
    if (nw < 16 || hd[0] != 0x454C4D01) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }
    long expected = 16L + 4L * ((long)d * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W1 = (const float *)(hd + 4);                 // d * H
    const float *b1 = W1 + (long)d * H;                        // H
    const float *W2 = b1 + H;                                  // H * C
    const float *b2 = W2 + (long)H * C;                        // C

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = (float *)malloc(sizeof(float) * H);
    float *scores = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* hidden = ReLU( x @ W1 + b1 ) */
        for (int h = 0; h < H; h++) hidden[h] = b1[h];
        for (int j = 0; j < d; j++) {
            float xj = (float)xi[j];
            const float *w = W1 + (long)j * H;
            for (int h = 0; h < H; h++) hidden[h] += xj * w[h];
        }
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;

        /* scores = hidden @ W2 + b2 */
        for (int c = 0; c < C; c++) scores[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float hh = hidden[h];
            const float *w = W2 + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += hh * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```