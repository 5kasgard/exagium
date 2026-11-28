```hypothesis
Family: label-elm (Extreme Learning Machine). A single hidden layer with 128 random ReLU units (fixed seed) and ridge-regressed output weights. The random features capture nonlinearities that linear models miss, while ridge regression avoids iterative training, ensuring determinism and speed. Uses the card's tuning split (last 20% of train) to select λ from {1e-4,1e-3,1e-2,1e-1,1,10,100}, then retrains on all data. Inference cost matches label-mlp16 (two-layer table lookup). Failure: dev accuracy ≤ 0.961 (label-linear) or inference Ir/item ≥ label-mlp16.
```
```python
"""label-elm: Extreme Learning Machine with random hidden layer and ridge output.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes ordinal bytes as z-scored scalars (like label-linear).
Random hidden layer: H=128, W1~N(0, sqrt(2/d)), b1~N(0,1) with seed 42.
Output weights: ridge regression on one-hot labels with λ tuned on last 20% of train.
Exported as per-byte lookup tables: magic 0x454C4D31, d, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

H = 128
SEED_DATA, SEED_ELM, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 42, 3e-3, 1e-4, 64, 600000, 100
LAMBDA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


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

    # Encode features: per-byte z-scoring for ordinal (all 64 are ordinal)
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals))); E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)  # (n, d)
    k = Z.shape[1]
    assert k == d

    # Fixed random hidden layer (ELM)
    rng_elm = np.random.default_rng(SEED_ELM)
    W1 = (rng_elm.normal(0, math.sqrt(2.0 / d), (d, H))).astype(np.float32)
    b1 = rng_elm.normal(0, 1.0, H).astype(np.float32)

    # Hidden activations: ReLU(Z @ W1 + b1)
    H_act = np.maximum(Z @ W1 + b1, 0, dtype=np.float32)  # (n, H)

    # One-hot labels
    Y = np.eye(C, dtype=np.float32)[y]  # (n, C)

    # Tuning split: last 20% in file order
    n_tune = n // 5
    n_train = n - n_tune
    H_train, H_tune = H_act[:n_train], H_act[n_train:]
    Y_train, Y_tune = Y[:n_train], Y[n_train:]

    # Augment with bias column for output bias
    H_train_aug = np.hstack([H_train, np.ones((n_train, 1), dtype=np.float32)])  # (n_train, H+1)
    H_tune_aug = np.hstack([H_tune, np.ones((n_tune, 1), dtype=np.float32)])    # (n_tune, H+1)

    # Precompute A = H_aug^T H_aug, B = H_aug^T Y for train
    A = H_train_aug.T @ H_train_aug  # (H+1, H+1)
    B = H_train_aug.T @ Y_train      # (H+1, C)

    # Ridge regularization matrix: λ on W2 weights, 0 on bias
    reg_diag = np.ones(H + 1, dtype=np.float32)
    reg_diag[H] = 0.0  # no regularization on bias term

    best_lam, best_acc = LAMBDA_GRID[0], -1.0
    for lam in LAMBDA_GRID:
        A_reg = A.copy()
        A_reg += lam * np.diag(reg_diag)
        try:
            W_aug = np.linalg.solve(A_reg, B)  # (H+1, C)
        except np.linalg.LinAlgError:
            continue
        W2_cand = W_aug[:H]
        b2_cand = W_aug[H]
        # Predict on tune
        scores_tune = H_tune @ W2_cand + b2_cand
        pred_tune = scores_tune.argmax(axis=1)
        acc = (pred_tune == y[n_train:]).mean()
        if acc > best_acc:
            best_acc, best_lam = acc, lam

    # Retrain on all data with best_lam
    H_all_aug = np.hstack([H_act, np.ones((n, 1), dtype=np.float32)])
    A_all = H_all_aug.T @ H_all_aug
    B_all = H_all_aug.T @ Y
    A_all += best_lam * np.diag(reg_diag)
    W_aug = np.linalg.solve(A_all, B_all)
    W2 = W_aug[:H].astype(np.float32)
    b2 = W_aug[H].astype(np.float32)

    # Build per-byte lookup tables T[j][v][h] = enc[j][v] @ W1[j, h] (summed over input dims per byte)
    # Since each ordinal feature encodes to 1 dim, T[j] is (256, H)
    off, T_list = 0, []
    for E in enc:
        # E is (256, 1), W1 slice is (1, H) -> result (256, H)
        T_list.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    T = np.stack(T_list, axis=0)  # (d, 256, H)

    # Write weights.bin
    magic = 0x454C4D31  # 'ELM1' little-endian
    out = struct.pack("<4i", magic, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-elm inference: ./infer weights.bin in.rec N
 * Input record: d bytes (64). Output: 1 byte class index (argmax).
 * Weights format: int32 magic(0x454C4D31), d, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 16 || hd[0] != 0x454C4D31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * H + H + (long)H * C + C)) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int h = 0; h < H; h++) acc[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) acc[h] += t[h];
        }
        // ReLU activation
        for (int h = 0; h < H; h++) if (acc[h] < 0.0f) acc[h] = 0.0f;
        // Output layer
        for (int c = 0; c < C; c++) scores[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float ah = acc[h];
            if (ah == 0.0f) continue;
            const float *w2h = W2 + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += ah * w2h[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```