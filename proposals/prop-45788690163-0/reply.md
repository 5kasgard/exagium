```hypothesis
Family: label-boosted-stumps (gradient-boosted depth-1 trees on 14 target-encoded features).
Why: Adult has 14 mixed ordinal/categorical features; target encoding gives strong 1D signals, and boosted stumps capture non-linearities/interactions with minimal parameters. Inference is 14 table lookups + M threshold checks (M≈100), far cheaper than GBDT (7570 Ir/item) and more expressive than linear (0.8524).
Failure: dev accuracy < 0.853 or Ir/item > 2000.
```
```python
"""label-boosted-stumps: gradient-boosted decision stumps on target-encoded features.
train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, stdlib+numpy only.
Encodes each of 14 bytes to a float: ordinal -> z-score, categorical -> target mean (train split only).
Boosts M stumps (depth-1 trees) with logistic loss, lr=0.1. Chooses M on last 20% tuning split.
Weights format: magic 0x53545542, d=14, M, init_bias, enc[14][256] (float32), then M stumps:
  feat (uint8), thresh (float32), left (float32), right (float32)."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 0.1
MAX_STUMPS = 200
CANDIDATE_MS = [25, 50, 75, 100, 125, 150, 175, 200]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def fit_stump(X_col, resid, rng):
    """Fit best depth-1 stump on one column to residuals (squared error)."""
    n = len(X_col)
    order = np.argsort(X_col, kind="mergesort")
    x_sorted = X_col[order]
    r_sorted = resid[order]

    # prefix sums for O(1) split evaluation
    pref_r = np.concatenate(([0.0], np.cumsum(r_sorted)))
    pref_r2 = np.concatenate(([0.0], np.cumsum(r_sorted * r_sorted)))
    total_r = pref_r[-1]
    total_r2 = pref_r2[-1]

    best_gain = -1.0
    best_thresh = 0.0
    best_left = 0.0
    best_right = 0.0

    for i in range(1, n):
        if x_sorted[i] == x_sorted[i - 1]:
            continue
        n_l, n_r = i, n - i
        sum_l, sum_r = pref_r[i], total_r - pref_r[i]
        sum2_l, sum2_r = pref_r2[i], total_r2 - pref_r2[i]
        # MSE reduction = total_var - (var_l + var_r)
        # var = sum2 - sum^2/n
        var_total = total_r2 - total_r * total_r / n
        var_l = sum2_l - sum_l * sum_l / n_l
        var_r = sum2_r - sum_r * sum_r / n_r
        gain = var_total - (var_l + var_r)
        if gain > best_gain:
            best_gain = gain
            best_thresh = (x_sorted[i - 1] + x_sorted[i]) * 0.5
            best_left = sum_l / n_l
            best_right = sum_r / n_r

    if best_gain <= 0:
        return None, 0.0, 0.0, 0.0
    return best_thresh, best_left, best_right, best_gain


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def log_loss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    assert d == 14

    # load all train data
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.float32)
    n_all = len(y_all)

    # split: last 20% for tuning
    n_tune = n_all // 5
    n_train = n_all - n_tune
    X_train, X_tune = X_all[:n_train], X_all[n_train:]
    y_train, y_tune = y_all[:n_train], y_all[n_train:]

    rng = np.random.default_rng(SEED)

    # compute encodings on train split only
    enc = np.zeros((d, 256), dtype=np.float32)
    for j, t in enumerate(types):
        col = X_train[:, j]
        if t == "ordinal":
            mu = col.mean()
            sd = col.std()
            if sd == 0:
                sd = 1.0
            enc[j] = (np.arange(256, dtype=np.float32) - mu) / sd
        else:
            # target mean encoding
            global_mean = y_train.mean()
            for v in range(256):
                mask = col == v
                if mask.any():
                    enc[j, v] = y_train[mask].mean()
                else:
                    enc[j, v] = global_mean

    # transform
    Z_train = enc[np.arange(d), X_train.T].T  # (n_train, d)
    Z_tune = enc[np.arange(d), X_tune.T].T

    # initial bias = log-odds
    init_bias = math.log(y_train.mean() / (1 - y_train.mean() + 1e-12))

    # find best M on tuning set
    best_M = CANDIDATE_MS[0]
    best_loss = float("inf")
    best_stumps = None

    for M in CANDIDATE_MS:
        stumps = []  # each: (feat, thresh, left, right)
        # train on train split
        pred = np.full(n_train, init_bias, dtype=np.float32)
        for m in range(M):
            p = sigmoid(pred)
            resid = (y_train - p).astype(np.float32)  # logistic gradient
            best_gain = -1.0
            best_feat = 0
            best_thresh = 0.0
            best_left = 0.0
            best_right = 0.0
            for j in range(d):
                thresh, left, right, gain = fit_stump(Z_train[:, j], resid, rng)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = j
                    best_thresh = thresh
                    best_left = left
                    best_right = right
            if best_gain <= 0:
                break
            stumps.append((best_feat, best_thresh, best_left, best_right))
            # update predictions
            col = Z_train[:, best_feat]
            pred += LR * np.where(col < best_thresh, best_left, best_right)

        # evaluate on tuning
        pred_tune = np.full(n_tune, init_bias, dtype=np.float32)
        for feat, thresh, left, right in stumps:
            col = Z_tune[:, feat]
            pred_tune += LR * np.where(col < thresh, left, right)
        p_tune = sigmoid(pred_tune)
        loss = log_loss(y_tune, p_tune)
        if loss < best_loss:
            best_loss = loss
            best_M = M
            best_stumps = stumps

    # retrain on all data with best_M
    # recompute encodings on all data
    for j, t in enumerate(types):
        col = X_all[:, j]
        if t == "ordinal":
            mu = col.mean()
            sd = col.std()
            if sd == 0:
                sd = 1.0
            enc[j] = (np.arange(256, dtype=np.float32) - mu) / sd
        else:
            global_mean = y_all.mean()
            for v in range(256):
                mask = col == v
                if mask.any():
                    enc[j, v] = y_all[mask].mean()
                else:
                    enc[j, v] = global_mean

    Z_all = enc[np.arange(d), X_all.T].T

    init_bias = math.log(y_all.mean() / (1 - y_all.mean() + 1e-12))
    stumps = []
    pred = np.full(n_all, init_bias, dtype=np.float32)
    for m in range(best_M):
        p = sigmoid(pred)
        resid = (y_all - p).astype(np.float32)
        best_gain = -1.0
        best_feat = 0
        best_thresh = 0.0
        best_left = 0.0
        best_right = 0.0
        for j in range(d):
            thresh, left, right, gain = fit_stump(Z_all[:, j], resid, rng)
            if gain > best_gain:
                best_gain = gain
                best_feat = j
                best_thresh = thresh
                best_left = left
                best_right = right
        if best_gain <= 0:
            # pad remaining stumps with no-op
            stumps.append((0, 0.0, 0.0, 0.0))
            continue
        stumps.append((best_feat, best_thresh, best_left, best_right))
        col = Z_all[:, best_feat]
        pred += LR * np.where(col < best_thresh, best_left, best_right)

    # ensure exactly best_M stumps
    while len(stumps) < best_M:
        stumps.append((0, 0.0, 0.0, 0.0))
    stumps = stumps[:best_M]

    # write weights.bin
    # magic, d, M, init_bias, enc[d][256], then M*(feat, thresh, left, right)
    magic = 0x53545542  # 'STUB'
    header = struct.pack("<3if", magic, d, best_M, init_bias)
    enc_bytes = enc.astype("<f4").tobytes()
    stump_bytes = b""
    for feat, thresh, left, right in stumps:
        stump_bytes += struct.pack("<Bfff", feat, thresh, left, right)

    with open(sys.argv[4], "wb") as f:
        f.write(header + enc_bytes + stump_bytes)


if __name__ == "__main__":
    main()
```
```c
/* label-boosted-stumps infer: ./infer weights.bin in.rec N
 * Input record: 14 bytes. Output: 1 byte class (0/1).
 * Weights: magic 0x53545542, d=14, M, init_bias, enc[14][256] (float32),
 *   then M stumps: feat(uint8), thresh(float32), left(float32), right(float32). */
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    if (magic != 0x53545542) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = wb[4] | wb[5] << 8 | wb[6] << 16 | wb[7] << 24;
    int M = wb[8] | wb[9] << 8 | wb[10] << 16 | wb[11] << 24;
    float init_bias = *(float*)(wb + 12);
    if (d != 14 || M < 0 || M > 1000) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16 + 14L * 256 * 4 + M * (1 + 3 * 4);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *enc = (const float*)(wb + 16);
    const unsigned char *stumps = wb + 16 + 14 * 256 * 4;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float z[14];
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 14) { fprintf(stderr, "item %ld is %u bytes, not 14\n", i, k); return 3; }
        for (int j = 0; j < 14; j++) z[j] = enc[j * 256 + xi[j]];
        float score = init_bias;
        const unsigned char *sp = stumps;
        for (int m = 0; m < M; m++) {
            int feat = sp[0];
            float thresh = *(float*)(sp + 1);
            float left = *(float*)(sp + 5);
            float right = *(float*)(sp + 9);
            score += (z[feat] < thresh) ? left : right;
            sp += 13;
        }
        out_byte = (1.0f / (1.0f + expf(-score)) > 0.5f) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```