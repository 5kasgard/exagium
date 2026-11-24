```hypothesis
label-rbf: RBF network with class-stratified centers (5 per class = 50 total). Ordinal features are z-scored per pixel; centers are selected from training data (fixed seed). Gamma tuned on last 20% of train, then model retrained on all data. Output layer is logistic regression trained with Adam. Expected >0.965 dev accuracy; if not, the 50-center RBF kernel lacks capacity for this task.
```
```python
"""label-rbf: RBF network with stratified centers + logistic output.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
CENTERS_PER_CLASS = 5
GAMMA_CANDIDATES = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def rbf_features(X, centers, gamma, mean, std):
    """X: (n, d) uint8, centers: (K, d) uint8 -> (n, K) float32"""
    n, d = X.shape
    K = centers.shape[0]
    Xn = (X.astype(np.float32) - mean) / std          # (n, d)
    Cn = (centers.astype(np.float32) - mean) / std    # (K, d)
    # squared Euclidean distance via broadcasting
    diff = Xn[:, None, :] - Cn[None, :, :]            # (n, K, d)
    dist2 = np.sum(diff * diff, axis=2)               # (n, K)
    return np.exp(-gamma * dist2, dtype=np.float32)


def train_logistic(Z, y, C, rng):
    """Z: (n, K), y: (n,), returns W:(K,C), b:(C,)"""
    n, K = Z.shape
    W = (rng.uniform(-1, 1, (K, C)) * math.sqrt(6.0 / K)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            logits = z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            P = np.exp(logits); P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            grads = [z.T @ g + L2 * W, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr; v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return W, b


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    n = X.shape[0]

    # per-feature z-score stats
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0

    # split: last 20% for tuning
    split = int(n * 0.8)
    idx_all = np.arange(n)
    idx_train, idx_tune = idx_all[:split], idx_all[split:]

    rng = np.random.default_rng(SEED)

    # select centers from training portion for tuning
    centers_list = []
    for c in range(C):
        class_idx = idx_train[y[idx_train] == c]
        if len(class_idx) == 0:
            continue
        take = min(CENTERS_PER_CLASS, len(class_idx))
        chosen = rng.choice(class_idx, size=take, replace=False)
        centers_list.append(X[chosen])
    centers_train = np.vstack(centers_list) if centers_list else np.empty((0, d), np.uint8)
    K = centers_train.shape[0]

    best_gamma, best_acc = GAMMA_CANDIDATES[0], -1.0
    for gamma in GAMMA_CANDIDATES:
        Ztr = rbf_features(X[idx_train], centers_train, gamma, mean, std)
        Ztu = rbf_features(X[idx_tune], centers_train, gamma, mean, std)
        W, b = train_logistic(Ztr, y[idx_train], C, np.random.default_rng(SEED))
        pred = np.argmax(Ztu @ W + b, axis=1)
        acc = (pred == y[idx_tune]).mean()
        if acc > best_acc:
            best_acc, best_gamma = acc, gamma

    # final: centers from ALL data
    centers_list = []
    for c in range(C):
        class_idx = np.where(y == c)[0]
        take = min(CENTERS_PER_CLASS, len(class_idx))
        chosen = rng.choice(class_idx, size=take, replace=False)
        centers_list.append(X[chosen])
    centers = np.vstack(centers_list) if centers_list else np.empty((0, d), np.uint8)
    K = centers.shape[0]

    Z = rbf_features(X, centers, best_gamma, mean, std)
    W, b = train_logistic(Z, y, C, np.random.default_rng(SEED))

    # export: magic, d, K, C, gamma, mean[d], std[d], centers[K*d] (uint8), W[K*C], b[C]
    magic = 0x52424631  # 'RBF1'
    hdr = struct.pack("<5i f", magic, d, K, C, C, best_gamma)  # 5 ints + float
    # Actually need: magic, d, K, C, gamma (float), then mean, std, centers, W, b
    # Let's pack carefully
    out = struct.pack("<4i f", magic, d, K, C, best_gamma)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += centers.astype(np.uint8).tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-rbf inference: ./infer weights.bin in.rec N
 * Input record: d bytes (uint8). Output: 1 byte class index. */
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
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }

    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x52424631) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = (int32_t)hd[1], K = (int32_t)hd[2], C = (int32_t)hd[3];
    float gamma = *(float *)(hd + 4);
    const float *mean = (const float *)(hd + 5);
    const float *std  = mean + d;
    const unsigned char *centers = (const unsigned char *)(std + d);
    const float *W = (const float *)(centers + (long)K * d);
    const float *b = W + (long)K * C;

    long expected = 20 + (long)d * 4 * 2 + (long)K * d + (long)K * C * 4 + (long)C * 4;
    if (nw != expected || C < 1 || C > 256 || K < 1) {
        fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3;
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *rbf = malloc(sizeof(float) * K);
    float *score = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // compute RBF features
        for (int cidx = 0; cidx < K; cidx++) {
            const unsigned char *cc = centers + (long)cidx * d;
            float dist2 = 0.0f;
            for (int j = 0; j < d; j++) {
                float diff = (xi[j] - cc[j]) / std[j];
                dist2 += diff * diff;
            }
            rbf[cidx] = expf(-gamma * dist2);
        }

        // linear output layer
        memcpy(score, b, sizeof(float) * C);
        for (int cidx = 0; cidx < K; cidx++) {
            float rv = rbf[cidx];
            const float *wrow = W + (long)cidx * C;
            for (int c = 0; c < C; c++) score[c] += rv * wrow[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```