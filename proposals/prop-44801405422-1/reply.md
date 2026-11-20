```hypothesis
Family: label-rff (Random Fourier Features). It approximates an RBF kernel by mapping inputs to cos(Wx + b) with random W,b, then trains a linear classifier. On optdigits, RBF SVMs achieve >98% accuracy, so this should beat label-linear (96.1%) while keeping inference fast (one matrix multiply + cos). Failure: if dev accuracy ≤ label-linear or inference cost exceeds label-mlp16.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 12345
D_CANDIDATES = [128, 256, 512, 1024]
GAMMA_CANDIDATES = [0.1, 0.5, 1.0, 2.0]
LR, L2, EPOCHS, BATCH = 0.05, 1e-4, 50, 64
FINAL_EPOCHS = 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d, C = X_all.shape[0], card["input"]["bytes"], card["classes"]
    split = int(0.8 * n)
    X_train, X_tune = X_all[:split], X_all[split:]
    y_train, y_tune = y_all[:split], y_all[split:]

    mean = X_train.mean(axis=0).astype(np.float32)
    std = X_train.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    X_train_norm = (X_train - mean) / std
    X_tune_norm = (X_tune - mean) / std

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_D = D_CANDIDATES[0]
    best_gamma = GAMMA_CANDIDATES[0]
    best_W_rff = best_b_rff = best_W_out = best_b_out = None

    for D in D_CANDIDATES:
        for gamma in GAMMA_CANDIDATES:
            W_rff = rng.normal(0, math.sqrt(2*gamma), size=(D, d)).astype(np.float32)
            b_rff = rng.uniform(0, 2*math.pi, size=D).astype(np.float32)
            phi_train = np.cos(X_train_norm @ W_rff.T + b_rff)
            phi_tune = np.cos(X_tune_norm @ W_rff.T + b_rff)

            W_out = np.zeros((C, D), dtype=np.float32)
            b_out = np.zeros(C, dtype=np.float32)
            n_train = phi_train.shape[0]
            for _ in range(EPOCHS):
                idx = rng.permutation(n_train)
                for i in range(0, n_train, BATCH):
                    batch_idx = idx[i:i+BATCH]
                    Xb = phi_train[batch_idx]
                    yb = y_train[batch_idx]
                    scores = Xb @ W_out.T + b_out
                    scores -= scores.max(axis=1, keepdims=True)
                    exp_scores = np.exp(scores)
                    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
                    probs[np.arange(len(yb)), yb] -= 1.0
                    grad_W = (probs.T @ Xb) / len(yb) + L2 * W_out
                    grad_b = probs.mean(axis=0)
                    W_out -= LR * grad_W
                    b_out -= LR * grad_b

            scores_tune = phi_tune @ W_out.T + b_out
            acc = (scores_tune.argmax(axis=1) == y_tune).mean()
            if acc > best_acc:
                best_acc = acc
                best_D = D
                best_gamma = gamma
                best_W_rff = W_rff.copy()
                best_b_rff = b_rff.copy()
                best_W_out = W_out.copy()
                best_b_out = b_out.copy()

    # Retrain on full data
    X_all_norm = (X_all - mean) / std
    phi_all = np.cos(X_all_norm @ best_W_rff.T + best_b_rff)
    W_out = np.zeros((C, best_D), dtype=np.float32)
    b_out = np.zeros(C, dtype=np.float32)
    for _ in range(FINAL_EPOCHS):
        idx = rng.permutation(n)
        for i in range(0, n, BATCH):
            batch_idx = idx[i:i+BATCH]
            Xb = phi_all[batch_idx]
            yb = y_all[batch_idx]
            scores = Xb @ W_out.T + b_out
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_W = (probs.T @ Xb) / len(yb) + L2 * W_out
            grad_b = probs.mean(axis=0)
            W_out -= LR * grad_W
            b_out -= LR * grad_b

    magic = 0x52464631  # "RFF1"
    out = struct.pack("<4i", magic, d, best_D, C)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += best_W_rff.astype("<f4").tobytes()
    out += best_b_rff.astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    out += b_out.astype("<f4").tobytes()
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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x52464631) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 64 || C != 10 || D <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16L + 4L*d + 4L*d + 4L*D*d + 4L*D + 4L*C*D + 4L*C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    float *mean = (float *)(hd + 4);
    float *std = mean + d;
    float *W_rff = std + d;
    float *b_rff = W_rff + (long)D * d;
    float *W_out = b_rff + D;
    float *b_out = W_out + (long)C * D;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_norm = malloc(d * sizeof(float));
    float *phi = malloc(D * sizeof(float));
    float *scores = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) x_norm[j] = (xi[j] - mean[j]) / std[j];
        for (int u = 0; u < D; u++) {
            float sum = b_rff[u];
            const float *w = W_rff + (long)u * d;
            for (int j = 0; j < d; j++) sum += w[j] * x_norm[j];
            phi[u] = cosf(sum);
        }
        for (int c = 0; c < C; c++) {
            float sum = b_out[c];
            const float *w = W_out + (long)c * D;
            for (int u = 0; u < D; u++) sum += w[u] * phi[u];
            scores[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```