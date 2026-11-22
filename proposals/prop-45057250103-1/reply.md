```hypothesis
The label-rbf family uses a Radial Basis Function network: centers are a fixed random subset of training points, Gaussian RBF features are computed, and a linear logistic regression is trained on top. This adds non-linearity to capture local pixel correlations in optdigits while keeping inference fast (distance to centers + linear layer). We expect it to beat label-linear (96.13%) and approach label-mlp16 (97.70%) at lower cost than label-gbdt. If dev accuracy does not exceed 96.5%, the family does not justify its added complexity.
```
```python
import json, math, struct, sys
import numpy as np

N_CENTERS_LIST = [20, 30, 40, 50, 60]
GAMMA_LIST = [0.1, 0.5, 1.0, 2.0]

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def rbf_features(X, centers, gamma):
    # X: (n, d), centers: (k, d) -> (n, k)
    diff = X[:, None, :] - centers[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    return np.exp(-gamma * dist2).astype(np.float32)


def train_logistic(Z, y, C, seed):
    n, k = Z.shape
    rng = np.random.default_rng(seed)
    W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            S = z @ W + b
            S = S - S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            grads = [z.T @ g + L2 * W, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return W, b


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    n = X.shape[0]
    split = int(n * 0.8)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]

    best_acc = -1.0
    best_params = None
    best_centers = None
    best_W = None
    best_b = None

    for i, n_centers in enumerate(N_CENTERS_LIST):
        for j, gamma in enumerate(GAMMA_LIST):
            combo_seed = SEED * 1000 + i * 100 + j
            rng = np.random.default_rng(combo_seed)
            center_idx = rng.choice(len(X_train), n_centers, replace=False)
            centers = X_train[center_idx].astype(np.float32)
            Z_train = rbf_features(X_train.astype(np.float32), centers, gamma)
            Z_tune = rbf_features(X_tune.astype(np.float32), centers, gamma)
            W, b = train_logistic(Z_train, y_train, C, combo_seed + 10000)
            S = Z_tune @ W + b
            preds = np.argmax(S, axis=1)
            acc = (preds == y_tune).mean()
            if acc > best_acc:
                best_acc = acc
                best_params = (n_centers, gamma)
                best_centers = centers
                best_W = W
                best_b = b
                print(f"New best: n_centers={n_centers}, gamma={gamma}, acc={acc:.4f}", file=sys.stderr)

    n_centers, gamma = best_params
    rng = np.random.default_rng(SEED)
    center_idx = rng.choice(n, n_centers, replace=False)
    centers = X[center_idx].astype(np.float32)
    Z_full = rbf_features(X.astype(np.float32), centers, gamma)
    W, b = train_logistic(Z_full, y, C, SEED + 20000)

    magic = 0x52424631  # RBF1
    out = struct.pack("<4i f", magic, d, n_centers, C, gamma)
    out += centers.astype("<f4").tobytes()
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    uint32_t magic = *(uint32_t*)wb;
    if (magic != 0x52424631) { fprintf(stderr, "bad weights: magic mismatch\n"); return 3; }
    int32_t *hd = (int32_t*)(wb + 4);
    int d = hd[0];
    int n_centers = hd[1];
    int C = hd[2];
    float gamma = *(float*)(wb + 4 + 12);
    float *centers = (float*)(wb + 20);
    float *W = centers + (long)n_centers * d;
    float *b = W + (long)n_centers * C;
    long expected = 20 + (long)n_centers * d * 4 + (long)n_centers * C * 4 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }
    if (C < 1 || C > 256) { fprintf(stderr, "bad weights: C out of range\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *phi = malloc(sizeof(float) * n_centers);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < n_centers; j++) {
            float dist2 = 0.0f;
            const float *c = centers + (long)j * d;
            for (int f = 0; f < d; f++) {
                float diff = (float)xi[f] - c[f];
                dist2 += diff * diff;
            }
            phi[j] = expf(-gamma * dist2);
        }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < n_centers; j++) {
            float p = phi[j];
            const float *w = W + (long)j * C;
            for (int c = 0; c < C; c++) scores[c] += p * w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```