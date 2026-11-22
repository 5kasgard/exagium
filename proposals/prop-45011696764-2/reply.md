```hypothesis
We propose label-lowrank: a factorized linear model where the per-position one-hot weight matrix W (d*256 x C) is decomposed as U (d x 256 x R) and V (R x C) with rank R=32. This captures pairwise interactions between context positions implicitly while reducing parameters (from ~1M to ~130k floats) and compute (2*d*R + R*C vs d*256*C). The factorization acts as a strong regularizer suited for the small dataset, and the embedding-like structure models n-gram statistics. If dev accuracy does not exceed label-linear (0.3375), the rank may be too low or the optimization may need more epochs/tuning.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameter grid (chosen on tuning split)
R_LIST = [16, 32, 64]
LR_LIST = [0.1, 0.05, 0.01]
EPOCHS_LIST = [10, 20]
BATCH_SIZE = 256
SEED = 1
L2 = 1e-5

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, R, lr, epochs, batch_size, seed):
    n, d = X.shape
    C = 65
    rng = np.random.default_rng(seed)
    scale = math.sqrt(6.0 / (256 + R))
    U = rng.uniform(-scale, scale, (d, 256, R)).astype(np.float32)
    V = rng.uniform(-scale, scale, (R, C)).astype(np.float32)
    bias = np.zeros(C, np.float32)
    mU = np.zeros_like(U); vU = np.zeros_like(U)
    mV = np.zeros_like(V); vV = np.zeros_like(V)
    mb = np.zeros_like(bias); vb = np.zeros_like(bias)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s+batch_size]
            xb = X[idx]
            yb = y[idx]
            b = xb.shape[0]
            h = np.zeros((b, R), np.float32)
            for j in range(d):
                h += U[j, xb[:, j], :]
            logits = h @ V + bias
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            dlogits = probs
            dlogits[np.arange(b), yb] -= 1.0
            dlogits /= b
            dbias = dlogits.sum(axis=0)
            dV = h.T @ dlogits + L2 * V
            dh = dlogits @ V.T
            dU = np.zeros_like(U)
            for j in range(d):
                np.add.at(dU[j], xb[:, j], dh)
            dU += L2 * U
            t += 1
            for param, grad, m, v in [(U, dU, mU, vU), (V, dV, mV, vV), (bias, dbias, mb, vb)]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return U, V, bias

def evaluate(X, y, U, V, bias):
    n, d = X.shape
    h = np.zeros((n, U.shape[2]), np.float32)
    for j in range(d):
        h += U[j, X[:, j], :]
    logits = h @ V + bias
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]
    tune_size = n // 5
    train_X, train_y = X[:-tune_size], y[:-tune_size]
    tune_X, tune_y = X[-tune_size:], y[-tune_size:]
    
    best_acc = -1.0
    best_params = None
    for R in R_LIST:
        for lr in LR_LIST:
            for epochs in EPOCHS_LIST:
                U, V, bias = train_model(train_X, train_y, R, lr, epochs, BATCH_SIZE, SEED)
                acc = evaluate(tune_X, tune_y, U, V, bias)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (R, lr, epochs)
                print(f"R={R}, lr={lr}, epochs={epochs}, tune_acc={acc:.4f}")
    
    print(f"Best: R={best_params[0]}, lr={best_params[1]}, epochs={best_params[2]}, acc={best_acc:.4f}")
    
    R, lr, epochs = best_params
    final_epochs = epochs * 2
    U, V, bias = train_model(X, y, R, lr, final_epochs, BATCH_SIZE, SEED)
    
    magic = 0x52574F4C  # "LOWR" little-endian
    out = struct.pack("<4i", magic, d, R, C)
    out += U.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x52574F4C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], R = hd[2], C = hd[3];
    if (d <= 0 || d > 256 || R <= 0 || R > 1024 || C <= 0 || C > 256) {
        fprintf(stderr, "bad weights dims\n"); return 3;
    }
    long expected = 16L + 4L * ((long)d * 256 * R + (long)R * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    float *U = (float *)(wb + 16);
    float *V = U + (long)d * 256 * R;
    float *bias = V + (long)R * C;
    
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * R);
    float *scores = malloc(sizeof(float) * C);
    if (!h || !scores) { fprintf(stderr, "malloc failed\n"); exit(3); }
    
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(h, 0, sizeof(float) * R);
        for (int j = 0; j < d; j++) {
            const float *u = U + ((long)j * 256 + xi[j]) * R;
            for (int r = 0; r < R; r++) h[r] += u[r];
        }
        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int r = 0; r < R; r++) {
            float hr = h[r];
            if (hr == 0.0f) continue;
            const float *v = V + (long)r * C;
            for (int c = 0; c < C; c++) scores[c] += hr * v[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```