```hypothesis
Factorization Machine (FM) with low-rank pairwise interactions: each feature value gets a weight and a latent vector; score = bias + sum weights + sum_{i<j} <v_i, v_j>. Adult has known feature interactions (e.g., education × occupation) that a linear model misses. FM adds O(d*k) parameters (d=14, k=4..8) and computes interactions in O(d*k) time via the (sum^2 - sum of squares) trick. Expect dev accuracy > 0.8524 (label-linear) with similar inference cost (few hundred ops/item). Failure if accuracy ≤ label-linear or instructions/item much higher.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
BATCH = 64

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_fm(X, y, d, k, lr, l2, epochs, seed):
    rng = np.random.default_rng(seed)
    b0 = 0.0
    w = rng.normal(0, 0.01, size=(d, 256)).astype(np.float32)
    V = rng.normal(0, 0.01, size=(d, 256, k)).astype(np.float32)
    m_w = np.zeros_like(w); v_w = np.zeros_like(w)
    m_V = np.zeros_like(V); v_V = np.zeros_like(V)
    m_b0 = 0.0; v_b0 = 0.0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    n = X.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, BATCH):
            idx = order[start:start+BATCH]
            xb = X[idx]; yb = y[idx]
            B = len(yb)
            linear = b0 + np.sum(w[np.arange(d), xb.T], axis=0)
            interaction = 0.0
            for f in range(k):
                Vf = V[:, :, f]
                S = np.sum(Vf[np.arange(d), xb.T], axis=0)
                sq = np.sum(Vf[np.arange(d), xb.T]**2, axis=0)
                interaction += 0.5 * (S*S - sq)
            score = linear + interaction
            p = 1.0 / (1.0 + np.exp(-score))
            g = (p - yb) / B
            grad_b0 = np.sum(g)
            grad_w = np.zeros_like(w)
            for j in range(d):
                np.add.at(grad_w[j], xb[:, j], g)
            grad_w += 2 * l2 * w
            grad_V = np.zeros_like(V)
            for f in range(k):
                Vf = V[:, :, f]
                S = np.sum(Vf[np.arange(d), xb.T], axis=0)
                for j in range(d):
                    vals = xb[:, j]
                    Vjv = Vf[j, vals]
                    np.add.at(grad_V[j, :, f], vals, g * (S - Vjv))
            grad_V += 2 * l2 * V
            t += 1
            m_b0 = beta1 * m_b0 + (1 - beta1) * grad_b0
            v_b0 = beta2 * v_b0 + (1 - beta2) * grad_b0**2
            b0 -= lr * (m_b0 / (1 - beta1**t)) / (np.sqrt(v_b0 / (1 - beta2**t)) + eps)
            for j in range(d):
                m_w[j] = beta1 * m_w[j] + (1 - beta1) * grad_w[j]
                v_w[j] = beta2 * v_w[j] + (1 - beta2) * grad_w[j]**2
                w[j] -= lr * (m_w[j] / (1 - beta1**t)) / (np.sqrt(v_w[j] / (1 - beta2**t)) + eps)
                for f in range(k):
                    m_V[j, :, f] = beta1 * m_V[j, :, f] + (1 - beta1) * grad_V[j, :, f]
                    v_V[j, :, f] = beta2 * v_V[j, :, f] + (1 - beta2) * grad_V[j, :, f]**2
                    V[j, :, f] -= lr * (m_V[j, :, f] / (1 - beta1**t)) / (np.sqrt(v_V[j, :, f] / (1 - beta2**t)) + eps)
    return b0, w, V

def eval_fm(X, y, weights, d, k):
    b0, w, V = weights
    linear = b0 + np.sum(w[np.arange(d), X.T], axis=0)
    interaction = 0.0
    for f in range(k):
        Vf = V[:, :, f]
        S = np.sum(Vf[np.arange(d), X.T], axis=0)
        sq = np.sum(Vf[np.arange(d), X.T]**2, axis=0)
        interaction += 0.5 * (S*S - sq)
    score = linear + interaction
    pred = (score > 0).astype(np.int64)
    return np.mean(pred == y)

def save_weights(path, weights, d, k):
    b0, w, V = weights
    with open(path, "wb") as f:
        f.write(struct.pack("<4i", 0x464D4C31, d, k, 1))
        f.write(struct.pack("<f", b0))
        for j in range(d):
            for v in range(256):
                f.write(struct.pack("<f", w[j, v]))
                for f_idx in range(k):
                    f.write(struct.pack("<f", V[j, v, f_idx]))

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    n = X.shape[0]
    tune_size = int(0.2 * n)
    X_tr, y_tr = X[:n-tune_size], y[:n-tune_size]
    X_tu, y_tu = X[n-tune_size:], y[n-tune_size:]

    best_acc = -1.0
    best_weights = None
    best_cfg = None
    for k in (4, 8):
        for lr in (0.01, 0.05):
            for l2 in (1e-4, 1e-3):
                for epochs in (10, 20):
                    weights = train_fm(X_tr, y_tr, d, k, lr, l2, epochs, SEED)
                    acc = eval_fm(X_tu, y_tu, weights, d, k)
                    if acc > best_acc:
                        best_acc = acc
                        best_weights = weights
                        best_cfg = (k, lr, l2, epochs)
    k, lr, l2, epochs = best_cfg
    final_weights = train_fm(X, y, d, k, lr, l2, epochs, SEED)
    save_weights(sys.argv[4], final_weights, d, k)

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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t*)wb;
    if (hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (d != 14 || k <= 0 || C != 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4 + (long)d * 256 * (1 + k) * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    float *params = (float*)(wb + 16);
    float b0 = params[0];
    float *w = params + 1;
    float *V = w + (long)d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        float score = b0;
        float S[16] = {0}, sq[16] = {0};
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            float w_jv = w[(long)j * 256 + v];
            score += w_jv;
            for (int f = 0; f < k; f++) {
                float V_jvf = V[((long)j * 256 + v) * k + f];
                S[f] += V_jvf;
                sq[f] += V_jvf * V_jvf;
            }
        }
        for (int f = 0; f < k; f++) score += 0.5f * (S[f]*S[f] - sq[f]);
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```