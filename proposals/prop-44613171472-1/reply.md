```hypothesis
Family: label-fm (Factorization Machine, degree 2, rank 4, one-vs-rest).
Why: Optdigits pixels exhibit local pairwise correlations that linear models miss; FM captures all pairwise interactions with only O(d*k) params per class (k=4), adding ~13KB weights and ~2500 ops/item. Beats linear (0.961) if interactions matter; fails (≤linear) if data is truly linear or overfits.
```

```python
import json, math, struct, sys
import numpy as np

# Fixed hyperparameters
K = 4           # FM rank
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
EPOCHS = 30

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X_bytes.shape[0]

    # Standardize features (ordinal z-score)
    mu = np.zeros(d, dtype=np.float32)
    sd = np.ones(d, dtype=np.float32)
    for j in range(d):
        v = X_bytes[:, j].astype(np.float64)
        mu[j] = v.mean()
        sd[j] = v.std() or 1.0
    X = ((X_bytes.astype(np.float32) - mu) / sd).astype(np.float32)  # (n, d)

    rng = np.random.default_rng(SEED)
    # Weights per class: bias, w[d], V[d][K]
    biases = np.zeros(C, dtype=np.float32)
    W = np.zeros((C, d), dtype=np.float32)
    V = np.zeros((C, d, K), dtype=np.float32)
    for c in range(C):
        V[c] = (rng.normal(0, 0.01, (d, K))).astype(np.float32)

    # Train each class independently (one-vs-rest)
    for c in range(C):
        y_bin = (y == c).astype(np.float32) * 2 - 1  # +1 / -1
        # Adam state
        m_b, v_b = 0.0, 0.0
        m_w = np.zeros(d, dtype=np.float32)
        v_w = np.zeros(d, dtype=np.float32)
        m_V = np.zeros((d, K), dtype=np.float32)
        v_V = np.zeros((d, K), dtype=np.float32)
        t = 0
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for epoch in range(EPOCHS):
            order = rng.permutation(n)
            for s in range(0, n, BATCH):
                idx = order[s:s+BATCH]
                xb = X[idx]  # (b, d)
                yb = y_bin[idx]  # (b,)
                bsz = len(idx)
                # Forward
                lin = xb @ W[c] + biases[c]  # (b,)
                xV = xb @ V[c]  # (b, K)
                x2 = xb * xb
                V2 = V[c] * V[c]
                x2V2 = x2 @ V2  # (b, K)
                inter = 0.5 * np.sum(xV * xV - x2V2, axis=1)
                pred = lin + inter  # (b,)
                # Logistic loss gradient
                z = yb * pred
                sig = 1.0 / (1.0 + np.exp(z))
                grad = -yb * sig  # (b,)
                # Gradients
                g_b = np.mean(grad)
                g_w = (grad[:, None] * xb).mean(axis=0) + L2 * W[c]
                g_V = np.zeros((d, K), dtype=np.float32)
                for f in range(K):
                    xV_f = xV[:, f]
                    term = xb * (xV_f[:, None] - xb * V[c][:, f])
                    g_V[:, f] = (grad[:, None] * term).mean(axis=0) + L2 * V[c][:, f]
                # Adam update
                t += 1
                m_b = beta1 * m_b + (1 - beta1) * g_b
                v_b = beta2 * v_b + (1 - beta2) * g_b * g_b
                biases[c] -= LR * (m_b / (1 - beta1**t)) / (np.sqrt(v_b / (1 - beta2**t)) + eps)
                m_w = beta1 * m_w + (1 - beta1) * g_w
                v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
                W[c] -= LR * (m_w / (1 - beta1**t)) / (np.sqrt(v_w / (1 - beta2**t)) + eps)
                m_V = beta1 * m_V + (1 - beta1) * g_V
                v_V = beta2 * v_V + (1 - beta2) * g_V * g_V
                V[c] -= LR * (m_V / (1 - beta1**t)) / (np.sqrt(v_V / (1 - beta2**t)) + eps)

    # Pack weights: magic, d, K, C; mu[d], sd[d]; per class: bias, w[d], V[d*K]
    out = struct.pack("<4i", 0x464D3130, d, K, C)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    for c in range(C):
        out += struct.pack("<f", biases[c])
        out += W[c].astype("<f4").tobytes()
        out += V[c].astype("<f4").tobytes()
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
    unsigned char *wbuf = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wbuf;
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 64 || K < 1 || K > 16 || C != 10) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 8L * d + 4L * C * (1 + d + d * K);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    float *mu = (float *)(hd + 4);
    float *sd = mu + d;
    float *class_data = sd + d;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *xV = malloc(sizeof(float) * K);
    float *x2V2 = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            float *cptr = class_data + c * (1 + d + d * K);
            float bias = cptr[0];
            float *w = cptr + 1;
            float *V = cptr + 1 + d;

            float score = bias;
            for (int f = 0; f < K; f++) { xV[f] = 0.0f; x2V2[f] = 0.0f; }
            for (int j = 0; j < d; j++) {
                float z = (xi[j] - mu[j]) / sd[j];
                score += z * w[j];
                for (int f = 0; f < K; f++) {
                    float Vjf = V[j * K + f];
                    xV[f] += z * Vjf;
                    x2V2[f] += z * z * Vjf * Vjf;
                }
            }
            for (int f = 0; f < K; f++) {
                score += 0.5f * (xV[f] * xV[f] - x2V2[f]);
            }
            scores[c] = score;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```