```hypothesis
Family: label-fm (Factorization Machine with pairwise interaction vector). 
Each (position, byte) gets a K-dim embedding and linear weight. The model sums embeddings to compute a K-dim pairwise interaction vector p_k = 0.5*(S_k^2 - Q_k), then scores = linear + p @ W + bias. 
Captures all pairwise position interactions (not just adjacent) with O(d*K + K*C) inference, beating linear's 33.7% by modeling character co-occurrence patterns (e.g., "th", "ing") at ~30K Ir/item. 
Fails if K too small (<16) or if overfitting on 24K samples hurts vs. linear's 266K params.
```

```python
import json, math, struct, sys
import numpy as np

K = 32
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-3, 1e-5, 128, 600000, 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def softmax_grad(logits, y):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    E = rng.normal(0, 1/math.sqrt(K), size=(d, 256, K)).astype(np.float32)
    L = rng.normal(0, 1/math.sqrt(256), size=(d, 256, C)).astype(np.float32)
    W = rng.normal(0, 1/math.sqrt(K), size=(K, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    params = [E, L, W, b]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X[idx], y[idx]
            B = xb.shape[0]

            scores = np.broadcast_to(b, (B, C)).copy()
            S = np.zeros((B, K), np.float32)
            Q = np.zeros((B, K), np.float32)
            for j in range(d):
                scores += L[j, xb[:, j]]
                e = E[j, xb[:, j]]
                S += e
                Q += e * e
            p = 0.5 * (S * S - Q)
            scores += p @ W

            g_sc = softmax_grad(scores, yb)

            grad_W = p.T @ g_sc
            grad_b = g_sc.sum(axis=0)
            grad_p = g_sc @ W.T
            grad_S = grad_p * S
            grad_Q = -0.5 * grad_p

            grad_E = np.zeros_like(E)
            grad_L = np.zeros_like(L)
            for j in range(d):
                onehot = np.zeros((B, 256), np.float32)
                onehot[np.arange(B), xb[:, j]] = 1
                grad_L[j] = onehot.T @ g_sc
                e = E[j, xb[:, j]]
                g = grad_S + 2 * e * grad_Q
                grad_E[j] = onehot.T @ g

            grad_E += L2 * E
            grad_L += L2 * L
            grad_W += L2 * W
            grads = [grad_E, grad_L, grad_W, grad_b]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * gr * gr
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    magic = 0x464D0001
    header = struct.pack("<4i", magic, d, K, C)
    out = header + E.astype("<f4").tobytes() + L.astype("<f4").tobytes() + W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
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
    uint32_t k = buf[*pos] | (buf[*pos+1]<<8) | (buf[*pos+2]<<16) | ((uint32_t)buf[*pos+3]<<24);
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
    uint8_t *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D0001) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 16 || C < 1 || C > 256 || K < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * K + (long)d * 256 * C + (long)K * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld vs %ld\n", nw, expected); return 3; }

    float *ptr = (float *)(hd + 4);
    float *E = ptr; ptr += (long)d * 256 * K;
    float *L = ptr; ptr += (long)d * 256 * C;
    float *W = ptr; ptr += (long)K * C;
    float *b = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *S = malloc(sizeof(float) * K);
    float *Q = malloc(sizeof(float) * K);
    float *p = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
       