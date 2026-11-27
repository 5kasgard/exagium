```hypothesis
Family: label-fm (Factorization Machine for ordinal byte features).
Why: Optdigits pixels exhibit strong local pairwise interactions (strokes); FM captures all pairwise interactions via low-rank factorization (k=8), adding only ~5K params vs linear's 640. Same O(d*k*C) inference cost as a tiny MLP but with explicit interaction modeling.
Failure: If dev accuracy ≤ 0.9613 (label-linear) or Ir/item > 15000 (3× linear), the factorization does not justify its complexity.
```
```python
"""Factorization Machine for ordinal byte features: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Encodes ordinal bytes as z-scores. FM with k=8 factors per class.
Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)). Exports: magic, d, d, C, k; float32 mu[d], sd[d], V[d][k][C], W[d][C], b[C]."""
import json, math, struct, sys

import numpy as np

K_FACTORS = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x464D3100  # "FM1\0" little-endian

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
    for t in card["input"]["types"]:
        assert t == "ordinal", "label-fm only supports ordinal features"
    
    rng = np.random.default_rng(SEED)
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    Z = ((X.astype(np.float32) - mu) / sd)
    
    n, K = Z.shape
    W = (rng.uniform(-1, 1, (K, C)) * math.sqrt(6.0 / K)).astype(np.float32)
    b = np.zeros(C, np.float32)
    V = (rng.uniform(-1, 1, (K, K_FACTORS, C)) * math.sqrt(6.0 / K)).astype(np.float32)
    
    params = [W, b, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    
    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return (probs / len(yy)).astype(np.float32)
    
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z = Z[idx]
            yy = y[idx]
            linear = z @ W + b
            inter = np.zeros((len(yy), C), np.float32)
            for c in range(C):
                Vc = V[:, :, c]
                vx = z @ Vc
                inter[:, c] = 0.5 * ((vx * vx).sum(axis=1) - (z[:, :, None] * Vc[None, :, :] ** 2).sum(axis=(1,2)))
            logits = linear + inter
            g = softmax_grad(logits, yy)
            
            grad_W = z.T @ g + L2 * W
            grad_b = g.sum(axis=0)
            grad_V = np.zeros_like(V)
            for c in range(C):
                Vc = V[:, :, c]
                vx_c = z @ Vc
                term1 = z.T @ vx_c
                term2 = (z ** 2).T @ Vc
                grad_V[:, :, c] = term1 - term2 + L2 * Vc
            
            grads = [grad_W, grad_b, grad_V]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    
    out = struct.pack("<5i", MAGIC, d, K, C, K_FACTORS)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine for ordinal byte features: ./infer weights.bin in.rec N
 * Input: d bytes per record (z-scored using stored mu, sd). 
 * Model: score_c = b_c + sum_j z_j W_{j,c} + 0.5 * (sum_u (sum_j z_j V_{j,u,c})^2 - sum_j sum_u (z_j V_{j,u,c})^2)
 * Output: one byte class index (argmax, first max wins). */
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
    int32_t *hdr = (int32_t *)wb;
    if (nw < 20 || hdr[0] != 0x464D3100) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hdr[1], K = hdr[2], C = hdr[3], k = hdr[4];
    if (K != d) { fprintf(stderr, "only ordinal features supported (K!=d)\n"); return 3; }
    if (C < 1 || C > 256 || k < 1 || k > 64) { fprintf(stderr, "bad dims\n"); return 3; }
    
    size_t expected = 20 + (size_t)(2*d + d*k*C + d*C + C) * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size: got %ld, expected %zu\n", nw, expected); return 3; }
    
    float *p = (float *)(wb + 20);
    float *mu = p; p += d;
    float *sd = p; p += d;
    float *V = p; p += (size_t)d * k * C;
    float *W = p; p += (size_t)d * C;
    float *b = p;
    
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *scores = malloc(sizeof(float) * C);
    float *vx = malloc(sizeof(float) * k);
    
    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }
        
        for (int j = 0; j < d; j++) {
            z[j] = (xi[j] - mu[j]) / sd[j];
        }
        
        for (int c = 0; c < C; c++) {
            scores[c] = b[c];
            for (int j = 0; j < d; j++) {
                scores[c] += z[j] * W[(size_t)j * C + c];
            }
        }
        
        for (int c = 0; c < C; c++) {
            for (int u = 0; u < k; u++) vx[u] = 0.0f;
            float sum_vx2 = 0.0f;
            for (int j = 0; j < d; j++) {
                float zj = z[j];
                for (int u = 0; u < k; u++) {
                    float v = V[((size_t)j * k + u) * C + c];
                    float vx_ju = zj * v;
                    vx[u] += vx_ju;
                    sum_vx2 += vx_ju * vx_ju;
                }
            }
            float sum_vx_sq = 0.0f;
            for (int u = 0; u < k; u++) sum_vx_sq += vx[u] * vx[u];
            scores[c] += 0.5f * (sum_vx_sq - sum_vx2);
        }
        
        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (scores[c] > scores[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }
    
    return 0;
}
```