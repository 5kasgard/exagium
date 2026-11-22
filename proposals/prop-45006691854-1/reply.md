```hypothesis
Factorization Machine (label-fm) with rank-4 latent vectors per feature per class. Captures pairwise pixel interactions crucial for digit shapes with only ~3K params vs linear's 160K lookup tables. Expect >0.97 accuracy at ~4000 Ir/item. Fails if interactions don't help or SGD converges poorly.
```
```python
import json, math, struct, sys
import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 64, 600000, 100

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
    rng = np.random.default_rng(SEED)
    
    # Encode features: ordinal z-scored
    scaled_table = np.zeros((d, 256), np.float32)
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            m, s = v.mean(), v.std() or 1.0
            scaled_table[j] = ((np.arange(256) - m) / s).astype(np.float32)
        else:
            m, s = v.mean(), v.std() or 1.0
            scaled_table[j] = ((np.arange(256) - m) / s).astype(np.float32)
    
    # Prepare scaled data
    Z = scaled_table[np.arange(d), X.T].T  # (n, d)
    n = Z.shape[0]
    
    # Init parameters
    w = (rng.normal(0, 0.01, (C, d))).astype(np.float32)
    v = (rng.normal(0, 0.01, (C, d, K))).astype(np.float32)
    bias = np.zeros(C, np.float32)
    
    # Adam state
    mw, mv, mb = np.zeros_like(w), np.zeros_like(v), np.zeros_like(bias)
    vw, vv, vb = np.zeros_like(w), np.zeros_like(v), np.zeros_like(bias)
    t = 0
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            z = Z[idx]  # (b, d)
            yy = y[idx]  # (b,)
            b = z.shape[0]
            
            # Forward
            lin = z @ w.T + bias  # (b, C)
            inter = np.zeros((b, C), np.float32)
            for c in range(C):
                lat_sum = z @ v[c]  # (b, K)
                v_norm_sq = np.sum(v[c]**2, axis=1)  # (d,)
                lat_sq_sum = np.sum(z**2 * v_norm_sq, axis=1)  # (b,)
                inter[:, c] = 0.5 * (np.sum(lat_sum**2, axis=1) - lat_sq_sum)
            scores = lin + inter
            
            # Softmax gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(b), yy] -= 1.0
            grad_scores = probs / b  # (b, C)
            
            # Gradients
            grad_w = z.T @ grad_scores  # (d, C)
            grad_bias = grad_scores.sum(axis=0)
            grad_v = np.zeros_like(v)
            for c in range(C):
                g = grad_scores[:, c]
                lat_sum = z @ v[c]
                d_lat_sum = g[:, None] * lat_sum
                grad_v[c] = z.T @ d_lat_sum
                factor = np.sum(g[:, None] * z**2, axis=0)
                grad_v[c] -= v[c] * factor[:, None]
            
            grad_w += L2 * w
            grad_v += L2 * v
            
            # Adam update
            t += 1
            mw = beta1 * mw + (1 - beta1) * grad_w.T
            vw = beta2 * vw + (1 - beta2) * (grad_w.T ** 2)
            w -= LR * (mw / (1 - beta1**t)) / (np.sqrt(vw / (1 - beta2**t)) + eps)
            
            mv = beta1 * mv + (1 - beta1) * grad_v
            vv = beta2 * vv + (1 - beta2) * (grad_v ** 2)
            v -= LR * (mv / (1 - beta1**t)) / (np.sqrt(vv / (1 - beta2**t)) + eps)
            
            mb = beta1 * mb + (1 - beta1) * grad_bias
            vb = beta2 * vb + (1 - beta2) * (grad_bias ** 2)
            bias -= LR * (mb / (1 - beta1**t)) / (np.sqrt(vb / (1 - beta2**t)) + eps)
    
    # Export weights
    out = struct.pack("<4i", 0x464D31, d, K, C)
    out += scaled_table.astype("<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 16) { fprintf(stderr, "bad dims\n"); return 3; }
    
    long expected = 16;
    expected += (long)d * 256 * 4;
    expected += (long)C * d * 4;
    expected += (long)C * d * K * 4;
    expected += (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    
    const float *ptr = (const float *)(hd + 4);
    const float *scaled_table = ptr; ptr += (long)d * 256;
    const float *w = ptr; ptr += (long)C * d;
    const float *v = ptr; ptr += (long)C * d * K;
    const float *bias = ptr;
    
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *lat_sum = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);
    
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        
        for (int c = 0; c < C; c++) scores[c] = bias[c];
        
        for (int j = 0; j < d; j++) {
            float s = scaled_table[j * 256 + xi[j]];
            for (int c = 0; c < C; c++) {
                scores[c] += w[c * d + j] * s;
            }
        }
        
        for (int c = 0; c < C; c++) {
            for (int kk = 0; kk < K; kk++) lat_sum[kk] = 0.0f;
            float lat_sq_sum = 0.0f;
            
            for (int j = 0; j < d; j++) {
                float s = scaled_table[j * 256 + xi[j]];
                const float *v_cj = v + (c * d + j) * K;
                for (int kk = 0; kk < K; kk++) {
                    float lv = v_cj[kk] * s;
                    lat_sum[kk] += lv;
                    lat_sq_sum += lv * lv;
                }
            }
            
            float sum_sq = 0.0f;
            for (int kk = 0; kk < K; kk++) sum_sq += lat_sum[kk] * lat_sum[kk];
            scores[c] += 0.5f * (sum_sq - lat_sq_sum);
        }
        
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    
    return 0;
}
```