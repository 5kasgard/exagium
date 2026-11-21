```hypothesis
Family: label-fm (Factorization Machine with per-feature-value embeddings, K=8).
Why: Adult income has strong pairwise interactions (education×occupation, capital-gain×hours). FM captures all O(d^2) interactions via low-rank dot products, adding only ~256 KB weights. Inference is O(d*K*C) ≈ 224 FMAs/item, faster than MLP. Beats linear (0.8524) by modeling interactions; if it doesn't, interactions are weak or K=8 is insufficient.
```

```python
"""Factorization Machine for tabular label tasks with byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Learns per-feature-value embeddings (K=8) for pairwise interactions.
Exported: int32 magic 0x464D3130, d, K, C; float32 bias[C], W_lin[d][256][C], V[d][256][K][C]."""
import json, math, struct, sys
import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100

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
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    
    rng = np.random.default_rng(SEED)
    
    bias = np.zeros(C, np.float32)
    W_lin = rng.normal(0, 0.01, (d, 256, C)).astype(np.float32)
    V = rng.normal(0, 0.01, (d, 256, K, C)).astype(np.float32)
    
    params = [bias, W_lin, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]
            yb = y[idx]
            B = len(yb)
            
            # Forward
            scores = np.tile(bias, (B, 1))
            s = np.zeros((B, K, C), np.float32)
            s2 = np.zeros((B, K, C), np.float32)
            
            for j in range(d):
                vals = xb[:, j]
                scores += W_lin[j, vals, :]
                v_j = V[j, vals, :, :]  # (B, K, C)
                s += v_j
                s2 += v_j * v_j
            
            scores += 0.5 * np.sum(s * s - s2, axis=1)
            
            # Softmax + loss grad
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            grad_scores = probs
            grad_scores[np.arange(B), yb] -= 1.0
            grad_scores /= B
            
            # Backward
            grad_bias = grad_scores.sum(axis=0)
            
            grad_W_lin = np.zeros_like(W_lin)
            grad_V = np.zeros_like(V)
            
            for j in range(d):
                vals = xb[:, j]
                np.add.at(grad_W_lin[j], vals, grad_scores)
                
                v_j = V[j, vals, :, :]  # (B, K, C)
                for f in range(K):
                    diff = s[:, f, :] - v_j[:, f, :]  # (B, C)
                    grad_contrib = diff * grad_scores  # (B, C)
                    np.add.at(grad_V[j, :, f, :], vals, grad_contrib)
            
            grad_W_lin += L2 * W_lin
            grad_V += L2 * V
            
            # Adam
            t += 1
            grads = [grad_bias, grad_W_lin, grad_V]
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    
    # Save
    out = struct.pack("<4i", 0x464D3130, d, K, C)
    out += bias.astype("<f4").tobytes()
    out += W_lin.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 14) { fprintf(stderr, "bad weights dims\n"); return 3; }
    
    long expected = 16 + (long)C * 4 + (long)d * 256 * C * 4 + (long)d * 256 * K * C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    
    const float *p = (const float *)(hd + 4);
    const float *bias = p; p += C;
    const float *W_lin = p; p += (long)d * 256 * C;
    const float *V = p;
    
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *s = malloc(sizeof(float) * C * K);
    float *s2 = malloc(sizeof(float) * C * K);
    
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        
        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int c = 0; c < C; c++)
            for (int f = 0; f < K; f++)
                s[c * K + f] = s2[c * K + f] = 0.0f;
        
        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            const float *wlin_ptr = W_lin + ((long)j * 256 + val) * C;
            const float *v_ptr = V + ((long)j * 256 + val) * K * C;
            
            for (int c = 0; c < C; c++) {
                scores[c] += wlin_ptr[c];
                for (int f = 0; f < K; f++) {
                    float v = v_ptr[f * C + c];
                    s[c * K + f] += v;
                    s2[c * K + f] += v * v;
                }
            }
        }
        
        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int f = 0; f < K; f++) {
                float sf = s[c * K + f];
                float s2f = s2[c * K + f];
                inter += 0.5f * (sf * sf - s2f);
            }
            scores[c] += inter;
        }
        
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```