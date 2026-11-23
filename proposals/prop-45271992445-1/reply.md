```hypothesis
Family: label-localpoly (linear model on pixels + adjacent pixel products).
Why: Optdigits 8x8 digits have strong local stroke correlations; pairwise products of neighboring pixels capture edge/curve info missed by linear model.
Failure: If dev accuracy ≤ label-linear (0.9613) or inference cost > 2× label-linear.
```
```python
"""label-localpoly: Linear model on original pixels + products of spatially adjacent pixels.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, numpy only.
Exports: magic, d=64, num_pairs=112, C=10, mu[64], inv_sd[64], W[176][10], b[10] all float32 little-endian."""
import json, math, struct, sys
import numpy as np

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
    C = card["classes"]
    d = card["input"]["bytes"]  # 64
    assert d == 64
    
    # Adjacent pairs for 8x8 grid
    pairs = []
    for r in range(8):
        for c in range(7):
            pairs.append((r*8 + c, r*8 + c + 1))  # horizontal
    for r in range(7):
        for c in range(8):
            pairs.append((r*8 + c, (r+1)*8 + c))  # vertical
    num_pairs = len(pairs)  # 112
    
    rng = np.random.default_rng(SEED)
    
    # Compute per-pixel mean and std (z-score)
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    inv_sd = 1.0 / sd
    
    # Encode training data
    X_enc = ((X.astype(np.float32) - mu) * inv_sd)  # (n, 64)
    # Pair features
    X_pair = np.empty((X.shape[0], num_pairs), dtype=np.float32)
    for idx, (i, j) in enumerate(pairs):
        X_pair[:, idx] = X_enc[:, i] * X_enc[:, j]
    Z = np.concatenate([X_enc, X_pair], axis=1)  # (n, 176)
    
    n, k = Z.shape  # k = 176
    # Initialize weights
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
            # Softmax gradient
            logits = z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad_logits = (probs / len(yy)).astype(np.float32)
            
            grad_W = z.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
            grads = [grad_W, grad_b]
            
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    
    # Export weights
    MAGIC = 0x50434F4C  # "LOCP" little-endian
    out = struct.pack("<4i", MAGIC, d, num_pairs, C)
    out += mu.astype("<f4").tobytes()
    out += inv_sd.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-localpoly: ./infer weights.bin in.rec N
 * Linear model on 64 pixels + 112 adjacent products (8x8 grid). */
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
    if (nw < 16 || hd[0] != 0x50434F4C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], num_pairs = hd[2], C = hd[3];
    if (d != 64 || num_pairs != 112 || C != 10) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    int k = d + num_pairs; // 176
    long expected = 16L + 4L * (d * 2 + (long)k * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }
    
    float *mu = (float *)(hd + 4);
    float *inv_sd = mu + d;
    float *W = inv_sd + d;
    float *b = W + (long)k * C;
    
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *encoded = malloc(sizeof(float) * d);
    float *scores = malloc(sizeof(float) * C);
    
    // Predefine pairs (same order as training)
    int pairs[112][2];
    int pidx = 0;
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 7; c++) {
            pairs[pidx][0] = r*8 + c;
            pairs[pidx][1] = r*8 + c + 1;
            pidx++;
        }
    }
    for (int r = 0; r < 7; r++) {
        for (int c = 0; c < 8; c++) {
            pairs[pidx][0] = r*8 + c;
            pairs[pidx][1] = (r+1)*8 + c;
            pidx++;
        }
    }
    
    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }
        
        // Encode pixels
        for (int j = 0; j < d; j++) {
            encoded[j] = (xi[j] - mu[j]) * inv_sd[j];
        }
        
        // Linear scores: b + sum_j encoded[j] * W[j][c] + sum_p encoded[i]*encoded[j] * W[d+p][c]
        for (int c = 0; c < C; c++) scores[c] = b[c];
        
        // Main effects
        for (int j = 0; j < d; j++) {
            float ej = encoded[j];
            float *wj = W + (long)j * C;
            for (int c = 0; c < C; c++) scores[c] += ej * wj[c];
        }
        
        // Pair interactions
        for (int p = 0; p < num_pairs; p++) {
            float prod = encoded[pairs[p][0]] * encoded[pairs[p][1]];
            float *wp = W + (long)(d + p) * C;
            for (int c = 0; c < C; c++) scores[c] += prod * wp[c];
        }
        
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```