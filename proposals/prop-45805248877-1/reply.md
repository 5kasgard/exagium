```hypothesis
Family: label-poly2local — Linear model (logistic regression) with spatially local pairwise interactions on z-scored 8×8 optdigits pixels. The 64 ordinal features (0–16) gain 56 horizontal + 56 vertical neighbor products = 176 features total, capturing stroke edges/corners that a purely linear model misses. Trained with the same Adam/L2 regimen as label-linear (zero init, lr=3e-3, L2=1e-4, batch=64). Expected: >96.13% dev accuracy at ~6000 Ir/item. Fails if accuracy ≤ label-linear or cost >2× label-linear.
```

```python
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
    rng = np.random.default_rng(SEED)
    n = X.shape[0]
    
    # Z-score original features
    Xf = X.astype(np.float64)
    mu = Xf.mean(axis=0)
    sd = Xf.std(axis=0)
    sd[sd == 0] = 1.0
    Xz = (Xf - mu) / sd
    
    # Build interaction features: horizontal and vertical adjacent pairs
    inter_indices = []
    for r in range(8):
        for c in range(7):
            inter_indices.append((r*8 + c, r*8 + c + 1))
    for r in range(7):
        for c in range(8):
            inter_indices.append((r*8 + c, (r+1)*8 + c))
    
    n_inter = len(inter_indices)  # 112
    X_inter = np.zeros((n, n_inter), dtype=np.float32)
    for k, (i, j) in enumerate(inter_indices):
        X_inter[:, k] = Xz[:, i] * Xz[:, j]
    
    # Combined features: 64 + 112 = 176
    Z = np.concatenate([Xz.astype(np.float32), X_inter], axis=1)
    K = Z.shape[1]
    
    # Linear model (zero init like label-linear)
    W = np.zeros((K, C), dtype=np.float32)
    b = np.zeros(C, dtype=np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    
    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)
    
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            logits = z @ W + b
            g = softmax_grad(logits, yy)
            grads = [z.T @ g + L2 * W, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    
    # Save weights: magic, d, n_inter, C, mu[d], sd[d], W[K*C], b[C]
    magic = 0x504F4C32  # "POL2"
    out = struct.pack("<4i", magic, d, n_inter, C)
    out += mu.astype(np.float32).tobytes()
    out += sd.astype(np.float32).tobytes()
    out += W.astype(np.float32).tobytes()
    out += b.astype(np.float32).tobytes()
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
    if (nw < 16 || hd[0] != 0x504F4C32) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], n_inter = hd[2], C = hd[3];
    if (d != 64 || n_inter != 112 || C != 10) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    int K = d + n_inter; // 176
    size_t expected = 16 + (size_t)d * 4 * 2 + (size_t)K * C * 4 + (size_t)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %zu\n", nw, expected); return 3; }
    
    float *mu = (float *)(hd + 4);
    float *sd = mu + d;
    float *W = sd + d;
    float *b = W + (size_t)K * C;
    
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);
    
    // Adjacent pairs indices (same as training)
    int inter_i[112], inter_j[112];
    int idx = 0;
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 7; c++) {
            inter_i[idx] = r * 8 + c;
            inter_j[idx] = r * 8 + c + 1;
            idx++;
        }
    }
    for (int r = 0; r < 7; r++) {
        for (int c = 0; c < 8; c++) {
            inter_i[idx] = r * 8 + c;
            inter_j[idx] = (r + 1) * 8 + c;
            idx++;
        }
    }
    
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        
        // Z-score original features
        for (int j = 0; j < d; j++) {
            z[j] = (xi[j] - mu[j]) / sd[j];
        }
        // Interaction features
        for (int p = 0; p < n_inter; p++) {
            z[d + p] = z[inter_i[p]] * z[inter_j[p]];
        }
        // Linear scores
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int f = 0; f < K; f++) {
                sum += z[f] * W[(size_t)f * C + c];
            }
            scores[c] = sum;
        }
        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (scores[c] > scores[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```