```hypothesis
Family: label-poly2 (explicit degree-2 polynomial logistic regression).
Why: Captures pairwise pixel interactions (strokes, corners) that linear models miss, crucial for 8x8 digits.
The expanded feature space (2144 dims) is still tiny; inference ~21k FLOPs/item beats MLP cost.
Failure: If accuracy <= 0.9613 (label-linear) or inference cost > 15k Ir/item.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def expand_poly2(X):
    n, d = X.shape
    M = d + d * (d + 1) // 2
    Phi = np.empty((n, M), dtype=np.float32)
    Phi[:, :d] = X
    col = d
    for i in range(d):
        for j in range(i, d):
            Phi[:, col] = X[:, i] * X[:, j]
            col += 1
    return Phi

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    
    X_f64 = X_bytes.astype(np.float64)
    mu = X_f64.mean(axis=0)
    std = X_f64.std(axis=0)
    std[std == 0] = 1.0
    X_norm = ((X_f64 - mu) / std).astype(np.float32)
    
    Phi = expand_poly2(X_norm)
    n, M = Phi.shape
    
    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (M, C)) * math.sqrt(6.0 / M)).astype(np.float32)
    b = np.zeros(C, np.float32)
    
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            phi = Phi[idx]
            yy = y[idx]
            
            scores = phi @ W + b
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad = probs / len(yy)
            
            grad_W = phi.T @ grad + L2 * W
            grad_b = grad.sum(axis=0)
            
            t += 1
            for i, (p, gr) in enumerate(zip(params, [grad_W, grad_b])):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    
    magic = 0x504F4C32
    out = struct.pack("<4i", magic, d, M, C)
    out += mu.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x504F4C32) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], M = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1 || M != d + d*(d+1)/2) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L*d*2 + 4L*M*C + 4L*C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    
    const float *mu = (const float *)(hd + 4);
    const float *std = mu + d;
    const float *W = std + d;
    const float *b = W + (long)M * C;
    
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *phi = malloc(sizeof(float) * M);
    float *scores = malloc(sizeof(float) * C);
    
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        
        for (int j = 0; j < d; j++) phi[j] = (xi[j] - mu[j]) / std[j];
        int col = d;
        for (int ii = 0; ii < d; ii++) {
            float vi = phi[ii];
            for (int jj = ii; jj < d; jj++) phi[col++] = vi * phi[jj];
        }
        
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int m = 0; m < M; m++) {
            float phim = phi[m];
            const float *Wm = W + (long)m * C;
            for (int c = 0; c < C; c++) scores[c] += phim * Wm[c];
        }
        
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```