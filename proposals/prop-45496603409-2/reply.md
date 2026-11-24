```hypothesis
Family: label-embed (shared-embedding logistic regression).  
Learns a single 256×D embedding table for all byte values, sums the 16 context embeddings, and applies a linear classifier.  
This captures inter-character similarities (e.g., letters vs punctuation) with fewer parameters than one-hot linear models, improving generalization on small data.  
Failure: dev accuracy ≤ 0.3375 (label-linear) or inference cost per item not lower.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
D = 64
LR = 1e-3
L2 = 1e-5
BATCH = 64
EPOCHS = 50
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X_bytes = records(sys.argv[2])
    y_bytes = records(sys.argv[3])
    n = len(X_bytes)
    X = np.frombuffer(b"".join(X_bytes), dtype=np.uint8).reshape(n, d)
    y = np.array([b[0] for b in y_bytes], dtype=np.int64)
    rng = np.random.default_rng(SEED)

    E = rng.normal(0, 1.0/math.sqrt(D), size=(256, D)).astype(np.float32)
    W = rng.normal(0, math.sqrt(2.0/(D+C)), size=(D, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    mE, vE = np.zeros_like(E), np.zeros_like(E)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)

    t = 0
    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X[idx], y[idx]
            B = len(yb)
            emb = E[xb]
            h = emb.sum(axis=1)
            logits = h @ W + b
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(B), yb] -= 1.0
            grad_logits /= B
            grad_W = h.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
            grad_h = grad_logits @ W.T
            grad_E = np.zeros_like(E)
            for j in range(d):
                np.add.at(grad_E, xb[:, j], grad_h)
            grad_E += L2 * E
            t += 1
            mE = BETA1 * mE + (1 - BETA1) * grad_E
            vE = BETA2 * vE + (1 - BETA2) * (grad_E * grad_E)
            E -= LR * (mE / (1 - BETA1**t)) / (np.sqrt(vE / (1 - BETA2**t)) + EPS)
            mW = BETA1 * mW + (1 - BETA1) * grad_W
            vW = BETA2 * vW + (1 - BETA2) * (grad_W * grad_W)
            W -= LR * (mW / (1 - BETA1**t)) / (np.sqrt(vW / (1 - BETA2**t)) + EPS)
            mb = BETA1 * mb + (1 - BETA1) * grad_b
            vb = BETA2 * vb + (1 - BETA2) * (grad_b * grad_b)
            b -= LR * (mb / (1 - BETA1**t)) / (np.sqrt(vb / (1 - BETA2**t)) + EPS)

    magic = 0x454D4244
    header = struct.pack("<4i", magic, d, D, C)
    out = header + E.astype("<f4").tobytes() + W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C < 1 || C > 256 || D < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)256 * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *E = (const float *)(hd + 4);
    const float *W = E + (long)256 * D;
    const float *b = W + (long)D * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * D);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < D; j++) h[j] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *e = E + (long)xi[j] * D;
            for (int kk = 0; kk < D; kk++) h[kk] += e[kk];
        }
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int kk = 0; kk < D; kk++) sum += h[kk] * W[(long)kk * C + c];
            scores[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```