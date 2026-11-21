```hypothesis
Family: label-poly2 (degree-2 polynomial logistic regression). 
Why: Optdigits pixels have local correlations; pairwise products capture stroke intersections and curves that linear models miss. 
Failure: If dev accuracy <= 0.9613 (label-linear) or cost per item > 2x label-linear, the family does not justify its complexity.
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


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    n = X.shape[0]

    # Standardize per feature
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0

    # Expanded feature dimension: d (linear) + d*(d+1)//2 (quadratic)
    D = d + d * (d + 1) // 2

    # Precompute expanded features for all training samples
    Z = (X.astype(np.float32) - mu) / sd  # (n, d)
    Phi = np.zeros((n, D), dtype=np.float32)
    Phi[:, :d] = Z
    idx = d
    for i in range(d):
        for j in range(i, d):
            Phi[:, idx] = Z[:, i] * Z[:, j]
            idx += 1

    rng = np.random.default_rng(SEED)
    # Initialize weights
    W = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx_batch = order[s:s + BATCH]
            phi = Phi[idx_batch]
            yy = y[idx_batch]
            # forward
            logits = phi @ W + b  # (batch, C)
            # softmax and gradient
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad_logits = (probs / len(yy)).astype(np.float32)  # (batch, C)
            # gradients
            grad_W = phi.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
            grads = [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9
                m += 0.1 * gr
                v *= 0.999
                v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    # Format: magic (4 bytes), d (int32), C (int32), D (int32), mu[d] (float32), sd[d] (float32), W[D*C] (float32 row-major), b[C] (float32)
    magic = 0x504F4C59  # "POLY"
    out = struct.pack("<4i", magic, d, C, D)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()  # row-major: D rows, C cols
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    uint32_t magic = hd[0];
    if (magic != 0x504F4C59) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], D = hd[3];
    if (d <= 0 || C <= 0 || C > 256 || D != d + d*(d+1)/2) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L*d + 4L*d + 4L*D*C + 4L*C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mu = (const float *)(wb + 16);
    const float *sd = mu + d;
    const float *W = sd + d;
    const float *b = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *z = malloc(sizeof(float) * d);
    float *phi = malloc(sizeof(float) * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Standardize
        for (int j = 0; j < d; j++) {
            z[j] = (xi[j] - mu[j]) / sd[j];
        }
        // Build phi: linear then quadratic
        int idx = 0;
        for (int j = 0; j < d; j++) {
            phi[idx++] = z[j];
        }
        for (int i1 = 0; i1 < d; i1++) {
            for (int i2 = i1; i2 < d; i2++) {
                phi[idx++] = z[i1] * z[i2];
            }
        }
        // Compute scores = phi * W + b
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int f = 0; f < D; f++) {
                sum += phi[f] * W[(long)f * C + c];
            }
            scores[c] = sum;
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (scores[c] > scores[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }

    free(z);
    free(phi);
    free(scores);
    free(wb);
    if (x) free(x);
    return 0;
}
```