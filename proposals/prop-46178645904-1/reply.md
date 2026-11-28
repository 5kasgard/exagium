```hypothesis
label-mlp2h: a 2-hidden-layer MLP (64→32→16→10) with per-feature z-score normalization and ReLU activations. The two hidden layers should capture hierarchical feature interactions (strokes → digit parts → digits) better than linear or single-layer models on this 8×8 pixel task. If dev accuracy ≤ 0.965 or inference cost exceeds 2× label-linear, the hypothesis fails.
```
```python
import json, struct, sys

import numpy as np

SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 50
H1, H2 = 32, 16


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
    n, d = X.shape

    # Per-feature z-score normalization
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    X_norm = (X.astype(np.float32) - mu) / sd

    rng = np.random.default_rng(SEED)

    # Xavier / He initialization
    W1 = (rng.normal(0, np.sqrt(2.0 / d), (H1, d))).astype(np.float32)
    b1 = np.zeros(H1, np.float32)
    W2 = (rng.normal(0, np.sqrt(2.0 / H1), (H2, H1))).astype(np.float32)
    b2 = np.zeros(H2, np.float32)
    W3 = (rng.normal(0, np.sqrt(2.0 / H2), (C, H2))).astype(np.float32)
    b3 = np.zeros(C, np.float32)

    params = [W1, b1, W2, b2, W3, b3]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x = X_norm[idx]
            yy = y[idx]
            bs = len(yy)

            # Forward pass
            h1 = x @ W1.T + b1
            a1 = np.maximum(h1, 0)
            h2 = a1 @ W2.T + b2
            a2 = np.maximum(h2, 0)
            logits = a2 @ W3.T + b3

            # Softmax gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yy] -= 1.0
            grad_out = probs / bs

            # Backward pass
            grad_W3 = grad_out.T @ a2 + L2 * W3
            grad_b3 = grad_out.sum(axis=0)

            grad_a2 = grad_out @ W3
            grad_h2 = grad_a2 * (h2 > 0)
            grad_W2 = grad_h2.T @ a1 + L2 * W2
            grad_b2 = grad_h2.sum(axis=0)

            grad_a1 = grad_h2 @ W2
            grad_h1 = grad_a1 * (h1 > 0)
            grad_W1 = grad_h1.T @ x + L2 * W1
            grad_b1 = grad_h1.sum(axis=0)

            grads = [grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3]

            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights: magic, d, h1, h2, C, mu[d], sd[d], W1[h1,d], b1[h1], W2[h2,h1], b2[h2], W3[C,h2], b3[C]
    out = struct.pack("<5i", 0x4D4C5032, d, H1, H2, C)
    out += mu.tobytes() + sd.tobytes()
    out += W1.tobytes() + b1.tobytes()
    out += W2.tobytes() + b2.tobytes()
    out += W3.tobytes() + b3.tobytes()
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
    if (nw < 20 || hd[0] != 0x4D4C5032) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], h1 = hd[2], h2 = hd[3], C = hd[4];
    if (d != 64 || h1 != 32 || h2 != 16 || C != 10) { fprintf(stderr, "unexpected dims\n"); return 3; }

    long expected = 20 + 4L * (d + d + h1*d + h1 + h2*h1 + h2 + C*h2 + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    float *ptr = (float *)(hd + 5);
    float *mu = ptr; ptr += d;
    float *sd = ptr; ptr += d;
    float *W1 = ptr; ptr += h1 * d;
    float *b1 = ptr; ptr += h1;
    float *W2 = ptr; ptr += h2 * h1;
    float *b2 = ptr; ptr += h2;
    float *W3 = ptr; ptr += C * h2;
    float *b3 = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *in = malloc(d * sizeof(float));
    float *h1_arr = malloc(h1 * sizeof(float));
    float *h2_arr = malloc(h2 * sizeof(float));
    float *logits = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) in[j] = (xi[j] - mu[j]) / sd[j];

        for (int u = 0; u < h1; u++) {
            float sum = b1[u];
            for (int j = 0; j < d; j++) sum += in[j] * W1[u * d + j];
            h1_arr[u] = sum > 0 ? sum : 0;
        }

        for (int u = 0; u < h2; u++) {
            float sum = b2[u];
            for (int v = 0; v < h1; v++) sum += h1_arr[v] * W2[u * h1 + v];
            h2_arr[u] = sum > 0 ? sum : 0;
        }

        for (int c = 0; c < C; c++) {
            float sum = b3[c];
            for (int u = 0; u < h2; u++) sum += h2_arr[u] * W3[c * h2 + u];
            logits[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```