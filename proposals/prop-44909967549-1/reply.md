```hypothesis
Family: label-poly2 (degree-2 polynomial classifier). Optdigits pixels have strong spatial correlations; quadratic terms capture pairwise interactions (strokes, curves) that linear models miss. With 64 features, 2080 quadratic features remain small enough for exact inference via quadratic forms. Expect dev accuracy > 0.965 (beating label-linear's 0.9613) at ~2x CPU cost. Failure: accuracy ≤ 0.9613 or inference > 10k Ir/item.
```
```python
"""label-poly2: Degree-2 polynomial classifier for ordinal byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, numpy only.
Exports: magic 0x504F4C32, d, C; float32 mean[d], std[d]; float32 W[C][d*(d+1)/2] (lower-triangular); float32 bias[C]."""
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
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Z-score each feature
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    Z = ((X - mu) / sd).astype(np.float32)  # n x d

    # Build polynomial features: lower-triangular indices
    idx = []
    for i in range(d):
        for j in range(i, d):
            idx.append((i, j))
    D = len(idx)  # d*(d+1)/2

    # Design matrix Phi: n x D
    Phi = np.empty((n, D), dtype=np.float32)
    for k, (i, j) in enumerate(idx):
        if i == j:
            Phi[:, k] = Z[:, i] * Z[:, j]
        else:
            Phi[:, k] = Z[:, i] * Z[:, j] * 2.0  # factor 2 for symmetric gradient

    # Logistic regression with Adam
    W = (rng.uniform(-1, 1, (C, D)) * math.sqrt(6.0 / D)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(logits, y_idx):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(y_idx)), y_idx] -= 1.0
        return (probs / len(y_idx)).astype(np.float32)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            batch_idx = order[s:s + BATCH]
            phi = Phi[batch_idx]
            yy = y[batch_idx]
            logits = phi @ W.T + b
            g = softmax_grad(logits, yy)
            grad_W = g.T @ phi + L2 * W
            grad_b = g.sum(axis=0)
            grads = [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Remove the factor 2 from off-diagonal weights for inference
    W_out = W.copy()
    for k, (i, j) in enumerate(idx):
        if i != j:
            W_out[:, k] *= 0.5

    # Pack weights
    magic = 0x504F4C32  # 'POL2'
    out = struct.pack("<3i", magic, d, C)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-poly2 inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: one byte class (argmax).
 * Weights format: magic 0x504F4C32, int32 d, C; float32 mean[d], std[d];
 * float32 W[C][d*(d+1)/2] (lower-triangular, row-major); float32 bias[C]. */
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
    if (nw < 12 || hdr[0] != 0x504F4C32) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hdr[1], C = hdr[2];
    int D = d * (d + 1) / 2;
    long expected = 12L + 4L * d * 2 + 4L * C * D + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *mean = (const float *)(wb + 12);
    const float *std = mean + d;
    const float *W = std + d;
    const float *bias = W + (long)C * D;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) z[j] = (xi[j] - mean[j]) / std[j];

        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int c = 0; c < C; c++) {
            const float *Wc = W + (long)c * D;
            int idx = 0;
            for (int u = 0; u < d; u++) {
                for (int v = u; v < d; v++) {
                    scores[c] += Wc[idx++] * z[u] * z[v];
                }
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```