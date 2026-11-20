```hypothesis
label-linear-int8: Quantized linear model treating each pixel as 16-category categorical (0-15), weights int8/biases int32. Categorical encoding captures non-linear pixel responses better than ordinal z-scoring; 10KB quantized weights fit in L1 cache enabling ~3000 Ir/item (vs 5058 for label-linear). Expect dev accuracy ≥96.1% (matching label-linear) with 40% lower instruction count. Failure if accuracy <95.5% or Ir/item >5000.
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
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # Clamp to 16 categories (0..15)
    X = np.clip(X, 0, 15)

    # One-hot encode: (n, d*16)
    Z = np.zeros((n, d * 16), dtype=np.float32)
    for j in range(d):
        Z[np.arange(n), j * 16 + X[:, j]] = 1.0

    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (d * 16, C)) * math.sqrt(6.0 / (d * 16))).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            logits = z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad_logits = (probs / len(yy)).astype(np.float32)

            grad_W = z.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(0)

            t += 1
            for i, (p, gr) in enumerate(zip(params, [grad_W, grad_b])):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Quantize to int8/int32 with shared scale
    max_abs = max(np.abs(W).max(), np.abs(b).max())
    scale = 127.0 / max_abs if max_abs > 0 else 1.0
    W_q = np.round(W * scale).astype(np.int8)
    b_q = np.round(b * scale).astype(np.int32)

    # Write weights.bin: magic, d, C, scale, W_q, b_q
    magic = 0x4C493831  # "LI81"
    out = struct.pack("<3if", magic, d, C, scale)
    out += W_q.tobytes()
    out += b_q.tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x4C493831) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], C = hd[2];
    float scale = *(float *)(hd + 3);
    (void)scale;
    int expected = 16 + d * 16 * C + C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %d\n", nw, expected); return 3; }
    int8_t *W = (int8_t *)(wb + 16);
    int32_t *b = (int32_t *)(wb + 16 + d * 16 * C);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    int32_t *scores = malloc(C * sizeof(int32_t));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < d; j++) {
            int v = xi[j] > 15 ? 15 : xi[j];
            int base = (j * 16 + v) * C;
            for (int c = 0; c < C; c++) scores[c] += W[base + c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```