```hypothesis
Family: label-embed (per-position character embeddings summed + linear classifier).
Why: The 16-byte context has strong local character correlations; embeddings capture similarity (e.g., 'a'≈'b') unlike one-hot linear. With D=16, params ≈ 16*256*16 + 16*65 ≈ 65k vs linear's 4096*65 ≈ 266k, so faster inference and better generalization.
Failure signal: Dev accuracy ≤ 0.3375 (label-linear) or CPU cost ≥ label-linear.
```
```python
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
D = 16           # embedding dimension
SEED = 1
LR = 1e-2
L2 = 1e-4
BATCH = 128
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

    rng = np.random.default_rng(SEED)

    # Embeddings: [d, 256, D]
    emb = rng.normal(0, 0.01, size=(d, 256, D)).astype(np.float32)
    # Output layer: [D, C] and bias [C]
    W = rng.normal(0, 0.01, size=(D, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    params = [emb, W, b]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # [B, d]
            yb = y[idx]          # [B]
            B = len(yb)

            # Forward: sum embeddings over positions
            h = np.sum(emb[xb], axis=1)            # [B, D]
            logits = h @ W + b                      # [B, C]

            # Softmax cross-entropy gradient
            max_logits = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            probs /= B

            # Gradients
            grad_W = h.T @ probs                    # [D, C]
            grad_b = np.sum(probs, axis=0)          # [C]
            grad_h = probs @ W.T                    # [B, D]

            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_h)
            grad_emb += L2 * emb
            grad_W += L2 * W

            # Adam update
            t += 1
            grads = [grad_emb, grad_W, grad_b]
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights: magic, d, D, C, emb[d][256][D], W[D][C], b[C]
    magic = 0x44424D45  # "EMBD" little-endian
    out = struct.pack("<4i", magic, d, D, C)
    out += emb.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x44424D45) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * D;
    const float *b = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * D);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int l = 0; l < D; l++) h[l] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            for (int l = 0; l < D; l++) h[l] += e[l];
        }
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int l = 0; l < D; l++) sum += h[l] * W[(long)l * C + c];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```