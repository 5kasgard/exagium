```hypothesis
Family: label-1hot-linear (per-pixel one-hot embeddings + linear layer).
Why: Optdigits pixels are ordinal 0..16; z-scoring assumes linearity. One-hot per pixel (17 values) lets the model learn arbitrary non-linear intensity-to-class relationships at only 64*17*10=10880 weights. Inference is 640 adds/sample (no multiplies), faster than label-linear's 640 multiply-adds.
Fail signal: dev accuracy ≤ 0.9613 (label-linear) or Ir/item ≥ 5058.
```

```python
"""label-1hot-linear: per-pixel one-hot embeddings (0..16) + linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weight format: int32 magic 0x4F4E4548, d, V, C; float32 E[d][V][C], b[C]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x4F4E4548  # "ONEH"
MAX_VAL = 16        # optdigits pixel max


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
    d, C = X.shape[1], card["classes"]
    V = MAX_VAL + 1                     # 17 values 0..16
    rng = np.random.default_rng(SEED)

    # Embeddings: (d, V, C) -> flatten to (d*V, C) for fast gather
    E = (rng.uniform(-1, 1, (d * V, C)) * math.sqrt(6.0 / (d * V))).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [E, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # Pixel offsets for gather: pixel j uses rows j*V .. (j+1)*V
    offsets = (np.arange(d) * V).astype(np.int64)   # (d,)

    n = X.shape[0]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]                     # (B, d)
            yb = y[idx]                     # (B,)
            # Gather embeddings: indices (B, d) -> (B, d, C)
            idx_flat = (xb + offsets).reshape(-1)           # (B*d,)
            emb = E[idx_flat].reshape(-1, d, C)             # (B, d, C)
            scores = emb.sum(axis=1) + b                    # (B, C)

            # Softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores, dtype=np.float32)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_scores = (probs / len(yb)).astype(np.float32)   # (B, C)

            # Gradient wrt embeddings: scatter-add
            grad_E = np.zeros_like(E)
            np.add.at(grad_E, idx_flat, grad_scores.repeat(d, axis=0).reshape(-1, C))
            grad_E += L2 * E
            grad_b = grad_scores.sum(axis=0) + L2 * b
            grads = [grad_E, grad_b]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    out = struct.pack("<4i", MAGIC, d, V, C)
    out += E.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-1hot-linear inference: ./infer weights.bin in.rec N
 * Input: 64-byte records (pixels 0..16). Output: 1-byte class index. */
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
    if (nw < 16 || hd[0] != 0x4F4E4548) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], V = hd[2], C = hd[3];
    if (d != 64 || V != 17 || C < 1 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16 + 4L * d * V * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *E = (const float *)(hd + 4);
    const float *b = E + (long)d * V * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            if (v >= V) v = V - 1;              // clamp (should not happen)
            const float *row = E + ((long)j * V + v) * C;
            for (int c = 0; c < C; c++) scores[c] += row[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```