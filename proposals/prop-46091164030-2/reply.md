```hypothesis
Family: label-embmlp — position-specific byte embeddings + one ReLU hidden layer + linear output.
Why: The linear model (0.3375) treats each position independently; a tiny embedding (E=16) per position lets the
hidden layer (H=64) learn cross-position n-gram features without blowing up parameters (~86k weights). Inference
stays fast: 16 lookup + 256x64 + 64x65 ≈ 20k FLOPs/item. If dev accuracy ≤ 0.3375 or CPU cost ≥ label-linear,
the family does not justify its complexity.
```

```python
"""label-embmlp: position-specific embeddings (E=16) + one ReLU layer (H=64) + linear output.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

E = 16          # embedding size per position
H = 64          # hidden layer size
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
    d = card["input"]["bytes"]
    C = card["classes"]
    assert d == 16 and C == 65

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # parameters
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = xb.shape[0]

            # forward: embed -> flatten -> hidden -> output
            flat = np.empty((bsz, d * E), np.float32)
            for j in range(d):
                flat[:, j * E:(j + 1) * E] = emb[j, xb[:, j]]
            h_pre = flat @ W1 + b1
            h = np.maximum(h_pre, 0, dtype=np.float32)
            logits = h @ W2 + b2

            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yb] -= 1.0
            probs /= bsz

            # backward
            dW2 = h.T @ probs
            db2 = probs.sum(axis=0)
            dh = probs @ W2.T
            dh[h_pre <= 0] = 0.0
            dW1 = flat.T @ dh
            db1 = dh.sum(axis=0)
            dflat = dh @ W1.T

            # embedding gradients
            demb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(demb[j], xb[:, j], dflat[:, j * E:(j + 1) * E])

            grads = [demb, dW1, db1, dW2, db2]

            # L2 on weights (not biases)
            grads[1] += L2 * W1
            grads[3] += L2 * W2
            grads[0] += L2 * emb

            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * g
                v *= 0.999; v += 0.001 * g * g
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # pack weights: magic, d, E, H, C, then emb[d][256][E], W1, b1, W2, b2 (all float32 LE)
    magic = 0x454D424D  # 'EMBM'
    expected = 16 + 4 * (d * 256 * E + d * E * H + H + H * C + C)
    out = struct.pack("<5i", magic, d, E, H, C)
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    assert len(out) == expected, f"size mismatch {len(out)} != {expected}"
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 raw bytes. Output record: 1 byte (class index 0..64).
 * Weights layout (little-endian):
 *   int32 magic(0x454D424D), d, E, H, C
 *   float32 emb[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C] */
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
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20 + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * E);
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* flatten embeddings */
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(flat + j * E, e, E * sizeof(float));
        }

        /* hidden layer: h = ReLU(flat * W1 + b1) */
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < d * E; v++) sum += flat[v] * W1[(long)v * H + u];
            h[u] = sum > 0.0f ? sum : 0.0f;
        }

        /* output layer: logits = h * W2 + b2 */
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[(long)u * C + c];
            logits[c] = sum;
        }

        /* argmax */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```