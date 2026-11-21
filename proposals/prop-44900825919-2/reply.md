```hypothesis
Family: label-emb (per-position byte embeddings + linear classifier).
Why: The linear baseline uses one-hot per position (4096 features). Replacing with learned embeddings (D=16) reduces parameters 6x, acts as strong regularization, and captures character similarities. Should exceed label-linear's 0.3375 dev accuracy with lower CPU cost due to smaller memory footprint and fewer FLOPs.
Failure: If dev accuracy ≤ 0.3375 or CPU cost ≥ label-linear, the low-rank assumption doesn't hold for this task.
```

```python
"""label-emb: per-position byte embeddings followed by a linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
D = 16                # embedding dimension per position
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8

MAGIC = 0x454D424C  # "LMBE" little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Embeddings: (d, 256, D)
    scale = math.sqrt(6.0 / (256 + D))
    emb = rng.uniform(-scale, scale, (d, 256, D)).astype(np.float32)

    # Linear weights: (d*D, C), bias: (C,)
    W = rng.uniform(-scale, scale, (d * D, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    # Adam state
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward: gather embeddings
            # emb_gathered: (B, d, D)
            emb_gathered = emb[np.arange(d)[None, :], xb]   # fancy indexing
            # Flatten to (B, d*D)
            flat = emb_gathered.reshape(B, -1)

            # Linear scores
            scores = flat @ W + b          # (B, C)

            # Softmax + cross-entropy gradient
            max_s = scores.max(axis=1, keepdims=True)
            exp_s = np.exp(scores - max_s)
            probs = exp_s / exp_s.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_scores = probs / B        # (B, C)

            # Gradient wrt linear weights and bias
            grad_W = flat.T @ grad_scores + L2 * W     # (d*D, C)
            grad_b = grad_scores.sum(axis=0)           # (C,)

            # Gradient wrt flat embeddings
            grad_flat = grad_scores @ W.T              # (B, d*D)
            grad_emb_gathered = grad_flat.reshape(B, d, D)

            # Scatter back to embedding gradients
            grad_emb = np.zeros_like(emb)
            np.add.at(grad_emb, (np.arange(d)[None, :], xb), grad_emb_gathered)
            grad_emb += L2 * emb

            # Adam updates
            t += 1
            # embeddings
            m_emb = BETA1 * m_emb + (1 - BETA1) * grad_emb
            v_emb = BETA2 * v_emb + (1 - BETA2) * grad_emb * grad_emb
            m_emb_hat = m_emb / (1 - BETA1 ** t)
            v_emb_hat = v_emb / (1 - BETA2 ** t)
            emb -= LR * m_emb_hat / (np.sqrt(v_emb_hat) + EPS)

            # linear weights
            m_W = BETA1 * m_W + (1 - BETA1) * grad_W
            v_W = BETA2 * v_W + (1 - BETA2) * grad_W * grad_W
            m_W_hat = m_W / (1 - BETA1 ** t)
            v_W_hat = v_W / (1 - BETA2 ** t)
            W -= LR * m_W_hat / (np.sqrt(v_W_hat) + EPS)

            # bias
            m_b = BETA1 * m_b + (1 - BETA1) * grad_b
            v_b = BETA2 * v_b + (1 - BETA2) * grad_b * grad_b
            m_b_hat = m_b / (1 - BETA1 ** t)
            v_b_hat = v_b / (1 - BETA2 ** t)
            b -= LR * m_b_hat / (np.sqrt(v_b_hat) + EPS)

    # Write weights.bin
    # Header: magic, d, D, C (int32)
    header = struct.pack("<4i", MAGIC, d, D, C)
    # Embeddings: d * 256 * D floats (float32 little-endian)
    emb_bytes = emb.astype("<f4").tobytes()
    # Linear weights: (d*D) * C floats
    W_bytes = W.astype("<f4").tobytes()
    # Bias: C floats
    b_bytes = b.astype("<f4").tobytes()

    out = header + emb_bytes + W_bytes + b_bytes
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index.
 * Weights format: int32 magic, d, D, C; float32 emb[d][256][D], W[d*D][C], b[C]. */
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
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C < 1 || C > 256 || D < 1) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * D + (long)d * D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * D;
    const float *b = W + (long)d * D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            memcpy(concat + j * D, e, D * sizeof(float));
        }

        // Linear layer
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int k = 0; k < d * D; k++) {
            float v = concat[k];
            if (v == 0.0f) continue;
            const float *w = W + (long)k * C;
            for (int c = 0; c < C; c++) scores[c] += v * w[c];
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```