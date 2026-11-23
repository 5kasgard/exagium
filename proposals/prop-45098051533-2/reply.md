```hypothesis
Family: label-embedmlp — per-position byte embeddings (dim 8) fed into a 64-unit ReLU hidden layer, then a 65-class output. The embeddings capture richer byte interactions than one-hot linear while inference stays fast via precomputed lookup tables (same pattern as label-linear). If dev accuracy does not exceed label-linear's 0.3375, the extra capacity is not helping on this task.
```

```python
"""label-embedmlp: per-position byte embeddings + one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format: magic 0x454D4244, int32 d, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
SEED = 1
E = 8          # embedding dimension per position
H = 64         # hidden layer size
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
MAGIC = 0x454D4244  # "EMBD" in little-endian


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

    rng = np.random.default_rng(SEED)

    # Initialize embeddings: list of d arrays (256, E)
    embeddings = []
    for _ in range(d):
        # Xavier uniform for embeddings
        limit = math.sqrt(6.0 / (256 + E))
        emb = rng.uniform(-limit, limit, (256, E)).astype(np.float32)
        embeddings.append(emb)

    # Initialize W1 (d*E, H), b1 (H)
    k = d * E
    limit = math.sqrt(6.0 / (k + H))
    W1 = rng.uniform(-limit, limit, (k, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)

    # Initialize W2 (H, C), b2 (C)
    limit = math.sqrt(6.0 / (H + C))
    W2 = rng.uniform(-limit, limit, (H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    # Adam state for all parameters
    params = [W1, b1, W2, b2] + embeddings
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            batch_size = len(idx)
            xb = X[idx]          # (batch, d)
            yb = y[idx]          # (batch,)

            # Forward: embeddings -> concat -> hidden -> ReLU -> output
            Z = np.zeros((batch_size, k), dtype=np.float32)
            for j in range(d):
                Z[:, j*E:(j+1)*E] = embeddings[j][xb[:, j]]
            h_pre = Z @ W1 + b1
            h = np.maximum(h_pre, 0, dtype=np.float32)
            scores = h @ W2 + b2

            # Softmax cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(batch_size), yb] -= 1.0
            grad_scores = probs / batch_size  # (batch, C)

            # Backprop output layer
            grad_W2 = h.T @ grad_scores + L2 * W2
            grad_b2 = grad_scores.sum(axis=0)
            grad_h = grad_scores @ W2.T

            # ReLU gradient
            grad_h_pre = grad_h * (h_pre > 0)

            # Backprop hidden layer
            grad_W1 = Z.T @ grad_h_pre + L2 * W1
            grad_b1 = grad_h_pre.sum(axis=0)
            grad_Z = grad_h_pre @ W1.T

            # Backprop embeddings
            grad_embeddings = [np.zeros_like(emb) for emb in embeddings]
            for j in range(d):
                g = grad_Z[:, j*E:(j+1)*E]  # (batch, E)
                # Accumulate gradients for each byte value
                np.add.at(grad_embeddings[j], xb[:, j], g)

            # Collect gradients in same order as params
            grads = [grad_W1, grad_b1, grad_W2, grad_b2] + grad_embeddings

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= BETA1; m += (1 - BETA1) * gr
                v *= BETA2; v += (1 - BETA2) * gr * gr
                m_hat = m / (1 - BETA1 ** t)
                v_hat = v / (1 - BETA2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + EPS)

    # Precompute T[d][256][H] = embedding[j][v] @ W1
    T = np.zeros((d, 256, H), dtype=np.float32)
    for j in range(d):
        T[j] = embeddings[j] @ W1

    # Write weights.bin
    header = struct.pack("<4i", MAGIC, d, H, C)
    body = T.astype("<f4").tobytes() + b1.astype("<f4").tobytes() + W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* label-embedmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes (d=16). Output record: 1 byte (class index).
 * Weights format: magic 0x454D4244, int32 d, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected_floats = (long)d * 256 * H + H + (long)H * C + C;
    if (nw != 16 + 4L * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // acc = b1
        memcpy(acc, b1, H * sizeof(float));
        // add T[j][xi[j]] for each position
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) acc[h] += t[h];
        }
        // ReLU
        for (int h = 0; h < H; h++) if (acc[h] < 0.0f) acc[h] = 0.0f;
        // scores = acc @ W2 + b2
        memcpy(scores, b2, C * sizeof(float));
        for (int h = 0; h < H; h++) {
            float ah = acc[h];
            if (ah == 0.0f) continue;
            const float *w = W2 + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += ah * w[c];
        }
        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(acc); free(scores);
    return 0;
}
```