```hypothesis
Family: label-embed (per-position byte embeddings + linear classifier).
Why: The 16-byte context has 256 categories per position; one-hot encoding (4096 dims) is high-dimensional and sparse. Embeddings (E=32) compress each position to a dense vector, reducing parameters from ~266k to ~164k and sharing statistical strength across bytes. This should improve generalization on 24k training examples and lower inference cost (smaller matvec).
Failure: If dev accuracy ≤ 0.3375 (label-linear) or instructions/item not lower, the compression loses critical information for this task.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters
E = 32          # embedding dimension
LR = 0.01
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 20
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]  # 16
    C = card["classes"]         # 65
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize embeddings: list of (256, E) for each position
    scale_emb = math.sqrt(6.0 / (256 + E))
    emb = [rng.uniform(-1, 1, (256, E)).astype(np.float32) * scale_emb for _ in range(d)]
    # Linear weights: (d*E, C)
    scale_W = math.sqrt(6.0 / (d * E + C))
    W = rng.uniform(-1, 1, (d * E, C)).astype(np.float32) * scale_W
    b = np.zeros(C, dtype=np.float32)

    # Adam state
    params = emb + [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    for epoch in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]  # (batch, d)
            yb = y[idx]  # (batch,)

            # Forward: gather embeddings
            batch_emb = np.zeros((len(idx), d * E), dtype=np.float32)
            for j in range(d):
                batch_emb[:, j * E:(j + 1) * E] = emb[j][xb[:, j]]

            logits = batch_emb @ W + b  # (batch, C)
            # Softmax
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            # Loss gradient
            grad_logits = probs
            grad_logits[np.arange(len(yb)), yb] -= 1.0
            grad_logits /= len(yb)

            # Backprop
            grad_W = batch_emb.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
            grad_emb = [np.zeros_like(e) for e in emb]
            grad_batch_emb = grad_logits @ W.T  # (batch, d*E)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_batch_emb[:, j * E:(j + 1) * E])
                grad_emb[j] += L2 * emb[j]

            grads = grad_emb + [grad_W, grad_b]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    MAGIC = 0x454D4244  # "EMBD"
    out = struct.pack("<4i", MAGIC, d, E, C)
    for j in range(d):
        out += emb[j].astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights config\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * E + (long)d * E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + (long)d * E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *vec = malloc(sizeof(float) * d * E);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * E;
            float *dst = vec + j * E;
            memcpy(dst, src, E * sizeof(float));
        }
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int k = 0; k < d * E; k++) {
                sum += vec[k] * W[(long)k * C + c];
            }
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (logits[c] > logits[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```