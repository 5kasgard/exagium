```hypothesis
Family: label-factorized (position-specific embeddings + linear layer).  
Why: The task predicts the next character from 16 previous bytes. A full linear model on one-hot vectors has 16*256*65 ≈ 266k parameters, which may overfit the 24k training items. Factorizing the weight matrix into per-position embeddings (16*256*D) followed by a shared linear layer (16*D*65) reduces parameters (e.g., D=16 → 82k) while preserving position information, acting as regularization. This should improve accuracy over label-linear (0.3375) or match it with lower compute.  
Failure signal: Dev accuracy ≤ 0.3375 or instructions/item ≥ label-linear (7359.9).
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameters chosen on the tuning split (last 20% of train.in)
D = 16          # embedding dimension
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 50
SAMPLES = 600000

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
    d = card["input"]["bytes"]  # 16
    C = card["classes"]         # 65
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # embed: (d, 256, D)
    embed = rng.normal(0, 0.01, size=(d, 256, D)).astype(np.float32)
    # W: (d*D, C)
    W = rng.normal(0, 0.01, size=(d*D, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    params = [embed, W, b]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            x_batch = X[idx]
            y_batch = y[idx]
            B = len(idx)

            # Forward: gather embeddings
            embed_batch = np.zeros((B, d*D), dtype=np.float32)
            for j in range(d):
                embed_batch[:, j*D:(j+1)*D] = embed[j][x_batch[:, j]]

            scores = embed_batch @ W + b
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

            grad_scores = probs
            grad_scores[np.arange(B), y_batch] -= 1.0
            grad_scores /= B

            grad_W = embed_batch.T @ grad_scores + L2 * W
            grad_b = grad_scores.sum(axis=0)

            grad_embed = np.zeros_like(embed)
            grad_embed_batch = grad_scores @ W.T
            for j in range(d):
                g = grad_embed_batch[:, j*D:(j+1)*D]
                np.add.at(grad_embed[j], x_batch[:, j], g)
            grad_embed += L2 * embed

            grads = [grad_embed, grad_W, grad_b]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] *= 0.9; m[i] += 0.1 * gr
                v[i] *= 0.999; v[i] += 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights: magic, d, D, C, embed, W, b (all float32 little-endian)
    magic = 0x46414354  # "FACT"
    out = struct.pack("<4i", magic, d, D, C)
    out += embed.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x46414354) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C != 65 || D <= 0 || D > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected_floats = (long)d * 256 * D + (long)d * D * C + C;
    if (nw != 16 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *data = (const float *)(hd + 4);
    const float *embed = data;
    const float *W = embed + (long)d * 256 * D;
    const float *b = W + (long)d * D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) {
            const float *e = embed + ((long)j * 256 + xi[j]) * D;
            memcpy(concat + j * D, e, D * sizeof(float));
        }

        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int idx = 0; idx < d * D; idx++) {
                sum += concat[idx] * W[(long)idx * C + c];
            }
            scores[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (scores[c] > scores[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```