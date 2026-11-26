```hypothesis
Family: label-embed-mlp. Uses 16 position-specific 8-d embeddings (16*256*8 params) followed by two ReLU layers (128->64->64) and a 65-class output. Learned embeddings capture character similarities; two hidden layers model context interactions better than linear. Beats label-linear (0.3375) if embedding+MLP structure captures n-gram statistics; fails if 8-d embeddings are too small or training overfits the 24k samples.
```

```python
"""label-embed-mlp: 16 position-specific 8-d embeddings -> ReLU(128->64) -> ReLU(64->64) -> 65 logits.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, numpy only."""
import json, math, struct, sys

import numpy as np

E = 8          # embedding dim per position
H1, H2 = 64, 64
SEED = 1
LR, L2 = 1e-3, 1e-4
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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    rng = np.random.default_rng(SEED)

    # embeddings: list of (256, E) per position
    emb = [(rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32) for _ in range(d)]
    # W1: (d*E, H1), b1: (H1,)
    W1 = (rng.uniform(-1, 1, (d * E, H1)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H1, np.float32)
    # W2: (H1, H2), b2: (H2,)
    W2 = (rng.uniform(-1, 1, (H1, H2)) * math.sqrt(6.0 / H1)).astype(np.float32)
    b2 = np.zeros(H2, np.float32)
    # W3: (H2, C), b3: (C,)
    W3 = (rng.uniform(-1, 1, (H2, C)) * math.sqrt(6.0 / H2)).astype(np.float32)
    b3 = np.zeros(C, np.float32)

    params = [W1, b1, W2, b2, W3, b3]
    for e in emb:
        params.insert(0, e)  # embeddings first
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    steps_per_epoch = (n + BATCH - 1) // BATCH
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # forward
            # embeddings lookup: (bs, d, E)
            emb_vecs = np.stack([e[xb[:, j]] for j, e in enumerate(emb)], axis=1).astype(np.float32)
            x = emb_vecs.reshape(bs, d * E)
            a1 = x @ W1 + b1
            h1 = np.maximum(a1, 0)
            a2 = h1 @ W2 + b2
            h2 = np.maximum(a2, 0)
            logits = h2 @ W3 + b3

            # softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs

            # backprop
            gW3 = h2.T @ probs
            gb3 = probs.sum(0)
            gh2 = probs @ W3.T
            gh2[a2 <= 0] = 0
            gW2 = h1.T @ gh2
            gb2 = gh2.sum(0)
            gh1 = gh2 @ W2.T
            gh1[a1 <= 0] = 0
            gW1 = x.T @ gh1
            gb1 = gh1.sum(0)
            gx = gh1 @ W1.T  # (bs, d*E)

            # embedding gradients
            gemb = []
            for j in range(d):
                gj = gx[:, j * E:(j + 1) * E]
                # accumulate per embedding index
                ge = np.zeros((256, E), np.float32)
                np.add.at(ge, xb[:, j], gj)
                gemb.append(ge)

            grads = gemb + [gW1, gb1, gW2, gb2, gW3, gb3]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * (m_hat / (np.sqrt(v_hat) + 1e-8) + L2 * p)

    # pack weights: magic, d, E, H1, H2, C; then embeddings[d][256][E], W1, b1, W2, b2, W3, b3 (all float32 LE)
    magic = 0x454D4244  # 'EMBD'
    header = struct.pack("<6i", magic, d, E, H1, H2, C)
    blob = header
    for e in emb:
        blob += e.astype("<f4").tobytes()
    for p in [W1, b1, W2, b2, W3, b3]:
        blob += p.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(blob)


if __name__ == "__main__":
    main()
```

```c
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    if (nw < 24 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H1 = hd[3], H2 = hd[4], C = hd[5];
    if (d != 16 || C < 1 || C > 256) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 24L
        + (long)d * 256 * E
        + (long)d * E * H1 + H1
        + (long)H1 * H2 + H2
        + (long)H2 * C + C;
    if (nw != expected * 4) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected * 4); return 3; }

    const float *p = (const float *)(hd + 6);
    const float *emb = p; p += (long)d * 256 * E;
    const float *W1 = p; p += (long)d * E * H1;
    const float *b1 = p; p += H1;
    const float *W2 = p; p += (long)H1 * H2;
    const float *b2 = p; p += H2;
    const float *W3 = p; p += (long)H2 * C;
    const float *b3 = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h1_buf = malloc(sizeof(float) * H1);
    float *h2_buf = malloc(sizeof(float) * H2);
    float *logits = malloc(sizeof(float) * C);
    float *emb_concat = malloc(sizeof(float) * d * E);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(emb_concat + j * E, e, E * sizeof(float));
        }

        // h1 = relu(emb_concat @ W1 + b1)
        for (int u = 0; u < H1; u++) {
            float sum = b1[u];
            for (int v = 0; v < d * E; v++) sum += emb_concat[v] * W1[v * H1 + u];
            h1_buf[u] = sum > 0.0f ? sum : 0.0f;
        }

        // h2 = relu(h1 @ W2 + b2)
        for (int u = 0; u < H2; u++) {
            float sum = b2[u];
            for (int v = 0; v < H1; v++) sum += h1_buf[v] * W2[v * H2 + u];
            h2_buf[u] = sum > 0.0f ? sum : 0.0f;
        }

        // logits = h2 @ W3 + b3
        for (int c = 0; c < C; c++) {
            float sum = b3[c];
            for (int u = 0; u < H2; u++) sum += h2_buf[u] * W3[u * C + c];
            logits[c] = sum;
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```