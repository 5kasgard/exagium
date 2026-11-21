```hypothesis
Family: label-embmlp (embedding + one-hidden-layer MLP). Treat all 14 byte features as categorical with 256 values, learn a small per-feature embedding (E=4), concatenate to 56-d vector, then a ReLU hidden layer (H=32) and 2-class output. This captures non-linear feature interactions cheaper than one-hot linear (label-linear) or full one-hot MLP (label-mlp16). Expect dev accuracy >0.855 with lower Ir/item than label-mlp16. Failure: accuracy ≤ label-linear or Ir/item ≥ label-mlp16.
```
```python
import json, math, struct, sys
import numpy as np

E = 4          # embedding dim per feature
H = 32         # hidden layer size
SEED = 1
LR = 1e-3
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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]          # 14
    C = card["classes"]     # 2
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Embeddings: d x 256 x E
    emb = rng.normal(0, 0.01, size=(d, 256, E)).astype(np.float32)

    # Hidden layer: (d*E) x H
    W1 = rng.uniform(-1, 1, (d * E, H)).astype(np.float32) * math.sqrt(6.0 / (d * E))
    b1 = np.zeros(H, np.float32)

    # Output layer: H x C
    W2 = rng.uniform(-1, 1, (H, C)).astype(np.float32) * math.sqrt(6.0 / H)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def forward(batch_idx):
        xb = X[batch_idx]                 # (B, d)
        # Embedding lookup: (B, d, E)
        e = emb[:, xb.T].transpose(1, 0, 2)   # (B, d, E)
        z = e.reshape(len(batch_idx), d * E)  # (B, d*E)
        a = z @ W1 + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
        return z, a, h, logits

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return (probs / len(yy)).astype(np.float32)

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            yy = y[idx]
            z, a, h, logits = forward(idx)
            g = softmax_grad(logits, yy)                 # (B, C)

            # Output layer grads
            gW2 = h.T @ g + L2 * W2
            gb2 = g.sum(0)

            # Hidden layer grads
            gh = g @ W2.T * (a > 0)                      # (B, H)

            # W1, b1 grads
            gW1 = z.T @ gh + L2 * W1
            gb1 = gh.sum(0)

            # Embedding grads: (d, 256, E)
            # gh @ W1.T -> (B, d*E) -> reshape (B, d, E)
            gez = (gh @ W1.T).reshape(len(idx), d, E)    # (B, d, E)
            # scatter-add into gemb
            gemb = np.zeros_like(emb)
            xb = X[idx]                                  # (B, d)
            np.add.at(gemb, (np.arange(d)[:, None], xb.T), gez.transpose(1, 0, 2))
            gemb += L2 * emb

            grads = [gemb, gW1, gb1, gW2, gb2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    # Header: magic, d, E, H, C (5 int32)
    magic = 0x4D4C5031  # 'MLP1'
    header = struct.pack("<5i", magic, d, E, H, C)
    # Embeddings: d * 256 * E floats (row-major: feature, value, dim)
    emb_bytes = emb.astype("<f4").tobytes()
    # W1: (d*E) x H
    W1_bytes = W1.astype("<f4").tobytes()
    # b1: H
    b1_bytes = b1.astype("<f4").tobytes()
    # W2: H x C
    W2_bytes = W2.astype("<f4").tobytes()
    # b2: C
    b2_bytes = b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + emb_bytes + W1_bytes + b1_bytes + W2_bytes + b2_bytes)

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
    if (nw < 20 || hd[0] != 0x4D4C5031) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C != 2 || d != 14 || E != 4 || H != 32) { fprintf(stderr, "unexpected dims\n"); return 3; }

    size_t expected = 20 +
        (size_t)d * 256 * E * 4 +
        (size_t)d * E * H * 4 +
        (size_t)H * 4 +
        (size_t)H * C * 4 +
        (size_t)C * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld vs %zu\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (size_t)d * 256 * E;
    const float *b1 = W1 + (size_t)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (size_t)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings into a flat vector of size d*E
        float concat[14 * 4];  // d=14, E=4
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((size_t)j * 256 + xi[j]) * E;
            for (int ee = 0; ee < E; ee++) concat[j * E + ee] = e[ee];
        }

        // Hidden layer: ReLU(concat @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k = 0; k < d * E; k++) sum += concat[k] * W1[(size_t)k * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(size_t)h * C + c];
            logits[c] = sum;
        }

        unsigned char best = logits[1] > logits[0] ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```