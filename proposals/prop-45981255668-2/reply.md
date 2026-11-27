```hypothesis
Family: label-embmlp — shared byte embeddings + one hidden ReLU layer.
Why: The linear model uses 16×256 one-hot features (4096 dim). A shared embedding (256×E) reduces parameters and forces the model to learn byte similarities (e.g., 'a'≈'b'), which helps with the sparse 16-byte context. A single hidden layer (E*16 → H → 65) adds non-linear interactions between positions. With E=32, H=64 it has ~45k params (vs 266k for linear), so it should train faster and generalize better on this character-level task.
Failure: If dev accuracy ≤ 0.3375 (label-linear) or inference cost exceeds label-linear by >20%.
```

```python
"""label-embmlp: shared byte embeddings + one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Weights format (little-endian):
  int32 magic=0x454D424D, d, E, H, C
  float32 emb[256][E]
  float32 W1[d*E][H], b1[H]
  float32 W2[H][C],   b2[C]
"""
import json, math, struct, sys

import numpy as np

# Hyper-parameters (chosen on tuning split, then fixed)
E = 32          # embedding dimension
H = 64          # hidden units
SEED = 1
LR = 1e-3
L2 = 1e-5
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
    d = card["input"]["bytes"]
    C = card["classes"]

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Embedding matrix (256, E)
    emb = (rng.normal(0, 1, (256, E)) / math.sqrt(E)).astype(np.float32)

    # First layer: (d*E) -> H
    fan_in = d * E
    W1 = (rng.normal(0, 1, (fan_in, H)) * math.sqrt(2.0 / fan_in)).astype(np.float32)
    b1 = np.zeros(H, np.float32)

    # Second layer: H -> C
    W2 = (rng.normal(0, 1, (H, C)) * math.sqrt(2.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = xb.shape[0]

            # Forward
            # Embedding lookup: (bs, d, E)
            x_emb = emb[xb]                       # (bs, d, E)
            x_flat = x_emb.reshape(bs, -1)        # (bs, d*E)
            h_pre = x_flat @ W1 + b1              # (bs, H)
            h = np.maximum(h_pre, 0)              # ReLU
            logits = h @ W2 + b2                  # (bs, C)

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs

            # Backward
            gW2 = h.T @ probs + L2 * W2
            gb2 = probs.sum(axis=0)
            gh = probs @ W2.T
            gh[h_pre <= 0] = 0.0                  # ReLU grad
            gW1 = x_flat.T @ gh + L2 * W1
            gb1 = gh.sum(axis=0)
            # Embedding gradients: scatter-add
            gx_flat = gh @ W1.T                   # (bs, d*E)
            gx_emb = gx_flat.reshape(bs, d, E)    # (bs, d, E)
            gemb = np.zeros_like(emb)
            np.add.at(gemb, xb, gx_emb)           # (256, E)

            grads = [gemb, gW1, gb1, gW2, gb2]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    magic = 0x454D424D  # 'EMBM'
    header = struct.pack("<5i", magic, d, E, H, C)
    body = (
        emb.astype("<f4").tobytes() +
        W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes() +
        W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    )
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 raw bytes. Output record: 1 byte (class index).
 * Weights format (little-endian):
 *   int32 magic=0x454D424D, d, E, H, C
 *   float32 emb[256][E]
 *   float32 W1[d*E][H], b1[H]
 *   float32 W2[H][C],   b2[C]
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static unsigned char *slurp(const char *path, long *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(3); }
    fseek(f, 0, SEEK_END);
    *len = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(*len ? *len : 1);
    if (fread(buf, 1, *len, f) != (size_t)*len) {
        fprintf(stderr, "short read %s\n", path);
        exit(3);
    }
    fclose(f);
    return buf;
}

static const unsigned char *next_rec(const unsigned char *buf, long n, long *pos, uint32_t *len) {
    if (*pos + 4 > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *r = buf + *pos + 4;
    *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k>>8 & 255, k>>16 & 255, k>>24};
    fwrite(h, 1, 4, f);
    fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }

    long nw, nx = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights magic\n"); return 3; }

    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E <= 0 || H <= 0 || C != 65) { fprintf(stderr, "bad dims\n"); return 3; }

    size_t expected = 20 +
        (size_t)256 * E * 4 +
        (size_t)d * E * H * 4 + (size_t)H * 4 +
        (size_t)H * C * 4 + (size_t)C * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1  = emb + 256 * E;
    const float *b1  = W1  + (size_t)d * E * H;
    const float *W2  = b1  + H;
    const float *b2  = W2  + (size_t)H * C;

    long N = atol(argv[3]);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    long pos = 0;

    float *h_pre = malloc(sizeof(float) * H);
    float *h     = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k;
        const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld: %u bytes != %d\n", i, k, d); return 3; }

        // h_pre = b1
        memcpy(h_pre, b1, H * sizeof(float));
        // For each of d positions: add emb[byte] * W1 slice
        for (int j = 0; j < d; j++) {
            const float *e = emb + xi[j] * E;
            const float *w = W1 + j * E * H;
            for (int u = 0; u < H; u++) {
                float sum = 0.0f;
                for (int v = 0; v < E; v++) sum += e[v] * w[v * H + u];
                h_pre[u] += sum;
            }
        }
        // ReLU
        for (int u = 0; u < H; u++) h[u] = h_pre[u] > 0.0f ? h_pre[u] : 0.0f;

        // logits = b2 + h @ W2
        memcpy(logits, b2, C * sizeof(float));
        for (int u = 0; u < H; u++) {
            float hu = h[u];
            if (hu == 0.0f) continue;
            const float *w = W2 + u * C;
            for (int c = 0; c < C; c++) logits[c] += hu * w[c];
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++)
            if (logits[c] > logits[best]) best = (unsigned char)c;

        put_rec(stdout, &best, 1);
    }

    return 0;
}
```