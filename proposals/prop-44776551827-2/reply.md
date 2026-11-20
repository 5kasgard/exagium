```hypothesis
Family: label-emb-mlp — per-position byte embeddings summed, then a one-hidden-layer MLP.
Why: The linear model treats each position independently; summing learned embeddings lets the model
share statistical strength across positions while a small ReLU layer captures non-linear interactions.
With E=32, H=32 the model has ~1.05M parameters (similar to label-linear) but adds non-linearity.
If dev accuracy ≤ 0.3375 (label-linear) the non-linearity does not help for this context length.
```
```python
"""label-emb-mlp: per-position embeddings + one ReLU layer for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Exports: magic 0x454D424D, d, E, H, C; float32 emb[d][256][E], W1[E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

E = 32          # embedding dimension per position
H = 32          # hidden layer size
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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # embeddings: [d, 256, E]
    emb = (rng.normal(0, 0.02, (d, 256, E))).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(2.0 / E), (E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # [B, d]
            yb = y[idx]          # [B]
            B = xb.shape[0]

            # forward: sum embeddings
            # emb_sum[b, e] = sum_j emb[j, xb[b,j], e]
            emb_sum = np.zeros((B, E), np.float32)
            for j in range(d):
                emb_sum += emb[j, xb[:, j]]      # [B, E]

            # hidden layer
            hidden = emb_sum @ W1 + b1           # [B, H]
            hidden_relu = np.maximum(hidden, 0)
            logits = hidden_relu @ W2 + b2       # [B, C]

            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            probs /= B

            # gradients
            gW2 = hidden_relu.T @ probs + L2 * W2
            gb2 = probs.sum(axis=0)
            ghidden = probs @ W2.T
            ghidden[hidden <= 0] = 0.0
            gW1 = emb_sum.T @ ghidden + L2 * W1
            gb1 = ghidden.sum(axis=0)
            # gradient wrt emb_sum
            g_emb_sum = ghidden @ W1.T           # [B, E]

            # gradient wrt embeddings
            gemb = np.zeros_like(emb)
            for j in range(d):
                # accumulate g_emb_sum[b, e] into gemb[j, xb[b,j], e]
                np.add.at(gemb[j], xb[:, j], g_emb_sum)

            grads = [gemb, gW1, gb1, gW2, gb2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # pack weights
    # magic, d, E, H, C
    out = struct.pack("<5i", 0x454D424D, d, E, H, C)
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index.
 * Weights format: magic 0x454D424D, int32 d, E, H, C; float32 emb[d][256][E], W1[E][H], b1[H], W2[H][C], b2[C]. */
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
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad header dims\n"); return 3; }
    size_t expected = 20
        + (size_t)d * 256 * E * 4
        + (size_t)E * H * 4
        + (size_t)H * 4
        + (size_t)H * C * 4
        + (size_t)C * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (size_t)d * 256 * E;
    const float *b1 = W1 + (size_t)E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (size_t)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // sum embeddings
        memset(emb_sum, 0, E * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((size_t)j * 256 + xi[j]) * E;
            for (int e_idx = 0; e_idx < E; e_idx++) emb_sum[e_idx] += e[e_idx];
        }

        // hidden layer with ReLU
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int e = 0; e < E; e++) sum += emb_sum[e] * W1[(size_t)e * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(size_t)h * C + c];
            logits[c] = sum;
        }

        // argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(emb_sum); free(hidden); free(logits);
    return 0;
}
```