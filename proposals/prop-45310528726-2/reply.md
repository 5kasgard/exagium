```hypothesis
Family: label-embed (embedding + 1-hidden-layer MLP). Each of the 16 context positions gets a learned
16-dim embedding (4096 params), concatenated to 256-dim vector, then 128 ReLU units, then 65-class
softmax. Factorizes the 16×256 one-hot input, reducing parameters vs linear (4096×65) while capturing
position-specific character affinities. Should beat label-linear (0.3375) by modeling interactions.
Failure: dev accuracy ≤ 0.3375 or train time > 20 min.
```
```python
"""label-embed: embedding + 1-hidden-layer MLP for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, numpy only. Exports: magic 0x454D4245, d, E, H, C;
float32 emb[d][256][E], W1[E*d][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

E = 16          # embedding dim per position
H = 128         # hidden units
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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    rng = np.random.default_rng(SEED)

    # Embeddings: [d, 256, E]
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    # MLP weights
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    steps_per_epoch = (n + BATCH - 1) // BATCH
    max_epochs = min(MAX_EPOCHS, max(1, math.ceil(SAMPLES / n)))
    t = 0

    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = xb.shape[0]

            # Forward
            # Embedding lookup: [bs, d, E]
            emb_x = emb[np.arange(d)[None, :], xb]  # [bs, d, E]
            flat = emb_x.reshape(bs, d * E)         # [bs, d*E]
            h = flat @ W1 + b1                      # [bs, H]
            h_relu = np.maximum(h, 0)
            logits = h_relu @ W2 + b2               # [bs, C]

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grad_out = (probs / bs).astype(np.float32)  # [bs, C]

            # Backprop
            grad_W2 = h_relu.T @ grad_out + L2 * W2
            grad_b2 = grad_out.sum(axis=0)
            grad_h = grad_out @ W2.T
            grad_h[h <= 0] = 0.0
            grad_W1 = flat.T @ grad_h + L2 * W1
            grad_b1 = grad_h.sum(axis=0)

            # Embedding gradients: [d, 256, E]
            grad_emb = np.zeros_like(emb)
            grad_flat = grad_h @ W1.T  # [bs, d*E]
            grad_emb_seq = grad_flat.reshape(bs, d, E)
            # Scatter-add: for each position j, character xb[:,j], add grad_emb_seq[:,j,:]
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_emb_seq[:, j, :])
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            beta1, beta2 = 0.9, 0.999
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= beta1; m += (1 - beta1) * gr
                v *= beta2; v += (1 - beta2) * gr * gr
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export weights
    # magic, d, E, H, C
    out = struct.pack("<5i", 0x454D4245, d, E, H, C)
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-embed: embedding + 1-hidden-layer MLP. ./infer weights.bin in.rec N
 * Input: d bytes (16). Output: 1 byte class index (0..C-1).
 * Weights format: int32 magic=0x454D4245, d, E, H, C;
 * float32 emb[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d < 1 || E < 1 || H < 1 || C < 1 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L
        + 4L * d * 256 * E
        + 4L * d * E * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *emb = p; p += (long)d * 256 * E;
    const float *W1 = p; p += (long)d * E * H;
    const float *b1 = p; p += H;
    const float *W2 = p; p += (long)H * C;
    const float *b2 = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup and flatten
        for (int h = 0; h < H; h++) hidden[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int u = 0; u < H; u++) {
                float sum = 0.0f;
                for (int v = 0; v < E; v++) sum += e[v] * W1[((long)j * E + v) * H + u];
                hidden[u] += sum;
            }
        }
        // ReLU
        for (int u = 0; u < H; u++) if (hidden[u] < 0.0f) hidden[u] = 0.0f;

        // Output layer
        for (int c = 0; c < C; c++) scores[c] = b2[c];
        for (int u = 0; u < H; u++) {
            float hu = hidden[u];
            if (hu == 0.0f) continue;
            for (int c = 0; c < C; c++) scores[c] += hu * W2[(long)u * C + c];
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```