```hypothesis
Family: label-emb (per-position embeddings summed, one ReLU hidden layer, linear output).
Why: The linear model uses one-hot per position (4096 features) but cannot capture interactions between positions. Learned embeddings (32-d) compress each position's 256 values into a dense vector, and summing them preserves position-specific information while allowing a hidden layer (64 units) to model pairwise interactions. This should improve accuracy over linear with similar parameter count (~137k vs 266k) and fast inference (fewer memory accesses).
Failure: If dev accuracy <= 0.3375 (label-linear) or inference cost exceeds label-linear by >20%.
```

```python
"""label-emb: per-position embeddings summed, one ReLU hidden layer, linear output.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split, then trained on all train)
EMBED_DIM = 32
HIDDEN_DIM = 64
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 20

MAGIC = 0x454D424C  # "LBME" little-endian


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
    # Load training data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # Embeddings: d x 256 x EMBED_DIM
    embed = (rng.normal(0, 1/math.sqrt(EMBED_DIM), (d, 256, EMBED_DIM))).astype(np.float32)
    # Hidden layer: EMBED_DIM -> HIDDEN_DIM
    W_h = (rng.normal(0, math.sqrt(2.0/EMBED_DIM), (EMBED_DIM, HIDDEN_DIM))).astype(np.float32)
    b_h = np.zeros(HIDDEN_DIM, np.float32)
    # Output layer: HIDDEN_DIM -> C
    W_o = (rng.normal(0, math.sqrt(2.0/HIDDEN_DIM), (HIDDEN_DIM, C))).astype(np.float32)
    b_o = np.zeros(C, np.float32)

    params = [embed, W_h, b_h, W_o, b_o]
    # Adam state: m and v for each parameter array
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward
            # Embedding lookup and sum: (B, EMBED_DIM)
            emb_sum = np.zeros((B, EMBED_DIM), np.float32)
            for p in range(d):
                emb_sum += embed[p, xb[:, p], :]   # (B, EMBED_DIM)

            hidden_pre = emb_sum @ W_h + b_h       # (B, HIDDEN_DIM)
            hidden = np.maximum(hidden_pre, 0)     # ReLU
            logits = hidden @ W_o + b_o            # (B, C)

            # Softmax cross-entropy gradient
            max_logits = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            probs /= B                               # (B, C)

            # Backward
            # Output layer
            g_W_o = hidden.T @ probs + L2 * W_o      # (HIDDEN_DIM, C)
            g_b_o = np.sum(probs, axis=0)            # (C,)
            g_hidden = probs @ W_o.T                 # (B, HIDDEN_DIM)
            # ReLU
            g_hidden[hidden_pre <= 0] = 0
            # Hidden layer
            g_W_h = emb_sum.T @ g_hidden + L2 * W_h  # (EMBED_DIM, HIDDEN_DIM)
            g_b_h = np.sum(g_hidden, axis=0)         # (HIDDEN_DIM,)
            g_emb_sum = g_hidden @ W_h.T             # (B, EMBED_DIM)
            # Embeddings: scatter-add per position
            g_embed = np.zeros_like(embed)
            for p in range(d):
                # g_embed[p] shape (256, EMBED_DIM)
                np.add.at(g_embed[p], xb[:, p], g_emb_sum)

            grads = [g_embed, g_W_h, g_b_h, g_W_o, g_b_o]

            t += 1
            beta1, beta2 = 0.9, 0.999
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= beta1; m += (1 - beta1) * gr
                v *= beta2; v += (1 - beta2) * gr * gr
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    # Header: magic, d, EMBED_DIM, HIDDEN_DIM, C
    header = struct.pack("<5i", MAGIC, d, EMBED_DIM, HIDDEN_DIM, C)
    # Embeddings: d * 256 * EMBED_DIM floats (row-major: position, value, dim)
    embed_flat = embed.astype("<f4").tobytes()
    # Hidden weights: EMBED_DIM * HIDDEN_DIM
    W_h_bytes = W_h.astype("<f4").tobytes()
    b_h_bytes = b_h.astype("<f4").tobytes()
    # Output weights: HIDDEN_DIM * C
    W_o_bytes = W_o.astype("<f4").tobytes()
    b_o_bytes = b_o.astype("<f4").tobytes()

    out = header + embed_flat + W_h_bytes + b_h_bytes + W_o_bytes + b_o_bytes
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "weights too small\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x454D424C) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E <= 0 || H <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }

    // Expected size: header(20) + embed(d*256*E) + W_h(E*H) + b_h(H) + W_o(H*C) + b_o(C)
    long expected = 20
        + 4L * d * 256 * E
        + 4L * E * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    float *ptr = (float *)(wb + 20);
    float *embed = ptr; ptr += d * 256 * E;
    float *W_h = ptr; ptr += E * H;
    float *b_h = ptr; ptr += H;
    float *W_o = ptr; ptr += H * C;
    float *b_o = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *emb_sum = malloc(E * sizeof(float));
    float *hidden = malloc(H * sizeof(float));
    float *scores = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Zero emb_sum
        for (int e = 0; e < E; e++) emb_sum[e] = 0.0f;
        // Sum embeddings for each position
        for (int p = 0; p < d; p++) {
            const float *eptr = embed + ((long)p * 256 + xi[p]) * E;
            for (int e = 0; e < E; e++) emb_sum[e] += eptr[e];
        }

        // Hidden layer: ReLU(emb_sum * W_h + b_h)
        for (int h = 0; h < H; h++) {
            float sum = b_h[h];
            const float *wptr = W_h + h; // column-major? W_h is (E, H) row-major in Python: W_h[e, h]
            // In Python: W_h shape (E, H), stored row-major. So W_h[e*H + h] for e in 0..E-1.
            for (int e = 0; e < E; e++) sum += emb_sum[e] * wptr[e * H];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: hidden * W_o + b_o
        for (int c = 0; c < C; c++) {
            float sum = b_o[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W_o[h * C + c];
            scores[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(emb_sum); free(hidden); free(scores);
    free(wb); if (x) free(x);
    return 0;
}
```