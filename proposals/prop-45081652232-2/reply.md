```hypothesis
Propose "label-cnn": a 1D convolutional network over learned byte embeddings.
The 16‑byte context is embedded per position (16×256×E), then a single convolutional layer
(kernel=3, K filters, ReLU) with global max‑pooling feeds a linear classifier.
Weight sharing across positions captures local n‑gram patterns more efficiently than
a full linear layer, while the low‑dimensional embeddings keep inference fast.
If dev accuracy does not exceed label‑linear (0.3375), it indicates that either the
receptive field is too small or the embedding dimension is insufficient for this task.
```

```python
"""label-cnn: 1D CNN over per-position byte embeddings for next-char prediction.
train.py card.json train.in train.key weights.bin
Architecture: embed[d][256][E] -> conv1d(kernel=3, K filters, ReLU) -> global max pool -> linear(C).
Adam lr=1e-3, L2=1e-4, batch=64, epochs=20, seed=1. Deterministic, single-threaded.
Weights format (little-endian):
  magic=0x434E4E31 ('CNN1'), d, E, K, C (int32)
  embed[d*256*E] (float32), conv_w[K*3*E], conv_b[K], lin_w[K*C], lin_b[C] (float32)
"""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
E = 32          # embedding dimension
K = 16          # number of conv filters
LR = 1e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 20
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
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # He initialization for embeddings and conv weights
    embed = rng.normal(0, math.sqrt(2.0 / 256), size=(d, 256, E)).astype(np.float32)
    conv_w = rng.normal(0, math.sqrt(2.0 / (3 * E)), size=(K, 3, E)).astype(np.float32)
    conv_b = np.zeros(K, np.float32)
    lin_w = rng.normal(0, math.sqrt(2.0 / K), size=(K, C)).astype(np.float32)
    lin_b = np.zeros(C, np.float32)

    params = [embed, conv_w, conv_b, lin_w, lin_b]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    def forward(Xb):
        # Xb: (b, d) uint8
        b = Xb.shape[0]
        # embeddings: (b, d, E)
        emb = embed[Xb]  # advanced indexing
        # convolution valid: (b, d-2, K)
        conv_out = np.zeros((b, d - 2, K), np.float32)
        for k in range(K):
            # vectorized over batch and positions
            # window shape (b, d-2, 3, E)
            window = emb[:, :d-2, :][:, :, None, :]  # not correct, need sliding
        # We'll do a simpler loop over positions for clarity (d=16 small)
        conv_out = np.zeros((b, d - 2, K), np.float32)
        for p in range(d - 2):
            window = emb[:, p:p+3, :]  # (b, 3, E)
            # (b, 3, E) * (K, 3, E) -> sum over (3,E) -> (b, K)
            conv_out[:, p, :] = np.einsum('bje,kje->bk', window, conv_w) + conv_b
        relu = np.maximum(conv_out, 0)
        pooled = np.max(relu, axis=1)  # (b, K)
        logits = pooled @ lin_w + lin_b  # (b, C)
        return logits, pooled, relu, emb

    def cross_entropy_grad(logits, yb):
        # logits (b, C), yb (b,)
        b = logits.shape[0]
        max_log = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_log)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        probs[np.arange(b), yb] -= 1.0
        return probs / b  # gradient w.r.t logits

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            b = Xb.shape[0]

            # Forward
            logits, pooled, relu, emb = forward(Xb)
            g_logits = cross_entropy_grad(logits, yb)  # (b, C)

            # Backward
            # Linear layer
            g_lin_w = pooled.T @ g_logits + L2 * lin_w
            g_lin_b = np.sum(g_logits, axis=0)
            g_pooled = g_logits @ lin_w.T  # (b, K)

            # Max pooling + ReLU
            g_relu = np.zeros_like(relu)
            # For each filter, find max position per batch
            max_pos = np.argmax(relu, axis=1)  # (b, K)
            for i in range(b):
                for k in range(K):
                    g_relu[i, max_pos[i, k], k] = g_pooled[i, k]

            # Convolution
            g_conv_w = np.zeros_like(conv_w)
            g_conv_b = np.zeros_like(conv_b)
            g_emb = np.zeros_like(emb)
            for p in range(d - 2):
                window = emb[:, p:p+3, :]  # (b, 3, E)
                # g_conv_w: sum over batch of g_relu[:, p, k] * window
                for k in range(K):
                    gk = g_relu[:, p, k:k+1]  # (b, 1)
                    g_conv_w[k] += (gk * window).sum(axis=0)  # (3, E)
                    g_conv_b[k] += gk.sum()
                # g_emb contribution
                # g_emb[:, p:p+3, :] += g_relu[:, p, :, None, None] * conv_w[None, :, :, :]
                # vectorized:
                g_emb[:, p:p+3, :] += np.einsum('bk,kje->bje', g_relu[:, p, :], conv_w)

            # Embedding gradients (accumulate from g_emb)
            # embed shape (d, 256, E), Xb (b, d)
            # For each position j, byte value v, add g_emb[:, j, :] to embed[j, v, :]
            g_embed = np.zeros_like(embed)
            for j in range(d):
                vals = Xb[:, j]  # (b,)
                np.add.at(g_embed[j], vals, g_emb[:, j, :])
            g_embed += L2 * embed

            grads = [g_embed, g_conv_w, g_conv_b, g_lin_w, g_lin_b]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    # magic, d, E, K, C
    header = struct.pack("<5i", 0x434E4E31, d, E, K, C)
    body = (
        embed.astype("<f4").tobytes() +
        conv_w.astype("<f4").tobytes() +
        conv_b.astype("<f4").tobytes() +
        lin_w.astype("<f4").tobytes() +
        lin_b.astype("<f4").tobytes()
    )
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* label-cnn inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes (d=16). Output record: 1 byte (class index).
 * Weights format: magic=0x434E4E31, d, E, K, C (int32), then float32 arrays:
 *   embed[d][256][E], conv_w[K][3][E], conv_b[K], lin_w[K][C], lin_b[C]
 */
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
    if (nw < 20 || hd[0] != 0x434E4E31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], K = hd[3], C = hd[4];
    if (d != 16 || E <= 0 || K <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)K * 3 * E + K + (long)K * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *embed = (const float *)(hd + 5);
    const float *conv_w = embed + (long)d * 256 * E;
    const float *conv_b = conv_w + (long)K * 3 * E;
    const float *lin_w = conv_b + K;
    const float *lin_b = lin_w + (long)K * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_buf = malloc(sizeof(float) * d * E);
    float *conv_out = malloc(sizeof(float) * (d - 2) * K);
    float *pooled = malloc(sizeof(float) * K);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Gather embeddings: for each position j, byte xi[j], copy embed[j][xi[j]][:] into emb_buf
        for (int j = 0; j < d; j++) {
            const float *src = embed + ((long)j * 256 + xi[j]) * E;
            memcpy(emb_buf + (long)j * E, src, E * sizeof(float));
        }

        // Convolution (valid, kernel=3)
        for (int p = 0; p < d - 2; p++) {
            for (int kk = 0; kk < K; kk++) {
                float sum = conv_b[kk];
                const float *w = conv_w + ((long)kk * 3 * E);
                for (int kj = 0; kj < 3; kj++) {
                    const float *emb = emb_buf + (long)(p + kj) * E;
                    for (int e = 0; e < E; e++) {
                        sum += emb[e] * w[kj * E + e];
                    }
                }
                // ReLU
                conv_out[(long)p * K + kk] = sum > 0.0f ? sum : 0.0f;
            }
        }

        // Global max pooling over positions
        for (int kk = 0; kk < K; kk++) {
            float maxv = -1e30f;
            for (int p = 0; p < d - 2; p++) {
                float v = conv_out[(long)p * K + kk];
                if (v > maxv) maxv = v;
            }
            pooled[kk] = maxv;
        }

        // Linear layer
        for (int c = 0; c < C; c++) {
            float sum = lin_b[c];
            for (int kk = 0; kk < K; kk++) {
                sum += pooled[kk] * lin_w[(long)kk * C + c];
            }
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (logits[c] > logits[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }

    free(emb_buf); free(conv_out); free(pooled); free(logits);
    if (x) free(x);
    free(hd);
    return 0;
}
```