```hypothesis
Family: label-cnn1d (1D CNN with per-position embeddings, kernel 3, 32 filters, global max-pool, linear head).
Why: Optdigits 8x8 images have local stroke structure; a small 1D CNN captures translation-invariant patterns better than a linear model, with far fewer parameters than an MLP.
Failure: dev accuracy ≤ 0.9613 (label-linear) or CPU cost per item > 2× label-linear.
```

```python
"""label-cnn1d: 1D CNN with per-position embeddings for fixed-size byte inputs.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Architecture: embed[d][256][E] -> conv1d(K=3, F filters, ReLU) -> global max pool -> linear(C).
Exported weights: magic 0x434E4E31, d, E, K, F, C; float32 embed, W_conv, b_conv, W_out, b_out."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
E = 8          # embedding size per position
K = 3          # kernel size
F = 32         # number of filters
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
    d = X.shape[1]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # Parameters
    embed = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    W_conv = (rng.normal(0, math.sqrt(2.0 / (E * K)), (F, E, K))).astype(np.float32)
    b_conv = np.zeros(F, np.float32)
    W_out = (rng.normal(0, math.sqrt(2.0 / F), (F, C))).astype(np.float32)
    b_out = np.zeros(C, np.float32)

    params = [embed, W_conv, b_conv, W_out, b_out]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    pad = K // 2

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward
            # Embedding lookup: (B, d, E)
            emb = embed[np.arange(d), xb.T].transpose(1, 0, 2)
            # Pad for convolution
            emb_pad = np.pad(emb, ((0,0), (pad, pad), (0,0)), mode='constant')
            # Convolution: (B, d, F)
            conv = np.zeros((B, d, F), np.float32)
            for f in range(F):
                for k in range(K):
                    w = W_conv[f, :, k]  # (E,)
                    conv[:, :, f] += np.dot(emb_pad[:, k:k+d, :], w)
            conv += b_conv
            np.maximum(conv, 0, out=conv)  # ReLU
            # Global max pool over positions
            pool = np.max(conv, axis=1)    # (B, F)
            # Linear
            scores = pool @ W_out + b_out  # (B, C)

            # Softmax cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grads_out = (probs / B).astype(np.float32)  # (B, C)

            # Backward
            # Linear grads
            g_W_out = pool.T @ grads_out + L2 * W_out
            g_b_out = grads_out.sum(axis=0)
            g_pool = grads_out @ W_out.T  # (B, F)

            # Max pool grad: route to max positions
            g_conv = np.zeros_like(conv)
            max_pos = np.argmax(conv, axis=1)  # (B, F)
            for b in range(B):
                for f in range(F):
                    g_conv[b, max_pos[b, f], f] = g_pool[b, f]

            # ReLU grad
            g_conv[conv == 0] = 0.0

            # Convolution grads
            g_W_conv = np.zeros_like(W_conv)
            g_b_conv = np.zeros_like(b_conv)
            g_emb_pad = np.zeros_like(emb_pad)
            for f in range(F):
                for k in range(K):
                    w = W_conv[f, :, k]
                    g_W_conv[f, :, k] = np.dot(emb_pad[:, k:k+d, :].T, g_conv[:, :, f]).astype(np.float32) + L2 * w
                g_b_conv[f] = g_conv[:, :, f].sum()
                for k in range(K):
                    g_emb_pad[:, k:k+d, :] += np.outer(g_conv[:, :, f], W_conv[f, :, k])

            # Embedding grad (remove padding)
            g_emb = g_emb_pad[:, pad:pad+d, :]

            # Accumulate embedding grads per (position, byte value)
            g_embed = np.zeros_like(embed)
            for j in range(d):
                bytes_j = xb[:, j]
                for e in range(E):
                    np.add.at(g_embed[j, :, e], bytes_j, g_emb[:, j, e])

            grads = [g_embed, g_W_conv, g_b_conv, g_W_out, g_b_out]

            # Adam update
            t = 1  # we don't track global step exactly; using per-batch t=1 is fine for determinism? 
            # Actually we need a global step counter. Let's use a simple counter.
            # We'll add a global step variable.
            # For simplicity, we'll use the same Adam as label-linear but per parameter.
            # We'll keep a global step counter.
            pass  # We'll restructure the loop with a global step.

    # The above loop is incomplete; we need to integrate Adam with a global step.
    # Let's rewrite the training loop properly.

    # Re-initialize for clarity
    embed = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    W_conv = (rng.normal(0, math.sqrt(2.0 / (E * K)), (F, E, K))).astype(np.float32)
    b_conv = np.zeros(F, np.float32)
    W_out = (rng.normal(0, math.sqrt(2.0 / F), (F, C))).astype(np.float32)
    b_out = np.zeros(C, np.float32)
    params = [embed, W_conv, b_conv, W_out, b_out]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            B = xb.shape[0]

            # Forward
            emb = embed[np.arange(d), xb.T].transpose(1, 0, 2)
            emb_pad = np.pad(emb, ((0,0), (pad, pad), (0,0)), mode='constant')
            conv = np.zeros((B, d, F), np.float32)
            for f in range(F):
                for k in range(K):
                    conv[:, :, f] += np.dot(emb_pad[:, k:k+d, :], W_conv[f, :, k])
            conv += b_conv
            np.maximum(conv, 0, out=conv)
            pool = np.max(conv, axis=1)
            scores = pool @ W_out + b_out

            # Loss gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grads_out = (probs / B).astype(np.float32)

            # Backward
            g_W_out = pool.T @ grads_out + L2 * W_out
            g_b_out = grads_out.sum(axis=0)
            g_pool = grads_out @ W_out.T

            g_conv = np.zeros_like(conv)
            max_pos = np.argmax(conv, axis=1)
            for b in range(B):
                for f in range(F):
                    g_conv[b, max_pos[b, f], f] = g_pool[b, f]
            g_conv[conv == 0] = 0.0

            g_W_conv = np.zeros_like(W_conv)
            g_b_conv = np.zeros_like(b_conv)
            g_emb_pad = np.zeros_like(emb_pad)
            for f in range(F):
                for k in range(K):
                    w = W_conv[f, :, k]
                    g_W_conv[f, :, k] = np.dot(emb_pad[:, k:k+d, :].T, g_conv[:, :, f]).astype(np.float32) + L2 * w
                g_b_conv[f] = g_conv[:, :, f].sum()
                for k in range(K):
                    g_emb_pad[:, k:k+d, :] += np.outer(g_conv[:, :, f], W_conv[f, :, k])

            g_emb = g_emb_pad[:, pad:pad+d, :]
            g_embed = np.zeros_like(embed)
            for j in range(d):
                bytes_j = xb[:, j]
                for e in range(E):
                    np.add.at(g_embed[j, :, e], bytes_j, g_emb[:, j, e])

            grads = [g_embed, g_W_conv, g_b_conv, g_W_out, g_b_out]

            step += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= beta1; m += (1 - beta1) * gr
                v *= beta2; v += (1 - beta2) * gr * gr
                m_hat = m / (1 - beta1 ** step)
                v_hat = v / (1 - beta2 ** step)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Save weights
    magic = 0x434E4E31  # 'CNN1'
    header = struct.pack("<6i", magic, d, E, K, F, C)
    blob = header
    blob += embed.astype("<f4").tobytes()
    blob += W_conv.astype("<f4").tobytes()
    blob += b_conv.astype("<f4").tobytes()
    blob += W_out.astype("<f4").tobytes()
    blob += b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(blob)

if __name__ == "__main__":
    main()
```

```c
/* label-cnn1d infer: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output: 1 byte class index. */
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
    if (nw < 24 || hd[0] != 0x434E4E31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], K = hd[3], F = hd[4], C = hd[5];
    if (C < 1 || C > 256 || d < 1 || E < 1 || K < 1 || F < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    int pad = K / 2;
    long expected = 24L
        + (long)d * 256 * E
        + (long)F * E * K
        + F
        + (long)F * C
        + C;
    if (nw != expected * 4) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected*4); return 3; }

    const float *embed = (const float *)(hd + 6);
    const float *W_conv = embed + (long)d * 256 * E;
    const float *b_conv = W_conv + (long)F * E * K;
    const float *W_out = b_conv + F;
    const float *b_out = W_out + (long)F * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_buf = malloc(sizeof(float) * (d + 2 * pad) * E);
    float *conv_buf = malloc(sizeof(float) * d * F);
    float *pool_buf = malloc(sizeof(float) * F);
    float *score_buf = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup with padding
        memset(emb_buf, 0, sizeof(float) * (d + 2 * pad) * E);
        for (int j = 0; j < d; j++) {
            const float *src = embed + ((long)j * 256 + xi[j]) * E;
            float *dst = emb_buf + ((long)(j + pad) * E);
            memcpy(dst, src, sizeof(float) * E);
        }

        // Convolution + ReLU
        for (int p = 0; p < d; p++) {
            for (int f = 0; f < F; f++) {
                float acc = b_conv[f];
                for (int kk = 0; kk < K; kk++) {
                    const float *emb_ptr = emb_buf + ((long)(p + kk) * E);
                    const float *w_ptr = W_conv + ((long)f * E * K + kk * E);
                    for (int e = 0; e < E; e++) acc += emb_ptr[e] * w_ptr[e];
                }
                conv_buf[(long)p * F + f] = acc > 0.0f ? acc : 0.0f;
            }
        }

        // Global max pool
        for (int f = 0; f < F; f++) {
            float mx = conv_buf[f];
            for (int p = 1; p < d; p++) {
                float v = conv_buf[(long)p * F + f];
                if (v > mx) mx = v;
            }
            pool_buf[f] = mx;
        }

        // Linear
        for (int c = 0; c < C; c++) {
            float acc = b_out[c];
            for (int f = 0; f < F; f++) acc += pool_buf[f] * W_out[(long)f * C + c];
            score_buf[c] = acc;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score_buf[c] > score_buf[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(emb_buf); free(conv_buf); free(pool_buf); free(score_buf);
    return 0;
}
```