```hypothesis
A 1D CNN on position-specific byte embeddings captures local n-gram patterns (e.g., "th", "ing") that are critical for character-level language modeling, which the linear model misses because it only sums independent position contributions. The model uses embeddings (16×256×D), a ReLU convolution (kernel K, F filters), global max pooling, and a linear classifier. It should exceed label-linear's 0.3375 dev accuracy while keeping inference cost moderate (few thousand ops/item). Failure to beat label-linear would indicate that local interactions beyond position-wise independent effects are not learnable with this capacity or that the linear model already captures the dominant signal.
```
```python
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
D = 16          # embedding dimension
K = 3           # convolution kernel size
F = 32          # number of filters
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

def softmax_cross_entropy_grad(logits, y):
    # logits: (B, C), y: (B,)
    B = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits, dtype=np.float32)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(B), y] -= 1.0
    return probs / B  # (B, C)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]  # 16
    C = card["classes"]         # 65
    n = X.shape[0]

    # Split tuning (last 20%) for hyperparameter selection (here fixed, but structure allows grid search)
    tuning_start = int(n * 0.8)
    X_tune, y_tune = X[tuning_start:], y[tuning_start:]
    X_train, y_train = X[:tuning_start], y[:tuning_start]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # Embedding: (d, 256, D)
    emb = (rng.normal(0, 0.01, (d, 256, D))).astype(np.float32)
    # Conv weights: (F, K, D)
    conv_w = (rng.normal(0, math.sqrt(2.0 / (K * D)), (F, K, D))).astype(np.float32)
    conv_b = np.zeros(F, dtype=np.float32)
    # FC weights: (F, C)
    fc_w = (rng.normal(0, math.sqrt(2.0 / F), (F, C))).astype(np.float32)
    fc_b = np.zeros(C, dtype=np.float32)

    params = [emb, conv_w, conv_b, fc_w, fc_b]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    # Precompute embedding indices for speed
    # For each batch we will gather embeddings: (B, d, D)
    # We'll implement convolution as im2col + gemm for simplicity (small sizes)
    out_len = d - K + 1  # 14

    def forward(X_batch):
        # X_batch: (B, d) uint8
        B = X_batch.shape[0]
        # Embedding lookup: (B, d, D)
        emb_seq = emb[X_batch]  # numpy advanced indexing: (B, d, D)
        # Convolution: im2col -> (B, out_len, K*D)
        col = np.lib.stride_tricks.sliding_window_view(emb_seq, (K, D), axis=(1,2)).reshape(B, out_len, K*D)
        # (B, out_len, K*D) @ (K*D, F) -> (B, out_len, F)
        conv_out = np.matmul(col, conv_w.reshape(K*D, F).T) + conv_b  # (B, out_len, F)
        # ReLU
        conv_out = np.maximum(conv_out, 0, dtype=np.float32)
        # Global max pooling over out_len -> (B, F)
        pooled = conv_out.max(axis=1)
        # FC
        logits = np.matmul(pooled, fc_w) + fc_b  # (B, C)
        return logits, (emb_seq, col, conv_out, pooled)

    def backward(grad_logits, cache, X_batch):
        # grad_logits: (B, C)
        emb_seq, col, conv_out, pooled = cache
        B = X_batch.shape[0]

        # FC backward
        grad_fc_w = np.matmul(pooled.T, grad_logits)  # (F, C)
        grad_fc_b = grad_logits.sum(axis=0)
        grad_pooled = np.matmul(grad_logits, fc_w.T)  # (B, F)

        # Max pooling backward: route gradient to max positions
        grad_conv_out = np.zeros_like(conv_out)
        max_idx = (conv_out == pooled[:, None, :])  # (B, out_len, F)
        # For each batch and filter, there may be multiple max positions; split gradient equally
        count = max_idx.sum(axis=1, keepdims=True)  # (B, 1, F)
        grad_conv_out[max_idx] = np.repeat(grad_pooled / count, max_idx.sum(axis=1).flatten(), axis=0).flatten()

        # ReLU backward
        grad_conv_out[conv_out <= 0] = 0

        # Convolution backward
        # grad_conv_out: (B, out_len, F)
        # col: (B, out_len, K*D)
        grad_conv_w = np.matmul(col.transpose(0, 2, 1).reshape(-1, out_len).T, grad_conv_out.reshape(-1, F)).T  # (F, K*D)
        grad_conv_w = grad_conv_w.reshape(F, K, D)
        grad_conv_b = grad_conv_out.sum(axis=(0,1))
        grad_col = np.matmul(grad_conv_out, conv_w.reshape(K*D, F))  # (B, out_len, K*D)
        grad_col = grad_col.reshape(B, out_len, K, D)

        # col backward to emb_seq (sliding window view)
        grad_emb_seq = np.zeros_like(emb_seq)
        for i in range(out_len):
            grad_emb_seq[:, i:i+K, :] += grad_col[:, i, :, :]

        # Embedding backward: scatter-add
        grad_emb = np.zeros_like(emb)
        np.add.at(grad_emb, (np.arange(d)[:, None], X_batch.T), grad_emb_seq.transpose(1, 0, 2))

        # L2 regularization gradients
        grad_emb += L2 * emb
        grad_conv_w += L2 * conv_w
        grad_fc_w += L2 * fc_w

        return [grad_emb, grad_conv_w, grad_conv_b, grad_fc_w, grad_fc_b]

    # Training loop
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            X_batch, y_batch = X[idx], y[idx]
            logits, cache = forward(X_batch)
            grad_logits = softmax_cross_entropy_grad(logits, y_batch)
            grads = backward(grad_logits, cache, X_batch)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # After training, write weights.bin
    # Format: magic, d, D, K, F, C (all int32), then flat arrays in order:
    # emb (d*256*D), conv_w (F*K*D), conv_b (F), fc_w (F*C), fc_b (C) -- all float32 little-endian
    magic = 0x434E4E31  # 'CNN1'
    header = struct.pack("<6i", magic, d, D, K, F, C)
    body = (emb.astype("<f4").tobytes() +
            conv_w.astype("<f4").tobytes() +
            conv_b.astype("<f4").tobytes() +
            fc_w.astype("<f4").tobytes() +
            fc_b.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + body)

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
    if (nw < 24 || hd[0] != 0x434E4E31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], K = hd[3], F = hd[4], C = hd[5];
    if (C < 1 || C > 256 || d != 16 || K != 3 || D != 16 || F != 32) {
        fprintf(stderr, "unsupported architecture\n"); return 3;
    }
    long expected = 24 + 4L * ((long)d * 256 * D + (long)F * K * D + F + (long)F * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 6);
    const float *conv_w = emb + (long)d * 256 * D;
    const float *conv_b = conv_w + (long)F * K * D;
    const float *fc_w = conv_b + F;
    const float *fc_b = fc_w + (long)F * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_seq = malloc((size_t)d * D * sizeof(float));
    float *conv_out = malloc((size_t)(d - K + 1) * F * sizeof(float));
    float *pooled = malloc((size_t)F * sizeof(float));
    float *logits = malloc((size_t)C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup
        for (int p = 0; p < d; p++) {
            const float *e = emb + ((long)p * 256 + xi[p]) * D;
            memcpy(emb_seq + (long)p * D, e, D * sizeof(float));
        }

        // Convolution + ReLU
        int out_len = d - K + 1;
        for (int f = 0; f < F; f++) {
            const float *w = conv_w + (long)f * K * D;
            float b = conv_b[f];
            for (int p = 0; p < out_len; p++) {
                float sum = b;
                for (int kk = 0; kk < K; kk++) {
                    const float *e = emb_seq + (long)(p + kk) * D;
                    const float *wk = w + (long)kk * D;
                    for (int dd = 0; dd < D; dd++) sum += e[dd] * wk[dd];
                }
                conv_out[(long)p * F + f] = sum > 0 ? sum : 0;
            }
        }

        // Global max pooling
        for (int f = 0; f < F; f++) {
            float maxv = -1e30f;
            for (int p = 0; p < out_len; p++) {
                float v = conv_out[(long)p * F + f];
                if (v > maxv) maxv = v;
            }
            pooled[f] = maxv;
        }

        // FC layer
        for (int c = 0; c < C; c++) {
            float sum = fc_b[c];
            for (int f = 0; f < F; f++) sum += pooled[f] * fc_w[(long)f * C + c];
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(emb_seq); free(conv_out); free(pooled); free(logits);
    if (x) free(x);
    return 0;
}
```