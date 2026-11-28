```hypothesis
Family: label-cnn1d. A 1D convolutional network over learned byte embeddings captures local n-gram statistics crucial for character prediction. With kernel size 3, 16 filters, embedding dim 8, and global max pooling, it adds minimal parameters (~14 KiB) and compute over label-linear. If it fails to beat label-linear (0.3375), it suggests local convolutions without recurrence are insufficient for 16-byte context.
```
```python
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
E = 8          # embedding dimension
K = 3          # kernel size
F = 16         # number of filters
C = 65         # classes (from card)
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
    # Load data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    embed = (rng.normal(0, 0.01, (256, E))).astype(np.float32)
    # He init for conv (fan_in = K*E)
    conv_w = (rng.normal(0, math.sqrt(2.0 / (K * E)), (F, K, E))).astype(np.float32)
    conv_b = np.zeros(F, np.float32)
    # Xavier init for dense (fan_in = F)
    dense_w = (rng.normal(0, math.sqrt(2.0 / (F + C)), (F, C))).astype(np.float32)
    dense_b = np.zeros(C, np.float32)

    params = [embed, conv_w, conv_b, dense_w, dense_b]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    # Precompute number of epochs
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    pool_len = d - K + 1  # 14

    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward
            # Embedding lookup: (B, d, E)
            emb = embed[xb]      # (B, d, E)
            # Convolution: valid, stride 1
            # We'll compute for each filter and position
            conv_out = np.zeros((B, F, pool_len), np.float32)
            for f in range(F):
                for p in range(pool_len):
                    # (B, K, E) * (K, E) -> (B,)
                    window = emb[:, p:p+K, :]  # (B, K, E)
                    conv_out[:, f, p] = np.sum(window * conv_w[f], axis=(1,2))
            conv_out += conv_b.reshape(1, F, 1)
            # ReLU
            relu_out = np.maximum(conv_out, 0)
            # Max pooling over positions
            pooled = np.max(relu_out, axis=2)  # (B, F)
            # Dense
            logits = pooled @ dense_w + dense_b  # (B, C)

            # Softmax cross-entropy gradient
            max_logits = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = probs / B  # (B, C)

            # Backward
            # Dense gradients
            grad_dense_w = pooled.T @ grad_logits + L2 * dense_w
            grad_dense_b = np.sum(grad_logits, axis=0)
            grad_pooled = grad_logits @ dense_w.T  # (B, F)

            # Max pooling backward: only max positions get gradient
            grad_relu = np.zeros_like(relu_out)
            max_pos = np.argmax(relu_out, axis=2)  # (B, F)
            for b in range(B):
                for f in range(F):
                    grad_relu[b, f, max_pos[b, f]] = grad_pooled[b, f]

            # ReLU backward
            grad_conv = grad_relu * (conv_out > 0)

            # Convolution backward
            grad_conv_w = np.zeros_like(conv_w)
            grad_conv_b = np.sum(grad_conv, axis=(0,2))  # (F,)
            grad_emb = np.zeros_like(emb)

            for f in range(F):
                for p in range(pool_len):
                    window = emb[:, p:p+K, :]  # (B, K, E)
                    # grad_conv[:, f, p] is (B,)
                    g = grad_conv[:, f, p].reshape(B, 1, 1)
                    grad_conv_w[f] += np.sum(g * window, axis=0)  # (K, E)
                    grad_emb[:, p:p+K, :] += g * conv_w[f]

            grad_conv_w += L2 * conv_w

            # Embedding backward: scatter add
            # We'll accumulate using np.add.at
            # grad_emb is (B, d, E), xb is (B, d) indices
            flat_idx = xb.reshape(-1)
            flat_grad = grad_emb.reshape(-1, E)
            np.add.at(embed, flat_idx, flat_grad)
            grad_embed = np.zeros_like(embed)  # placeholder, we already updated embed directly? Wait, we need to update via Adam, so we need gradients for each param.
            # Actually we updated embed in-place above? No, we used np.add.at on embed, but embed is a parameter. We should compute gradient for embed and then update via Adam.
            # Let's recompute: we need grad_embed as an array of same shape as embed.
            grad_embed = np.zeros_like(embed)
            np.add.at(grad_embed, flat_idx, flat_grad)
            grad_embed += L2 * embed

            # Collect gradients in param order
            grads = [grad_embed, grad_conv_w, grad_conv_b, grad_dense_w, grad_dense_b]

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
    # Magic: 0x434E4E31 ('CNN1')
    header = struct.pack("<6i", 0x434E4E31, d, E, K, F, C)
    body = (
        embed.astype("<f4").tobytes() +
        conv_w.astype("<f4").tobytes() +
        conv_b.astype("<f4").tobytes() +
        dense_w.astype("<f4").tobytes() +
        dense_b.astype("<f4").tobytes()
    )
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 24) { fprintf(stderr, "weights too small\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x434E4E31) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], E = hd[2], K = hd[3], F = hd[4], C = hd[5];
    if (d != 16 || E != 8 || K != 3 || F != 16 || C != 65) {
        fprintf(stderr, "unexpected dimensions\n"); return 3;
    }
    int pool_len = d - K + 1; // 14
    size_t expected = 24
        + 256 * E * 4
        + F * K * E * 4
        + F * 4
        + F * C * 4
        + C * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld vs %zu\n", nw, expected); return 3; }

    const float *embed = (const float *)(wb + 24);
    const float *conv_w = embed + 256 * E;
    const float *conv_b = conv_w + F * K * E;
    const float *dense_w = conv_b + F;
    const float *dense_b = dense_w + F * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *emb_buf = malloc(d * E * sizeof(float));
    float *conv_out = malloc(F * pool_len * sizeof(float));
    float *pooled = malloc(F * sizeof(float));
    float *logits = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup
        for (int p = 0; p < d; p++) {
            const float *e = embed + xi[p] * E;
            memcpy(emb_buf + p * E, e, E * sizeof(float));
        }

        // Convolution + ReLU
        for (int f = 0; f < F; f++) {
            const float *fw = conv_w + f * K * E;
            float b = conv_b[f];
            for (int p = 0; p < pool_len; p++) {
                float sum = b;
                for (int kk = 0; kk < K; kk++) {
                    const float *win = emb_buf + (p + kk) * E;
                    for (int e = 0; e < E; e++) sum += win[e] * fw[kk * E + e];
                }
                conv_out[f * pool_len + p] = sum > 0 ? sum : 0;
            }
        }

        // Max pooling
        for (int f = 0; f < F; f++) {
            float mx = conv_out[f * pool_len];
            for (int p = 1; p < pool_len; p++) {
                float v = conv_out[f * pool_len + p];
                if (v > mx) mx = v;
            }
            pooled[f] = mx;
        }

        // Dense
        for (int c = 0; c < C; c++) {
            float sum = dense_b[c];
            for (int f = 0; f < F; f++) sum += pooled[f] * dense_w[f * C + c];
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(emb_buf); free(conv_out); free(pooled); free(logits);
    free(wb); if (x) free(x);
    return 0;
}
```