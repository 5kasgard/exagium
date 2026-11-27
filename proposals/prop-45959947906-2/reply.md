```hypothesis
A shared-embedding 1D CNN with global max-pooling captures local n-gram interactions across the 16-byte context while keeping parameters tiny (~5k). The shared 256×8 embedding reduces memory vs per-position one-hot (266k in label-linear). A single conv layer (kernel=3, 32 filters) + ReLU + max-pool + linear to 65 classes should exceed label-linear's 33.75% accuracy at lower inference cost (fewer FLOPs, better cache). Failure: dev accuracy ≤ 0.3375 or instructions/item ≥ label-linear's 7359.9.
```

```python
import json, math, struct, sys
import numpy as np

# Fixed hyper-parameters (tuned on last 20% of train.in, then trained on all)
E = 8          # embedding dimension
F = 32         # number of conv filters
K = 3          # kernel size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 50
MAGIC = 0x434E4E31  # "CNN1"

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
    C = card["classes"]
    n, d = X.shape
    assert d == 16

    rng = np.random.default_rng(SEED)

    # Parameters
    emb = (rng.normal(0, 0.01, (256, E))).astype(np.float32)
    W_conv = (rng.normal(0, math.sqrt(2.0 / (K * E)), (K, E, F))).astype(np.float32)
    b_conv = np.zeros(F, np.float32)
    W_out = (rng.normal(0, math.sqrt(2.0 / F), (F, C))).astype(np.float32)
    b_out = np.zeros(C, np.float32)

    params = [emb, W_conv, b_conv, W_out, b_out]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # One-hot for gradient wrt embeddings
    # We'll compute embedding gradients via scatter-add using indices
    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, 16)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward
            # Embedding lookup: (B, 16, E)
            x_emb = emb[xb]      # (B, 16, E)

            # Conv1d with kernel K, stride 1, no padding -> (B, 16-K+1, F)
            conv_out = np.zeros((B, 16 - K + 1, F), np.float32)
            for i in range(K):
                conv_out += x_emb[:, i:i + 16 - K + 1] @ W_conv[i]
            conv_out += b_conv
            h = np.maximum(conv_out, 0)  # ReLU

            # Global max pooling over time -> (B, F)
            pooled = h.max(axis=1)

            # Output layer
            logits = pooled @ W_out + b_out  # (B, C)

            # Softmax cross-entropy
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            # Gradient of loss wrt logits
            grad_logits = probs.copy()
            grad_logits[np.arange(B), yb] -= 1.0
            grad_logits /= B

            # Backward: output layer
            grad_W_out = pooled.T @ grad_logits + L2 * W_out
            grad_b_out = grad_logits.sum(axis=0)
            grad_pooled = grad_logits @ W_out.T  # (B, F)

            # Backward: max pooling
            # grad_h has same shape as h, zero except at max positions
            max_pos = h.argmax(axis=1)  # (B, F)
            grad_h = np.zeros_like(h)
            for b in range(B):
                for f in range(F):
                    grad_h[b, max_pos[b, f], f] = grad_pooled[b, f]

            # Backward: ReLU
            grad_h[h <= 0] = 0

            # Backward: conv
            grad_W_conv = np.zeros_like(W_conv)
            grad_b_conv = grad_h.sum(axis=(0, 1))
            grad_x_emb = np.zeros_like(x_emb)

            for i in range(K):
                # x_emb[:, i:i+16-K+1] shape (B, T, E)
                # grad_h shape (B, T, F)
                x_slice = x_emb[:, i:i + 16 - K + 1]
                grad_W_conv[i] = x_slice.transpose(0, 2, 1).reshape(-1, E).T @ grad_h.reshape(-1, F)
                grad_W_conv[i] += L2 * W_conv[i]
                # grad_x_emb for this slice
                grad_slice = grad_h @ W_conv[i].T  # (B, T, E)
                grad_x_emb[:, i:i + 16 - K + 1] += grad_slice

            # Backward: embedding lookup (scatter-add)
            grad_emb = np.zeros_like(emb)
            np.add.at(grad_emb, xb.ravel(), grad_x_emb.reshape(-1, E))
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W_conv, grad_b_conv, grad_W_out, grad_b_out]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Save weights
    # Order: emb (256*E), W_conv (K*E*F), b_conv (F), W_out (F*C), b_out (C)
    header = struct.pack("<5I", MAGIC, E, F, K, C)
    body = (
        emb.astype("<f4").tobytes() +
        W_conv.astype("<f4").tobytes() +
        b_conv.astype("<f4").tobytes() +
        W_out.astype("<f4").tobytes() +
        b_out.astype("<f4").tobytes()
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x434E4E31) { fprintf(stderr, "bad weights\n"); return 3; }
    int E = hd[1], F = hd[2], K = hd[3], C = hd[4];
    const int d = 16;
    const int T = d - K + 1;  // conv output length

    // Expected size
    long exp = 20L + 4L * (256L * E + K * E * F + F + F * C + C);
    if (nw != exp) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *emb = p; p += 256 * E;
    const float *W_conv = p; p += K * E * F;
    const float *b_conv = p; p += F;
    const float *W_out = p; p += F * C;
    const float *b_out = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_emb = malloc(16 * E * sizeof(float));
    float *h = malloc(T * F * sizeof(float));
    float *pooled = malloc(F * sizeof(float));
    float *logits = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup
        for (int j = 0; j < 16; j++) {
            const float *e = emb + xi[j] * E;
            memcpy(x_emb + j * E, e, E * sizeof(float));
        }

        // Conv1d + ReLU
        for (int t = 0; t < T; t++) {
            for (int f = 0; f < F; f++) {
                float sum = b_conv[f];
                for (int kk = 0; kk < K; kk++) {
                    const float *x_ptr = x_emb + (t + kk) * E;
                    const float *w_ptr = W_conv + (kk * E + 0) * F + f;  // W_conv[kk, :, f] stride F
                    for (int e = 0; e < E; e++) {
                        sum += x_ptr[e] * w_ptr[e * F];
                    }
                }
                h[t * F + f] = sum > 0.0f ? sum : 0.0f;
            }
        }

        // Global max pooling
        for (int f = 0; f < F; f++) {
            float mx = h[f];
            for (int t = 1; t < T; t++) {
                float v = h[t * F + f];
                if (v > mx) mx = v;
            }
            pooled[f] = mx;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b_out[c];
            for (int f = 0; f < F; f++) {
                sum += pooled[f] * W_out[f * C + c];
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
    return 0;
}
```