```hypothesis
Family: label-cnn (1D convolutional net with embeddings). Tiny Shakespeare is a character-level sequence task where local n-gram patterns (e.g., "ing", "the") are highly predictive. A small CNN with shared embeddings, positional encodings, and a kernel size of 3 captures these patterns with far fewer parameters than a full one-hot linear model, while being much faster than GBRT. If dev accuracy does not exceed 0.34 (label-linear) at lower Ir/item, the architecture is too weak for this context length.
```

```python
"""label-cnn: 1D CNN with embeddings for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Architecture: 65-class char embedding (E=32) + 16 positional embeddings -> 1D conv (kernel=3, C1=64, ReLU) -> global avg pool -> linear (65).
Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Weights: magic 0x434E4E31, d=16, E, C1, K=3, C=65; then float32 arrays in order:
  emb[65][E], pos_emb[16][E], conv_w[C1][E][3], conv_b[C1], fc_w[65][C1], fc_b[65].
"""
import json, math, struct, sys

import numpy as np

E = 32
C1 = 64
K = 3
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    class_chars = card["class_chars"]
    # map byte value -> class index
    byte_to_idx = {ord(ch): i for i, ch in enumerate(class_chars)}
    # load data
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, 16)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    # convert X to class indices
    X = np.vectorize(lambda b: byte_to_idx.get(b, 0))(X_bytes)  # default 0 (newline) if unseen
    n = X.shape[0]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # init parameters
    emb = (rng.normal(0, 0.01, (65, E))).astype(np.float32)
    pos_emb = (rng.normal(0, 0.01, (16, E))).astype(np.float32)
    conv_w = (rng.normal(0, math.sqrt(2.0 / (E * K)), (C1, E, K))).astype(np.float32)
    conv_b = np.zeros(C1, np.float32)
    fc_w = (rng.normal(0, math.sqrt(2.0 / C1), (C1, C))).astype(np.float32)
    fc_b = np.zeros(C, np.float32)

    params = [emb, pos_emb, conv_w, conv_b, fc_w, fc_b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def forward(X_batch):
        # X_batch: (B, 16) int indices
        B = X_batch.shape[0]
        # embedding + positional
        x = emb[X_batch] + pos_emb[None, :, :]  # (B, 16, E)
        # conv1d with padding=1 (valid on padded)
        # pad left and right with zeros
        x_pad = np.pad(x, ((0,0), (1,1), (0,0)), mode='constant')  # (B, 18, E)
        out = np.zeros((B, 16, C1), np.float32)
        for c in range(C1):
            for k in range(K):
                out[:, :, c] += x_pad[:, k:k+16, :] @ conv_w[c, :, k].T
        out += conv_b
        out = np.maximum(out, 0)  # ReLU
        # global average pool
        pooled = out.mean(axis=1)  # (B, C1)
        # linear
        logits = pooled @ fc_w + fc_b  # (B, C)
        return logits, out, pooled, x

    def softmax_grad(logits, y_idx):
        # logits (B, C), y_idx (B,)
        max_logits = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(y_idx)), y_idx] -= 1.0
        return probs / len(y_idx)  # (B, C)

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            logits, conv_out, pooled, x_emb = forward(xb)
            g = softmax_grad(logits, yb)  # (B, C)
            # backprop
            # fc
            grad_fc_w = pooled.T @ g + L2 * fc_w
            grad_fc_b = g.sum(axis=0)
            # pooled
            g_pooled = g @ fc_w.T  # (B, C1)
            # global avg pool: divide by 16
            g_conv = g_pooled[:, None, :] / 16.0  # (B, 1, C1) -> broadcast to (B, 16, C1)
            # ReLU
            g_conv[conv_out <= 0] = 0
            # conv
            grad_conv_w = np.zeros_like(conv_w)
            grad_conv_b = g_conv.sum(axis=(0,1))
            x_pad = np.pad(x_emb, ((0,0), (1,1), (0,0)), mode='constant')
            for c in range(C1):
                for k in range(K):
                    grad_conv_w[c, :, k] = x_pad[:, k:k+16, :].transpose(0,2,1) @ g_conv[:, :, c]
            grad_conv_w += L2 * conv_w
            # x_emb (emb + pos_emb)
            g_x_emb = np.zeros_like(x_emb)
            for c in range(C1):
                for k in range(K):
                    g_x_emb += g_conv[:, :, c][:, None, :] * conv_w[c, :, k][None, :, None]
            # split to emb and pos_emb
            grad_emb = np.zeros_like(emb)
            grad_pos_emb = np.zeros_like(pos_emb)
            # accumulate gradients for emb
            for b in range(xb.shape[0]):
                for p in range(16):
                    idx_emb = xb[b, p]
                    grad_emb[idx_emb] += g_x_emb[b, p]
            grad_pos_emb = g_x_emb.sum(axis=0)
            grad_emb += L2 * emb
            grad_pos_emb += L2 * pos_emb

            grads = [grad_emb, grad_pos_emb, grad_conv_w, grad_conv_b, grad_fc_w, grad_fc_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # pack weights
    magic = 0x434E4E31  # 'CNN1'
    header = struct.pack("<6i", magic, 16, E, C1, K, C)
    body = (emb.astype("<f4").tobytes() +
            pos_emb.astype("<f4").tobytes() +
            conv_w.astype("<f4").tobytes() +
            conv_b.astype("<f4").tobytes() +
            fc_w.astype("<f4").tobytes() +
            fc_b.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* label-cnn inference: ./infer weights.bin in.rec N
 * Input record: 16 raw bytes. Output record: 1 byte (class index 0..64).
 * Weights format: magic 0x434E4E31, d=16, E, C1, K=3, C=65 (all int32).
 * Then float32 arrays: emb[65][E], pos_emb[16][E], conv_w[C1][E][3], conv_b[C1], fc_w[65][C1], fc_b[65].
 * C11, libc+libm only, single-threaded. */
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
    if (nw < 24 || hd[0] != 0x434E4E31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], C1 = hd[3], K = hd[4], C = hd[5];
    if (d != 16 || K != 3 || C != 65 || C1 <= 0 || E <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 24L + 4L * (65L * E + 16L * E + (long)C1 * E * 3 + C1 + 65L * C1 + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 6);
    const float *emb = p; p += 65 * E;
    const float *pos_emb = p; p += 16 * E;
    const float *conv_w = p; p += (long)C1 * E * 3;
    const float *conv_b = p; p += C1;
    const float *fc_w = p; p += 65 * C1;
    const float *fc_b = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_emb = malloc(16 * E * sizeof(float));
    float *conv_out = malloc(16 * C1 * sizeof(float));
    float *pooled = malloc(C1 * sizeof(float));
    float *logits = malloc(65 * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }

        // embedding + positional
        for (int t = 0; t < 16; t++) {
            int idx = xi[t];
            const float *e = emb + idx * E;
            const float *pe = pos_emb + t * E;
            float *dst = x_emb + t * E;
            for (int j = 0; j < E; j++) dst[j] = e[j] + pe[j];
        }

        // conv1d with padding=1 (zero pad)
        for (int c = 0; c < C1; c++) {
            for (int t = 0; t < 16; t++) {
                float sum = conv_b[c];
                for (int kk = 0; kk < 3; kk++) {
                    int src_t = t + kk - 1;
                    if (src_t >= 0 && src_t < 16) {
                        const float *w = conv_w + ((long)c * E * 3 + kk * E);
                        const float *src = x_emb + src_t * E;
                        for (int j = 0; j < E; j++) sum += src[j] * w[j];
                    }
                }
                conv_out[t * C1 + c] = sum > 0 ? sum : 0; // ReLU
            }
        }

        // global average pool
        for (int c = 0; c < C1; c++) {
            float sum = 0;
            for (int t = 0; t < 16; t++) sum += conv_out[t * C1 + c];
            pooled[c] = sum / 16.0f;
        }

        // linear
        for (int c = 0; c < 65; c++) {
            float sum = fc_b[c];
            const float *w = fc_w + (long)c * C1;
            for (int j = 0; j < C1; j++) sum += pooled[j] * w[j];
            logits[c] = sum;
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < 65; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```