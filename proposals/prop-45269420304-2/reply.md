```hypothesis
Family: label-conv1d. A 1D convolutional network over position-specific byte embeddings (E=8) with 32 filters of kernel size 3, global max pooling, and a linear output layer. This captures local n-gram patterns in the 16-byte context, which are crucial for character-level language modeling. If dev accuracy does not exceed label-linear (0.3375) or inference cost exceeds 2x label-linear, the family is not worthwhile.
```

```python
"""label-conv1d: 1D CNN over per-position byte embeddings for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x434F4E31, d, E, F, K, C; float32 embeddings[d][256][E],
conv_w[K][E][F], conv_b[F], lin_w[F][C], lin_b[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
E = 8        # embedding dimension per position
F = 32       # number of conv filters
K = 3        # kernel size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 20
SAMPLES = 600000

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
    exp = np.exp(logits, dtype=np.float32)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(B), y] -= 1.0
    return probs / B

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    assert d == 16 and C == 65

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # embeddings: (d, 256, E)
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / (256 + E))).astype(np.float32)
    # conv_w: (K, E, F)
    conv_w = (rng.uniform(-1, 1, (K, E, F)) * math.sqrt(6.0 / (K * E))).astype(np.float32)
    conv_b = np.zeros(F, np.float32)
    # lin_w: (F, C)
    lin_w = (rng.uniform(-1, 1, (F, C)) * math.sqrt(6.0 / F)).astype(np.float32)
    lin_b = np.zeros(C, np.float32)

    params = [emb, conv_w, conv_b, lin_w, lin_b]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            B = xb.shape[0]

            # Forward
            # Embedding lookup: (B, d, E)
            emb_out = np.empty((B, d, E), np.float32)
            for j in range(d):
                emb_out[:, j, :] = emb[j, xb[:, j], :]

            # Conv1d + ReLU: (B, d-K+1, F)
            conv_out = np.zeros((B, d - K + 1, F), np.float32)
            for f in range(F):
                for p in range(d - K + 1):
                    # (B, K, E) * (K, E) -> sum over K,E
                    conv_out[:, p, f] = np.sum(emb_out[:, p:p+K, :] * conv_w[:, :, f], axis=(1,2)) + conv_b[f]
            np.maximum(conv_out, 0, out=conv_out)  # ReLU inplace

            # Max pooling over positions: (B, F)
            pool_out = np.max(conv_out, axis=1)
            argmax = np.argmax(conv_out, axis=1)  # (B, F)

            # Linear: (B, C)
            logits = pool_out @ lin_w + lin_b

            # Loss gradient
            dlogits = softmax_cross_entropy_grad(logits, yb)  # (B, C)

            # Backward
            # Linear grads
            dlin_w = pool_out.T @ dlogits
            dlin_b = dlogits.sum(axis=0)

            # Pooling grad
            dpool = dlogits @ lin_w.T  # (B, F)

            # Max pool unpooling
            dconv_out = np.zeros_like(conv_out)
            for b in range(B):
                for f in range(F):
                    dconv_out[b, argmax[b, f], f] = dpool[b, f]

            # ReLU grad
            dconv_out *= (conv_out > 0)

            # Conv grads
            dconv_w = np.zeros_like(conv_w)
            dconv_b = np.zeros_like(conv_b)
            demb_out = np.zeros_like(emb_out)

            for f in range(F):
                for p in range(d - K + 1):
                    # dconv_w[:, :, f] += sum_b (emb_out[b, p:p+K, :] * dconv_out[b, p, f])
                    # dconv_b[f] += sum_b dconv_out[b, p, f]
                    # demb_out[b, p:p+K, :] += dconv_out[b, p, f] * conv_w[:, :, f]
                    dconv_b[f] += dconv_out[:, p, f].sum()
                    # vectorized over batch
                    dconv_w[:, :, f] += np.sum(emb_out[:, p:p+K, :] * dconv_out[:, p, f, None, None], axis=0)
                    demb_out[:, p:p+K, :] += dconv_out[:, p, f, None, None] * conv_w[:, :, f]

            # Embedding grads
            demb = np.zeros_like(emb)
            for j in range(d):
                # demb[j, byte, :] += sum_{b where xb[b,j]==byte} demb_out[b, j, :]
                for b in range(B):
                    byte = xb[b, j]
                    demb[j, byte, :] += demb_out[b, j, :]

            # L2 regularization on weights (not biases)
            dconv_w += L2 * conv_w
            dlin_w += L2 * lin_w
            # Embeddings also get L2?
            demb += L2 * emb

            grads = [demb, dconv_w, dconv_b, dlin_w, dlin_b]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export weights
    # magic, d, E, F, K, C
    header = struct.pack("<6i", 0x434F4E31, d, E, F, K, C)
    # embeddings: d * 256 * E floats
    emb_bytes = emb.astype("<f4").tobytes()
    # conv_w: K * E * F
    conv_w_bytes = conv_w.astype("<f4").tobytes()
    # conv_b: F
    conv_b_bytes = conv_b.astype("<f4").tobytes()
    # lin_w: F * C
    lin_w_bytes = lin_w.astype("<f4").tobytes()
    # lin_b: C
    lin_b_bytes = lin_b.astype("<f4").tobytes()

    out = header + emb_bytes + conv_w_bytes + conv_b_bytes + lin_w_bytes + lin_b_bytes
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-conv1d inference: ./infer weights.bin in.rec N
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 24 || hd[0] != 0x434F4E31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], F = hd[3], K = hd[4], C = hd[5];
    if (d != 16 || E != 8 || F != 32 || K != 3 || C != 65) {
        fprintf(stderr, "unexpected hyperparameters: d=%d E=%d F=%d K=%d C=%d\n", d, E, F, K, C);
        return 3;
    }
    long expected = 24 + 4L * ((long)d * 256 * E + (long)K * E * F + F + (long)F * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 6);
    const float *conv_w = emb + (long)d * 256 * E;
    const float *conv_b = conv_w + (long)K * E * F;
    const float *lin_w = conv_b + F;
    const float *lin_b = lin_w + (long)F * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    // Temporary buffers
    float *emb_out = malloc(sizeof(float) * d * E);
    float *conv_out = malloc(sizeof(float) * (d - K + 1) * F);
    float *pool_out = malloc(sizeof(float) * F);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * E;
            float *dst = emb_out + j * E;
            memcpy(dst, src, E * sizeof(float));
        }

        // Conv1d + ReLU
        int conv_len = d - K + 1;
        for (int f = 0; f < F; f++) {
            const float *w = conv_w + f * K * E;
            float b = conv_b[f];
            for (int p = 0; p < conv_len; p++) {
                float sum = b;
                for (int kk = 0; kk < K; kk++) {
                    const float *e = emb_out + (p + kk) * E;
                    const float *wk = w + kk * E;
                    for (int e_idx = 0; e_idx < E; e_idx++) {
                        sum += e[e_idx] * wk[e_idx];
                    }
                }
                conv_out[p * F + f] = sum > 0 ? sum : 0;
            }
        }

        // Max pooling over positions
        for (int f = 0; f < F; f++) {
            float maxv = -INFINITY;
            for (int p = 0; p < conv_len; p++) {
                float v = conv_out[p * F + f];
                if (v > maxv) maxv = v;
            }
            pool_out[f] = maxv;
        }

        // Linear layer
        for (int c = 0; c < C; c++) {
            float sum = lin_b[c];
            for (int f = 0; f < F; f++) {
                sum += pool_out[f] * lin_w[f * C + c];
            }
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(emb_out); free(conv_out); free(pool_out); free(logits);
    return 0;
}
```