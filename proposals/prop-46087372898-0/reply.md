```hypothesis
Factorization Machine (label-fm) with 4-dim embeddings per field-value per class. Captures pairwise feature interactions crucial for tabular data (e.g., education × occupation), while staying efficient via factorized parameterization. Expect >0.855 accuracy at ~1000 Ir/item. Fails if interactions don't help or overfit (dev < 0.852).
```
```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Each of the d input bytes (0-255) is a categorical field. Learns per-field per-value
linear weights and K-dim embeddings per class. Logits: bias + linear + pairwise FM interactions.
Adam lr 1e-2, L2 1e-4, minibatch 256, epochs min(50, ceil(600000/n)). Seed 1.
Export: int32 magic 0x464D4C31, d, K, C; float32 bias[C], linear[d][256][C], emb[d][256][K][C]."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 256, 600000, 50
MAGIC = 0x464D4C31  # 'FML1'

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
    assert d == 14 and C == 2
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Parameters: bias[C], linear[d,256,C], emb[d,256,K,C]
    bias = np.zeros(C, np.float32)
    linear = (rng.normal(0, 0.01, (d, 256, C)) * np.sqrt(2.0 / 256)).astype(np.float32)
    emb = (rng.normal(0, 0.01, (d, 256, K, C)) * np.sqrt(2.0 / 256)).astype(np.float32)
    params = [bias, linear, emb]
    param_shapes = [p.shape for p in params]

    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            m = len(yb)

            # Forward
            # bias term
            logits = np.tile(bias, (m, 1))
            # linear term
            for j in range(d):
                logits += linear[j, xb[:, j], :]
            # FM interaction term per class
            for c in range(C):
                s_emb = np.zeros((m, K), np.float32)
                sq_emb = np.zeros((m, K), np.float32)
                for j in range(d):
                    e = emb[j, xb[:, j], :, c]  # (m, K)
                    s_emb += e
                    sq_emb += e * e
                inter = 0.5 * np.sum(s_emb * s_emb - sq_emb, axis=1)
                logits[:, c] += inter

            # Softmax + loss gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(m), yb] -= 1.0
            grad_logits = probs / m  # (m, C)

            # Backward
            g_bias = grad_logits.sum(axis=0).astype(np.float32)
            g_linear = np.zeros_like(linear)
            g_emb = np.zeros_like(emb)

            for c in range(C):
                glc = grad_logits[:, c]  # (m,)
                # linear grad
                for j in range(d):
                    np.add.at(g_linear[j, :, c], xb[:, j], glc)
                # emb grad: d(inter)/de = s_emb - e
                s_emb = np.zeros((m, K), np.float32)
                for j in range(d):
                    s_emb += emb[j, xb[:, j], :, c]
                for j in range(d):
                    e = emb[j, xb[:, j], :, c]  # (m, K)
                    grad_e = glc[:, None] * (s_emb - e)  # (m, K)
                    np.add.at(g_emb[j, :, :, c], xb[:, j], grad_e)

            # L2 regularization
            g_bias += L2 * bias
            g_linear += L2 * linear
            g_emb += L2 * emb

            grads = [g_bias, g_linear, g_emb]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m_adam, v_adam = state[i]
                m_adam *= 0.9; m_adam += 0.1 * gr
                v_adam *= 0.999; v_adam += 0.001 * gr * gr
                m_hat = m_adam / (1 - 0.9 ** t)
                v_hat = v_adam / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    out = struct.pack("<4i", MAGIC, d, K, C)
    out += bias.astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine for label tasks: ./infer weights.bin in.rec N
 * Input: d bytes (0-255 each). Output: 1 byte class index (argmax).
 * Weights: magic 0x464D4C31, d, K, C; float32 bias[C], linear[d][256][C], emb[d][256][K][C]. */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1 || K < 1) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16L + 4L * (C + (long)d * 256 * C + (long)d * 256 * K * C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + C;
    const float *emb = linear + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(sizeof(float) * C);
    float *s_emb = malloc(sizeof(float) * C * K);
    float *sq_emb = malloc(sizeof(float) * C * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            logits[c] = bias[c];
            for (int kk = 0; kk < K; kk++) {
                s_emb[c * K + kk] = 0.0f;
                sq_emb[c * K + kk] = 0.0f;
            }
        }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lin_jv = linear + ((long)j * 256 + v) * C;
            const float *emb_jv = emb + ((long)j * 256 + v) * K * C;
            for (int c = 0; c < C; c++) {
                logits[c] += lin_jv[c];
                for (int kk = 0; kk < K; kk++) {
                    float e = emb_jv[kk * C + c];
                    s_emb[c * K + kk] += e;
                    sq_emb[c * K + kk] += e * e;
                }
            }
        }

        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int kk = 0; kk < K; kk++) {
                float s = s_emb[c * K + kk];
                inter += 0.5f * (s * s - sq_emb[c * K + kk]);
            }
            logits[c] += inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```