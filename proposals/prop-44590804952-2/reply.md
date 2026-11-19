```hypothesis
Family: label-embedmlp (embedding + one hidden ReLU layer). The 16 input bytes are categorical; per-position embeddings capture character similarities (e.g., letters near each other) and the hidden layer models pairwise interactions. This should beat label-linear (0.3375) by sharing statistical strength across the 256 tokens. If dev accuracy ≤ 0.3375, the extra parameters are not justified.
```
```python
"""Embedding + one hidden ReLU layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x454D4244, d, D, H, C; float32 embeddings[d][256][D],
then if H>0: W1[D][H], b1[H], W2[H][C], b2[C] else W2[D][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split conceptually, then fixed)
D = 128          # embedding dimension
H = 64           # hidden size (0 for linear on embeddings)
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

MAGIC = 0x454D4244  # "EMBD"

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
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # Embeddings: [d, 256, D]
    E = rng.normal(0, 1/math.sqrt(D), size=(d, 256, D)).astype(np.float32)
    if H:
        W1 = rng.normal(0, math.sqrt(2.0/D), size=(D, H)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = rng.normal(0, math.sqrt(2.0/H), size=(H, C)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [E, W1, b1, W2, b2]
    else:
        W2 = rng.normal(0, math.sqrt(2.0/D), size=(D, C)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [E, W2, b2]

    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(X_batch):
        # X_batch: (b, d) uint8
        b = X_batch.shape[0]
        # Sum embeddings
        emb_sum = np.zeros((b, D), np.float32)
        for j in range(d):
            emb_sum += E[j, X_batch[:, j]]
        if H:
            hidden = np.maximum(emb_sum @ W1 + b1, 0, dtype=np.float32)
            logits = hidden @ W2 + b2
            return emb_sum, hidden, logits
        else:
            logits = emb_sum @ W2 + b2
            return emb_sum, None, logits

    def loss_grad(logits, y_batch):
        b = logits.shape[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits, dtype=np.float32)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(b), y_batch] -= 1.0
        return probs / b  # dL/dlogits

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            emb_sum, hidden, logits = forward(Xb)
            dlogits = loss_grad(logits, yb)

            # Gradients
            grads = [np.zeros_like(p) for p in params]
            if H:
                # W2, b2
                grads[3] = hidden.T @ dlogits + L2 * W2
                grads[4] = dlogits.sum(axis=0)
                # hidden grad
                dhidden = dlogits @ W2.T
                dhidden *= (hidden > 0).astype(np.float32)
                # W1, b1
                grads[1] = emb_sum.T @ dhidden + L2 * W1
                grads[2] = dhidden.sum(axis=0)
                # emb_sum grad
                demb_sum = dhidden @ W1.T
            else:
                # W2, b2
                grads[1] = emb_sum.T @ dlogits + L2 * W2
                grads[2] = dlogits.sum(axis=0)
                demb_sum = dlogits @ W2.T

            # Embedding gradients (scatter add)
            grad_E = np.zeros_like(E)
            for j in range(d):
                np.add.at(grad_E[j], Xb[:, j], demb_sum)
            grads[0] = grad_E + L2 * E

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9**t)
                v_hat = v / (1 - 0.999**t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    out = bytearray()
    out += struct.pack("<5i", MAGIC, d, D, H, C)
    out += E.astype("<f4").tobytes()
    if H:
        out += params[1].astype("<f4").tobytes()  # W1
        out += params[2].astype("<f4").tobytes()  # b1
        out += params[3].astype("<f4").tobytes()  # W2
        out += params[4].astype("<f4").tobytes()  # b2
    else:
        out += params[1].astype("<f4").tobytes()  # W2
        out += params[2].astype("<f4").tobytes()  # b2
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-embedmlp inference: ./infer weights.bin in.rec N
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || D <= 0 || H < 0 || C <= 0) { fprintf(stderr, "bad header\n"); return 3; }

    size_t emb_sz = (size_t)d * 256 * D;
    size_t w1_sz = H ? (size_t)D * H : 0;
    size_t b1_sz = H ? H : 0;
    size_t w2_sz = H ? (size_t)H * C : (size_t)D * C;
    size_t b2_sz = C;
    size_t total_floats = emb_sz + w1_sz + b1_sz + w2_sz + b2_sz;
    if (nw != 20 + (long)total_floats * 4) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + emb_sz;
    const float *b1 = W1 + w1_sz;
    const float *W2 = b1 + b1_sz;
    const float *b2 = W2 + w2_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = (float *)malloc(sizeof(float) * D);
    float *hidden = H ? (float *)malloc(sizeof(float) * H) : NULL;
    float *scores = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memset(acc, 0, D * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((size_t)j * 256 + xi[j]) * D;
            for (int kk = 0; kk < D; kk++) acc[kk] += e[kk];
        }

        if (H) {
            for (int u = 0; u < H; u++) {
                float sum = b1[u];
                for (int kk = 0; kk < D; kk++) sum += acc[kk] * W1[(size_t)kk * H + u];
                hidden[u] = sum > 0.0f ? sum : 0.0f;
            }
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int u = 0; u < H; u++) sum += hidden[u] * W2[(size_t)u * C + c];
                scores[c] = sum;
            }
        } else {
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int kk = 0; kk < D; kk++) sum += acc[kk] * W2[(size_t)kk * C + c];
                scores[c] = sum;
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```