```hypothesis
Family: label-minirnn. A tiny recurrent network (32 tanh units) shares parameters across the 16-byte context, capturing sequential dependencies that position-wise linear models miss. With ~12k parameters it stays far smaller than label-linear (266k) and faster than label-gbdt. Expected dev accuracy >0.34; if it falls ≤0.3375 the sequential bias doesn't help this task.
```
```python
"""label-minirnn: tiny RNN for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
H = 32            # hidden size
SEED = 1
LR = 5e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 30

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(logits, targets):
    # logits: (B, C), targets: (B,) int
    B = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits, dtype=np.float32)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(B), targets] -= 1.0
    return probs / B  # (B, C)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]  # 16
    N = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Parameters: emb(256,H), W_hh(H,H), b_h(H), W_hy(H,C), b_y(C)
    emb = (rng.normal(0, 0.01, (256, H))).astype(np.float32)
    W_hh = (rng.normal(0, 0.01, (H, H)) * math.sqrt(1.0 / H)).astype(np.float32)
    b_h = np.zeros(H, np.float32)
    W_hy = (rng.normal(0, 0.01, (H, C)) * math.sqrt(1.0 / H)).astype(np.float32)
    b_y = np.zeros(C, np.float32)

    params = [emb, W_hh, b_h, W_hy, b_y]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for epoch in range(EPOCHS):
        order = rng.permutation(N)
        for s in range(0, N, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward pass: store hidden states for BPTT
            h = np.zeros((B, H), np.float32)
            h_seq = np.zeros((d, B, H), np.float32)
            for step in range(d):
                x_emb = emb[xb[:, step]]          # (B, H)
                h = np.tanh(x_emb + h @ W_hh.T + b_h)
                h_seq[step] = h
            logits = h @ W_hy.T + b_y             # (B, C)

            # Loss gradient
            g_logits = softmax_cross_entropy_grad(logits, yb)  # (B, C)

            # Backward pass
            g_W_hy = h.T @ g_logits                          # (H, C)
            g_b_y = g_logits.sum(axis=0)                     # (C,)
            g_h = g_logits @ W_hy                            # (B, H)

            g_emb = np.zeros_like(emb)
            g_W_hh = np.zeros_like(W_hh)
            g_b_h = np.zeros_like(b_h)

            for step in reversed(range(d)):
                h_t = h_seq[step]
                g_h_raw = g_h * (1.0 - h_t * h_t)            # tanh derivative
                g_b_h += g_h_raw.sum(axis=0)
                if step > 0:
                    g_W_hh += h_seq[step-1].T @ g_h_raw
                else:
                    # no previous hidden state for step 0
                    pass
                # embedding gradient
                np.add.at(g_emb, xb[:, step], g_h_raw)
                # propagate to previous step
                g_h = g_h_raw @ W_hh

            grads = [g_emb, g_W_hh, g_b_h, g_W_hy, g_b_y]

            # L2 regularization
            for i, p in enumerate(params):
                grads[i] += L2 * p

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Pack weights: magic, d, H, C, then arrays in order
    magic = 0x4D4E5231  # 'MNR1'
    header = struct.pack("<4i", magic, d, H, C)
    blob = header
    for arr in params:
        blob += arr.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(blob)

if __name__ == "__main__":
    main()
```
```c
/* label-minirnn inference: ./infer weights.bin in.rec N
 * Input record: 16 raw bytes. Output record: 1 byte class index. */
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
    if (nw < 16 || hd[0] != 0x4D4E5231) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (d != 16 || H <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }

    size_t expected = 16
        + (size_t)256 * H * sizeof(float)
        + (size_t)H * H * sizeof(float)
        + (size_t)H * sizeof(float)
        + (size_t)H * C * sizeof(float)
        + (size_t)C * sizeof(float);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size: got %ld, expect %zu\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *emb = p; p += 256 * H;
    const float *W_hh = p; p += H * H;
    const float *b_h = p; p += H;
    const float *W_hy = p; p += H * C;
    const float *b_y = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = (float *)malloc(H * sizeof(float));
    float *scores = (float *)malloc(C * sizeof(float));
    if (!h || !scores) { fprintf(stderr, "oom\n"); return 3; }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memset(h, 0, H * sizeof(float));
        for (int step = 0; step < d; step++) {
            const float *e = emb + xi[step] * H;
            // h = tanh(e + h @ W_hh^T + b_h)
            for (int j = 0; j < H; j++) {
                float sum = b_h[j] + e[j];
                const float *w = W_hh + j * H;
                for (int k2 = 0; k2 < H; k2++) sum += h[k2] * w[k2];
                h[j] = tanhf(sum);
            }
        }
        // scores = h @ W_hy^T + b_y
        memcpy(scores, b_y, C * sizeof(float));
        for (int j = 0; j < H; j++) {
            const float *w = W_hy + j * C;
            float hj = h[j];
            for (int c = 0; c < C; c++) scores[c] += hj * w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```