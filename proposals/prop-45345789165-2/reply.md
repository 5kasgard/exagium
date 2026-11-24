```hypothesis
Family: label-embmlp. A shared byte embedding (size E) summed over the 16 context positions, followed by one hidden layer (size H, ReLU) and a linear output layer. This captures non-linear interactions between context characters while keeping parameters low via the embedding bottleneck. It should beat label-linear (0.3375) because next-character prediction benefits from non-linear context mixing. If E or H are too small, it may underfit and not surpass the linear baseline.
```
```python
"""label-embmlp: shared embedding + one hidden ReLU layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
E = 64          # embedding dimension
H = 128         # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 128
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
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    emb = (rng.normal(0, 1/math.sqrt(E), size=(256, E))).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(2.0 / E), size=(H, E))).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), size=(C, H))).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state: m, v for each param
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]          # (b, 16)
            yb = y[idx]          # (b,)
            bsize = xb.shape[0]

            # Forward
            emb_sum = np.zeros((bsize, E), dtype=np.float32)
            for j in range(d):
                emb_sum += emb[xb[:, j]]        # (bsize, E)
            hidden_pre = emb_sum @ W1.T + b1    # (bsize, H)
            hidden = np.maximum(hidden_pre, 0)
            logits = hidden @ W2.T + b2         # (bsize, C)

            # Softmax + cross-entropy gradient
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(bsize), yb] -= 1.0
            grad_logits /= bsize

            # Backward
            grad_W2 = grad_logits.T @ hidden                    # (C, H)
            grad_b2 = grad_logits.sum(axis=0)                   # (C,)

            grad_hidden = grad_logits @ W2                      # (bsize, H)
            grad_hidden_pre = grad_hidden * (hidden_pre > 0)    # (bsize, H)

            grad_W1 = grad_hidden_pre.T @ emb_sum               # (H, E)
            grad_b1 = grad_hidden_pre.sum(axis=0)               # (H,)

            grad_emb_sum = grad_hidden_pre @ W1                 # (bsize, E)
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb, xb[:, j], grad_emb_sum)

            # L2 regularization
            grad_W2 += L2 * W2
            grad_W1 += L2 * W1
            grad_emb  += L2 * emb

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

            # Adam update
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * g
                v *= 0.999; v += 0.001 * g * g
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    # Header: magic, d, E, H, C
    header = struct.pack("<5i", 0x454D424D, d, E, H, C)
    # Data order: emb (256*E), W1 (H*E), b1 (H), W2 (C*H), b2 (C)
    data = (emb.astype("<f4").tobytes() +
            W1.astype("<f4").tobytes() +
            b1.astype("<f4").tobytes() +
            W2.astype("<f4").tobytes() +
            b2.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + data)

if __name__ == "__main__":
    main()
```
```c
/* label-embmlp inference: ./infer weights.bin in.rec N
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k>>8 & 255, k>>16 & 255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 20L + 4L * (256L*E + (long)H*E + H + (long)C*H + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + 256L * E;
    const float *b1 = W1 + (long)H * E;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)C * H;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // embedding sum
        for (int e = 0; e < E; e++) emb_sum[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *row = emb + (long)xi[j] * E;
            for (int e = 0; e < E; e++) emb_sum[e] += row[e];
        }

        // hidden layer: ReLU(emb_sum * W1^T + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            const float *w = W1 + (long)h * E;
            for (int e = 0; e < E; e++) sum += emb_sum[e] * w[e];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            const float *w = W2 + (long)c * H;
            for (int h = 0; h < H; h++) sum += hidden[h] * w[h];
            logits[c] = sum;
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++)
            if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```