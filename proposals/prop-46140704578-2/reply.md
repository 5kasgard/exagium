```hypothesis
Family: label-embmlp (shared byte embeddings + positional embeddings + 1-hidden-layer MLP).
Why: Shared byte embeddings capture character similarities (e.g., vowels cluster) while positional embeddings retain order information. This reduces parameters vs one-hot linear (4096→256*D+16*D) and acts as a strong prior for language, improving generalization on small tinyshakespeare data.
Failure signal: If dev accuracy ≤ label-linear (33.75%), the task requires higher-order interactions not captured by this shallow architecture.
```
```python
"""label-embmlp: shared byte embeddings + positional embeddings + one ReLU layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x454D424D, int32 d,D,H,C; float32 byte_emb[256*D], pos_emb[d*D], W1[d*D*H], b1[H], W2[H*C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (fixed, no tuning split search)
D = 16          # embedding dimension
H = 32          # hidden layer size
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
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    byte_emb = (rng.normal(0, 0.01, (256, D))).astype(np.float32)
    pos_emb  = (rng.normal(0, 0.01, (d, D))).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * D)), (d * D, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [byte_emb, pos_emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    # Pre-allocate buffers for speed
    flat = np.empty((BATCH, d * D), np.float32)
    h = np.empty((BATCH, H), np.float32)
    logits = np.empty((BATCH, C), np.float32)
    probs = np.empty((BATCH, C), np.float32)
    dlogits = np.empty((BATCH, C), np.float32)
    dh = np.empty((BATCH, H), np.float32)
    dflat = np.empty((BATCH, d * D), np.float32)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            bsz = len(idx)
            xb = X[idx]
            yb = y[idx]

            # Forward
            # flat = concat(byte_emb[xb[:,j]] + pos_emb[j] for j in range(d))
            for j in range(d):
                np.add(byte_emb[xb[:, j]], pos_emb[j], out=flat[:, j*D:(j+1)*D])
            np.maximum(flat @ W1 + b1, 0, out=h)
            logits[:bsz] = h @ W2 + b2

            # Softmax + cross-entropy gradient
            max_log = logits[:bsz].max(axis=1, keepdims=True)
            np.exp(logits[:bsz] - max_log, out=probs[:bsz])
            probs[:bsz] /= probs[:bsz].sum(axis=1, keepdims=True)
            dlogits[:bsz] = probs[:bsz]
            dlogits[np.arange(bsz), yb] -= 1.0
            dlogits[:bsz] /= bsz

            # Backprop
            dW2 = h[:bsz].T @ dlogits[:bsz]
            db2 = dlogits[:bsz].sum(axis=0)
            dh[:bsz] = dlogits[:bsz] @ W2.T
            dh[:bsz] *= (h[:bsz] > 0)
            dW1 = flat[:bsz].T @ dh[:bsz]
            db1 = dh[:bsz].sum(axis=0)
            dflat[:bsz] = dh[:bsz] @ W1.T

            # Embedding gradients
            d_byte_emb = np.zeros_like(byte_emb)
            d_pos_emb = np.zeros_like(pos_emb)
            for j in range(d):
                slice_j = dflat[:bsz, j*D:(j+1)*D]
                d_pos_emb[j] = slice_j.sum(axis=0)
                # accumulate per byte value
                for i in range(bsz):
                    d_byte_emb[xb[i, j]] += slice_j[i]

            grads = [d_byte_emb, d_pos_emb, dW1, db1, dW2, db2]

            # Adam update with L2
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                gr += L2 * p
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    magic = 0x454D424D  # 'EMBM'
    header = struct.pack("<5i", magic, d, D, H, C)
    out = header
    for arr in params:
        out += arr.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

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
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C < 1 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20 + 4L * (256L * D + d * D + d * D * H + H + H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *byte_emb = (const float *)(hd + 5);
    const float *pos_emb  = byte_emb + 256 * D;
    const float *W1       = pos_emb + d * D;
    const float *b1       = W1 + d * D * H;
    const float *W2       = b1 + H;
    const float *b2       = W2 + H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * D);
    float *h_arr = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // flat = concat(byte_emb[xi[j]] + pos_emb[j])
        for (int j = 0; j < d; j++) {
            const float *be = byte_emb + xi[j] * D;
            const float *pe = pos_emb + j * D;
            float *dst = flat + j * D;
            for (int v = 0; v < D; v++) dst[v] = be[v] + pe[v];
        }

        // h = relu(flat @ W1 + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < d * D; v++) sum += flat[v] * W1[v * H + u];
            h_arr[u] = sum > 0.0f ? sum : 0.0f;
        }

        // logits = h @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h_arr[u] * W2[u * C + c];
            logits[c] = sum;
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```