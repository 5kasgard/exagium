```hypothesis
Family: label-emb (embedding + 1-hidden-layer MLP). Each of the 16 context positions gets a 256×E learned embedding; embeddings are summed, passed through a ReLU layer (size H), then a linear output to 65 classes. This captures byte similarities (e.g., vowels cluster) that one-hot linear cannot, while staying extremely fast: inference is 16×E adds + two small dense matvecs (E×H + H×65). With E=16, H=64 it has ~71k parameters and should exceed label-linear's 0.3375 accuracy at comparable or lower CPU cost. Failure to beat label-linear would indicate that linear one-hot is already optimal for this context length or that the factorized embedding bottleneck loses too much positional information.
```
```python
"""label-emb: embedding + 1-hidden-layer MLP for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

E = 16          # embedding dimension
H = 64          # hidden layer size
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
    """Return dlogits for softmax cross-entropy, shape (B, C)."""
    B = logits.shape[0]
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(B), y] -= 1.0
    return probs / B


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]          # 16
    C = card["classes"]     # 65
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters (float32)
    emb = (rng.normal(0, 1.0 / math.sqrt(E), size=(d, 256, E))).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(2.0 / E), size=(E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), size=(H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward
            emb_sum = np.zeros((B, E), np.float32)
            for j in range(d):
                emb_sum += emb[j, xb[:, j]]      # (B, E)
            hidden = emb_sum @ W1 + b1            # (B, H)
            np.maximum(hidden, 0, out=hidden)     # ReLU in-place
            logits = hidden @ W2 + b2             # (B, C)

            # Loss gradient
            dlogits = softmax_cross_entropy_grad(logits, yb)   # (B, C)

            # Backward
            dW2 = hidden.T @ dlogits + L2 * W2
            db2 = dlogits.sum(axis=0)
            dhidden = dlogits @ W2.T                 # (B, H)
            dhidden[hidden == 0] = 0.0               # ReLU grad
            dW1 = emb_sum.T @ dhidden + L2 * W1
            db1 = dhidden.sum(axis=0)
            demb_sum = dhidden @ W1.T                # (B, E)

            # Gradient for embeddings: accumulate per (position, byte)
            demb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(demb[j], xb[:, j], demb_sum)

            grads = [demb, dW1, db1, dW2, db2]

            # Adam update
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] *= 0.9; m[i] += 0.1 * g
                v[i] *= 0.999; v[i] += 0.001 * g * g
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights: magic, d, E, H, C; then emb, W1, b1, W2, b2 (all float32 LE)
    magic = 0x4C454D42  # "BLEM" little-endian
    header = struct.pack("<5i", magic, d, E, H, C)
    body = (
        emb.astype("<f4").tobytes() +
        W1.astype("<f4").tobytes() +
        b1.astype("<f4").tobytes() +
        W2.astype("<f4").tobytes() +
        b2.astype("<f4").tobytes()
    )
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```
```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index 0..64).
 * Weights format: int32 magic(0x4C454D42), d, E, H, C; float32 emb[d][256][E], W1[E][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x4C454D42) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || E > 256 || H <= 0 || H > 1024) {
        fprintf(stderr, "bad weights dims\n"); return 3;
    }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float emb_sum[256];   // max E=256
    float hidden[1024];   // max H=1024
    float logits[65];     // C=65

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // emb_sum = 0
        for (int e = 0; e < E; e++) emb_sum[e] = 0.0f;
        // accumulate embeddings
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) emb_sum[e] += eptr[e];
        }
        // hidden = ReLU(emb_sum @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int e = 0; e < E; e++) sum += emb_sum[e] * W1[(long)e * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }
        // logits = hidden @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
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