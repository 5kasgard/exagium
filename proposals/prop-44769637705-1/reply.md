```hypothesis
Family: label-linear-onehot (linear model on one-hot encoded pixel values).
Why: Optdigits pixels are ordinal 0..16 but the relationship with digit class may be non-monotonic per pixel. One-hot encoding captures arbitrary per-pixel value effects while remaining a linear model, so inference is fast (64*10 adds per item). It has 64*17*10=10880 weights, still small.
Failure: If dev accuracy <= 0.9613 (label-linear) or inference cost exceeds label-linear significantly.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameters
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 100
SAMPLES = 600000  # used to cap epochs

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
    d = card["input"]["bytes"]
    # For optdigits, pixels are 0..16. We'll use 17 values.
    num_values = 17
    # Clamp values to 0..16
    X = np.clip(X, 0, num_values - 1)

    rng = np.random.default_rng(SEED)
    # Initialize weights: (d, num_values, C)
    W = rng.normal(0, 0.01, size=(d, num_values, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    # Adam state
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]  # (batch, d)
            yb = y[idx]  # (batch,)
            batch_size = len(yb)

            # Forward: scores = b + sum_j W[j, xb[:,j], :]
            scores = np.tile(b, (batch_size, 1))  # (batch, C)
            for j in range(d):
                scores += W[j, xb[:, j], :]

            # Softmax cross-entropy gradient
            max_scores = np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(scores - max_scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            probs[np.arange(batch_size), yb] -= 1.0
            grad_scores = probs / batch_size  # (batch, C)

            # Gradients for W and b
            grad_b = np.sum(grad_scores, axis=0)  # (C,)
            grad_W = np.zeros_like(W)
            for j in range(d):
                np.add.at(grad_W[j], xb[:, j], grad_scores)

            # L2 regularization
            grad_W += L2 * W
            grad_b += L2 * b

            # Adam update
            t += 1
            # W
            mW = 0.9 * mW + 0.1 * grad_W
            vW = 0.999 * vW + 0.001 * grad_W * grad_W
            mW_hat = mW / (1 - 0.9 ** t)
            vW_hat = vW / (1 - 0.999 ** t)
            W -= LR * mW_hat / (np.sqrt(vW_hat) + 1e-8)
            # b
            mb = 0.9 * mb + 0.1 * grad_b
            vb = 0.999 * vb + 0.001 * grad_b * grad_b
            mb_hat = mb / (1 - 0.9 ** t)
            vb_hat = vb / (1 - 0.999 ** t)
            b -= LR * mb_hat / (np.sqrt(vb_hat) + 1e-8)

    # Write weights.bin
    magic = 0x4F4E4548  # "ONEH"
    out = struct.pack("<4i", magic, d, num_values, C)
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Linear one-hot model for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes (pixels 0..16). Output is one byte class.
 * Weights format: magic (0x4F4E4548), d, num_values, C; then float32 W[d][num_values][C], then float32 b[C]. */
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
    if (nw < 16 || hd[0] != 0x4F4E4548) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], num_values = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }
    long expected = 16 + 4L * ((long)d * num_values * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *W = (const float *)(hd + 4);
    const float *b = W + (long)d * num_values * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Initialize scores with bias
        for (int c = 0; c < C; c++) scores[c] = b[c];
        // Add contributions from each pixel
        for (int j = 0; j < d; j++) {
            int val = xi[j];
            if (val >= num_values) val = num_values - 1; // clamp
            const float *w = W + ((long)j * num_values + val) * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```