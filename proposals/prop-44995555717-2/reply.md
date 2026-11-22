```hypothesis
We propose "label-embed-mlp": per-position byte embeddings (size 64) summed, then a ReLU hidden layer (size 128), then a linear output to 65 classes. This captures pairwise interactions via the hidden layer while keeping inference fast (≈16*64 + 64*128 + 128*65 multiply-adds). If dev accuracy ≤ 0.35, the extra non-linearity does not help enough.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters chosen on tuning split (last 20% of train.in)
E = 64          # embedding size per position
H = 128         # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-5
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
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Embedding matrix: (d, 256, E) -> flatten to (d*256, E)
    emb = rng.normal(0, 1/math.sqrt(E), size=(d*256, E)).astype(np.float32)
    W1 = rng.normal(0, math.sqrt(2.0/E), size=(E, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = rng.normal(0, math.sqrt(2.0/H), size=(H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # Precompute flat indices for each position: pos*256 + byte_value
    pos_offset = np.arange(d) * 256

    def forward(emb_flat, W1, b1, W2, b2, X_batch):
        # X_batch: (batch, d) uint8
        idx = X_batch + pos_offset  # (batch, d)
        idx_flat = idx.ravel()
        # Sum embeddings
        summed = emb_flat[idx_flat].reshape(X_batch.shape[0], d, E).sum(axis=1)  # (batch, E)
        hidden = summed @ W1 + b1
        hidden = np.maximum(hidden, 0)
        logits = hidden @ W2 + b2
        return logits, hidden, summed

    def backward(logits, y_batch, hidden, summed, W1, W2):
        batch = logits.shape[0]
        # Softmax cross-entropy gradient
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(batch), y_batch] -= 1.0
        probs /= batch  # (batch, C)

        # Output layer gradients
        grad_W2 = hidden.T @ probs + L2 * W2
        grad_b2 = probs.sum(axis=0)

        # Hidden layer gradients
        grad_hidden = probs @ W2.T
        grad_hidden *= (hidden > 0)

        # W1, b1 gradients
        grad_W1 = summed.T @ grad_hidden + L2 * W1
        grad_b1 = grad_hidden.sum(axis=0)

        # Embedding gradients
        grad_summed = grad_hidden @ W1.T  # (batch, E)
        # scatter-add to embeddings
        grad_emb = np.zeros_like(emb_flat)
        # For each position, we add grad_summed to the corresponding embedding rows
        # We can use np.add.at for each position
        for j in range(d):
            idx_j = X_batch[:, j] + j*256
            np.add.at(grad_emb, idx_j, grad_summed)
        grad_emb += L2 * emb_flat
        return grad_emb, grad_W1, grad_b1, grad_W2, grad_b2

    emb_flat = emb  # alias
    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            X_batch = X[idx]
            y_batch = y[idx]

            logits, hidden, summed = forward(emb_flat, W1, b1, W2, b2, X_batch)
            grads = backward(logits, y_batch, hidden, summed, W1, W2)

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9**t)
                v_hat = v / (1 - 0.999**t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights: magic, d, E, H, C
    magic = 0x4D4C5031  # "MLP1"
    header = struct.pack("<5i", magic, d, E, H, C)
    # Order: emb_flat (d*256*E), W1 (E*H), b1 (H), W2 (H*C), b2 (C)
    body = (emb_flat.astype("<f4").tobytes() +
            W1.astype("<f4").tobytes() +
            b1.astype("<f4").tobytes() +
            W2.astype("<f4").tobytes() +
            b2.astype("<f4").tobytes())
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x4D4C5031) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected_floats = (long)d * 256 * E + (long)E * H + H + (long)H * C + C;
    if (nw != 20 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *summed = (float *)malloc(sizeof(float) * E);
    float *hidden = (float *)malloc(sizeof(float) * H);
    float *logits = (float *)malloc(sizeof(float) * C);
    if (!summed || !hidden || !logits) { fprintf(stderr, "alloc failed\n"); return 3; }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // zero summed
        for (int e = 0; e < E; e++) summed[e] = 0.0f;

        // sum embeddings
        for (int j = 0; j < d; j++) {
            const float *emb_row = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) summed[e] += emb_row[e];
        }

        // hidden = ReLU(summed @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float acc = b1[h];
            for (int e = 0; e < E; e++) acc += summed[e] * W1[(long)e * H + h];
            hidden[h] = acc > 0.0f ? acc : 0.0f;
        }

        // logits = hidden @ W2 + b2
        for (int c = 0; c < C; c++) {
            float acc = b2[c];
            for (int h = 0; h < H; h++) acc += hidden[h] * W2[(long)h * C + c];
            logits[c] = acc;
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(summed); free(hidden); free(logits);
    if (x) free(x);
    return 0;
}
```