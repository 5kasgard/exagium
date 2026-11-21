```hypothesis
An embedding-MLP (label-emb-mlp) learns a dense 8-d vector per (position, byte) pair, then a 64-unit ReLU hidden layer before the 65-class output. This captures positional character semantics and cross-position interactions that the linear model misses, while staying far smaller (≈45k params vs 266k) and faster at inference than label-gbdt. If dev accuracy ≤ 0.3375 (label-linear) or inference cost exceeds label-linear by >2×, the approach fails.
```

```python
"""Embedding-MLP family for fixed-size byte items: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy. Learns per-position byte embeddings (E=8), one ReLU layer (H=64),
then linear to C classes. Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)). Exports flat binary:
magic 0x454D424D, int32 d,E,H,C; float32 emb[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

E, H = 8, 64
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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    rng = np.random.default_rng(SEED)

    # Embeddings: (d, 256, E)
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    # Hidden layer
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Output layer
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    def forward(X_batch):
        # X_batch: (B, d) uint8
        B = X_batch.shape[0]
        # Gather embeddings: (B, d, E)
        e = emb[X_batch, np.arange(256)]  # advanced indexing: (B, d) -> (B, d, E) not directly; need proper indexing
        # Instead, loop over positions for clarity and memory efficiency
        e = np.empty((B, d, E), np.float32)
        for j in range(d):
            e[:, j] = emb[j, X_batch[:, j]]
        x = e.reshape(B, d * E)
        h = x @ W1 + b1
        np.maximum(h, 0, out=h)
        logits = h @ W2 + b2
        return logits, h, x

    def backward(grad_logits, h, x, B):
        # grad_logits: (B, C)
        grad_W2 = h.T @ grad_logits
        grad_b2 = grad_logits.sum(0)
        grad_h = grad_logits @ W2.T
        grad_h[h == 0] = 0.0  # ReLU derivative
        grad_W1 = x.T @ grad_h
        grad_b1 = grad_h.sum(0)
        grad_x = grad_h @ W1.T  # (B, d*E)
        # Scatter back to embeddings
        grad_emb = np.zeros_like(emb)
        grad_x_rs = grad_x.reshape(B, d, E)
        for j in range(d):
            idx = X_batch[:, j]
            np.add.at(grad_emb[j], idx, grad_x_rs[:, j])
        return [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

    t = 0
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            X_batch, y_batch = X[idx], y[idx]
            B = len(X_batch)

            logits, h, x = forward(X_batch)
            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), y_batch] -= 1.0
            grad_logits = (probs / B).astype(np.float32)

            grads = backward(grad_logits, h, x, B)
            # Add L2 regularization gradients
            grads[1] += L2 * W1
            grads[3] += L2 * W2

            t += 1
            beta1, beta2 = 0.9, 0.999
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= beta1; m += (1 - beta1) * gr
                v *= beta2; v += (1 - beta2) * gr * gr
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Flatten embeddings to (d, 256, E) -> (d*256*E)
    emb_flat = emb.reshape(-1)
    out = struct.pack("<5i", 0x454D424D, d, E, H, C)
    out += emb_flat.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Embedding-MLP inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index (argmax). */
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x454D424D) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C < 1 || C > 256) { fprintf(stderr, "bad header dims\n"); return 3; }

    size_t emb_sz = (size_t)d * 256 * E;
    size_t W1_sz = (size_t)d * E * H;
    size_t b1_sz = H;
    size_t W2_sz = (size_t)H * C;
    size_t b2_sz = C;
    long expected = 20 + 4L * (emb_sz + W1_sz + b1_sz + W2_sz + b2_sz);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(wb + 20);
    const float *W1 = emb + emb_sz;
    const float *b1 = W1 + W1_sz;
    const float *W2 = b1 + b1_sz;
    const float *b2 = W2 + W2_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);
    float *concat = malloc(sizeof(float) * d * E);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Gather embeddings into concat
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((size_t)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, src, E * sizeof(float));
        }

        // Hidden layer: ReLU(concat * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int f = 0; f < d * E; f++) sum += concat[f] * W1[(size_t)f * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(size_t)h * C + c];
            logits[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(hidden); free(logits); free(concat);
    free(wb); if (x) free(x);
    return 0;
}
```