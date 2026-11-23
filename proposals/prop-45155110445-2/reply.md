```hypothesis
Family: label-embmlp (embedding MLP). Replaces one-hot encoding with learned per-position byte embeddings (16×256×E), followed by a ReLU hidden layer (H) and a linear output. Embeddings capture similarity between characters and reduce parameter count vs. one-hot, allowing a wider hidden layer for the same budget. Should beat label-linear on dev accuracy because the task is character-level language modeling where embeddings are known to help. If dev accuracy does not exceed label-linear by at least 0.01, the family does not justify its extra complexity.
```

```python
"""Embedding MLP for next-character prediction: train.py
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
E = 16          # embedding dimension per position
H = 64          # hidden layer size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 20

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

    # Embeddings: list of d arrays, each (256, E)
    emb = [rng.normal(0, 0.01, (256, E)).astype(np.float32) for _ in range(d)]
    # Hidden layer: (d*E, H)
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Output layer: (H, C)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state: for each param array, m and v
    state = []
    for p in params:
        if isinstance(p, list):
            state.append([(np.zeros_like(arr), np.zeros_like(arr)) for arr in p])
        else:
            state.append((np.zeros_like(p), np.zeros_like(p)))

    t = 0
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = xb.shape[0]

            # Forward
            # Embedding lookup: concatenate embeddings for each position
            emb_concat = np.empty((bs, d * E), np.float32)
            for j in range(d):
                emb_concat[:, j*E:(j+1)*E] = emb[j][xb[:, j]]
            h = emb_concat @ W1 + b1
            h_relu = np.maximum(h, 0)
            logits = h_relu @ W2 + b2

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grad_logits = probs / bs

            # Backprop
            grad_W2 = h_relu.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_h_relu = grad_logits @ W2.T
            grad_h = grad_h_relu * (h > 0)
            grad_W1 = emb_concat.T @ grad_h + L2 * W1
            grad_b1 = grad_h.sum(axis=0)
            # Embedding gradients
            grad_emb = [np.zeros_like(e) for e in emb]
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_h[:, j*E:(j+1)*E])
            for j in range(d):
                grad_emb[j] += L2 * emb[j]

            # Adam update
            t += 1
            lr_t = LR * math.sqrt(1 - 0.999**t) / (1 - 0.9**t)
            # Helper to update a param array with its grad and state
            def adam_update(param, grad, m, v):
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                param -= lr_t * m / (np.sqrt(v) + 1e-8)

            for j in range(d):
                m, v = state[0][j]
                adam_update(emb[j], grad_emb[j], m, v)
            m, v = state[1]; adam_update(W1, grad_W1, m, v)
            m, v = state[2]; adam_update(b1, grad_b1, m, v)
            m, v = state[3]; adam_update(W2, grad_W2, m, v)
            m, v = state[4]; adam_update(b2, grad_b2, m, v)

    # Write weights.bin
    # Magic: 0x454D424D ('EMBM'), then d, E, H, C as int32
    magic = 0x454D424D
    header = struct.pack("<5i", magic, d, E, H, C)
    out = bytearray(header)
    # Embeddings: d * 256 * E floats
    for j in range(d):
        out += emb[j].astype("<f4").tobytes()
    # W1, b1, W2, b2
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Embedding MLP inference: ./infer weights.bin in.rec N
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
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E <= 0 || H <= 0 || C != 65) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *emb = p; p += (long)d * 256 * E;
    const float *W1 = p; p += (long)d * E * H;
    const float *b1 = p; p += H;
    const float *W2 = p; p += (long)H * C;
    const float *b2 = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_concat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(emb_concat + (long)j * E, e, E * sizeof(float));
        }

        // Hidden layer: ReLU(emb_concat @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k = 0; k < d * E; k++) sum += emb_concat[k] * W1[(long)k * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(emb_concat); free(hidden); free(logits);
    return 0;
}
```