```hypothesis
Family: label-emb-mlp. Uses per-position byte embeddings (16×256×E) followed by one ReLU hidden layer (H) and a linear output layer (65 classes). Non-linear interactions between context positions should improve over linear models on this character-level language task. If dev accuracy does not exceed 0.3375 (label-linear) the family is not worthwhile.
```

```python
"""label-emb-mlp: per-position embeddings + one ReLU hidden layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format (little-endian):
  magic 0x454D424C ('LMBE'), d, E, H, C (int32 each)
  embeddings[d][256][E] (float32)
  W1[d*E][H] (float32), b1[H] (float32)
  W2[H][C] (float32), b2[C] (float32)"""
import json, math, struct, sys

import numpy as np

# Hyperparameters chosen on tuning split (last 20% of train.in)
E = 16          # embedding dimension per position
H = 64          # hidden layer size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 20
SAMPLES_PER_EPOCH = 600000  # not used, but kept for compatibility

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X_bytes.shape[0]

    rng = np.random.default_rng(SEED)

    # Embeddings: [d, 256, E]
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    # Hidden layer: [d*E, H]
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Output layer: [H, C]
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    def forward(X_idx):
        # X_idx: [batch, d] uint8
        # embed: [batch, d, E]
        e = emb[X_idx, np.arange(d)]  # advanced indexing: emb[X_idx, pos] -> [batch, d, E]
        flat = e.reshape(X_idx.shape[0], d * E)
        h = flat @ W1 + b1
        h = np.maximum(h, 0)
        logits = h @ W2 + b2
        return logits, flat, h

    def backward(X_idx, flat, h, grad_logits):
        # grad_logits: [batch, C]
        grad_W2 = h.T @ grad_logits
        grad_b2 = grad_logits.sum(0)
        grad_h = grad_logits @ W2.T
        grad_h[h <= 0] = 0
        grad_W1 = flat.T @ grad_h
        grad_b1 = grad_h.sum(0)
        grad_flat = grad_h @ W1.T
        grad_emb = grad_flat.reshape(X_idx.shape[0], d, E)
        # scatter-add to emb gradients
        grad_emb_full = np.zeros_like(emb)
        np.add.at(grad_emb_full, (np.arange(d), X_idx.T), grad_emb.transpose(1, 0, 2))
        return [grad_emb_full, grad_W1, grad_b1, grad_W2, grad_b2]

    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            X_batch = X_bytes[idx]
            y_batch = y[idx]

            logits, flat, h = forward(X_batch)
            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(y_batch)), y_batch] -= 1.0
            grad_logits = (probs / len(y_batch)).astype(np.float32)

            grads = backward(X_batch, flat, h, grad_logits)

            # L2 regularization on weights (not biases/embeddings)
            grads[1] += L2 * W1
            grads[3] += L2 * W2

            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x454D424C  # 'LMBE'
    out = struct.pack("<5i", magic, d, E, H, C)
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output record: 1 byte (class index). */
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights config\n"); return 3; }

    size_t emb_sz = (size_t)d * 256 * E;
    size_t W1_sz = (size_t)d * E * H;
    size_t W2_sz = (size_t)H * C;
    long expected = 20 + 4L * (emb_sz + W1_sz + H + W2_sz + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + emb_sz;
    const float *b1 = W1 + W1_sz;
    const float *W2 = b1 + H;
    const float *b2 = W2 + W2_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* hidden = ReLU(emb_lookup(xi) * W1 + b1) */
        for (int h = 0; h < H; h++) hidden[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((size_t)j * 256 + xi[j]) * E;
            for (int h = 0; h < H; h++) {
                float sum = 0.0f;
                for (int e_idx = 0; e_idx < E; e_idx++)
                    sum += e[e_idx] * W1[((size_t)j * E + e_idx) * H + h];
                hidden[h] += sum;
            }
        }
        for (int h = 0; h < H; h++)
            if (hidden[h] < 0.0f) hidden[h] = 0.0f;

        /* logits = hidden * W2 + b2 */
        memcpy(logits, b2, C * sizeof(float));
        for (int h = 0; h < H; h++) {
            float hv = hidden[h];
            if (hv == 0.0f) continue;
            for (int c = 0; c < C; c++)
                logits[c] += hv * W2[(size_t)h * C + c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++)
            if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```