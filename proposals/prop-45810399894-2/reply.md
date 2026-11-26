```hypothesis
Family: label-emb-mlp (embedding MLP with factorized first layer).
Why: The 16-byte context has only 65 active byte values; a full one-hot first layer (label-linear with H>0) uses 16*256*H parameters. Factorizing into 16*256*D embeddings + (16*D)*H with D<<H reduces parameters, acts as regularization, and should generalize better on 24k train items. A hidden ReLU layer adds non-linearity missing from label-linear (H=0). This should beat label-linear's 0.3375 dev accuracy while keeping inference cost moderate (D=16, H=64 ~20k FLOPs/item).
Failure: Dev accuracy <= 0.3375 or instructions/item > 2x label-linear.
```

```python
"""label-emb-mlp: two-layer MLP with per-position byte embeddings (factorized first layer).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x454D424C, d, D, H, C; float32 emb[d][256][D], W1[d*D][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
D = 16          # embedding dimension
H = 64          # hidden layer size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 128
SAMPLES = 600000
MAX_EPOCHS = 50

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
    # Load data
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]

    # Tuning split: last 20% in file order
    split = int(0.8 * n)
    idx_train = np.arange(split)
    idx_tune = np.arange(split, n)

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # embeddings: (d, 256, D)
    emb = (rng.uniform(-1, 1, (d, 256, D)) * math.sqrt(6.0 / 256)).astype(np.float32)
    # W1: (d*D, H)
    W1 = (rng.uniform(-1, 1, (d * D, H)) * math.sqrt(6.0 / (d * D))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # W2: (H, C)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return (probs / len(yy)).astype(np.float32)

    def forward(emb, W1, b1, W2, b2, Xb):
        # Xb: (batch, d) uint8
        batch = Xb.shape[0]
        # Embedding lookup: (batch, d, D)
        e = emb[:, Xb, :]  # numpy advanced indexing: (d, batch, D) -> transpose
        e = np.transpose(e, (1, 0, 2))  # (batch, d, D)
        # Flatten to (batch, d*D)
        e_flat = e.reshape(batch, d * D)
        # Hidden layer
        h = e_flat @ W1 + b1
        h = np.maximum(h, 0)  # ReLU
        # Output
        logits = h @ W2 + b2
        return logits, h, e_flat

    def backward(grad_logits, h, e_flat, Xb, W2):
        # grad_logits: (batch, C)
        # Gradient wrt W2, b2
        gW2 = h.T @ grad_logits + L2 * W2
        gb2 = grad_logits.sum(axis=0)
        # Gradient wrt h
        gh = grad_logits @ W2.T
        gh[h <= 0] = 0  # ReLU derivative
        # Gradient wrt e_flat
        ge_flat = gh @ W1.T
        # Gradient wrt W1, b1
        gW1 = e_flat.T @ gh + L2 * W1
        gb1 = gh.sum(axis=0)
        # Gradient wrt embeddings
        ge = ge_flat.reshape(gh.shape[0], d, D)  # (batch, d, D)
        # Accumulate per position and byte value
        gemb = np.zeros_like(emb)
        for j in range(d):
            bytes_j = Xb[:, j]
            np.add.at(gemb[j], bytes_j, ge[:, j, :])
        gemb += L2 * emb
        return [gemb, gW1, gb1, gW2, gb2]

    def evaluate(Xb, yy):
        logits, _, _ = forward(emb, W1, b1, W2, b2, Xb)
        pred = logits.argmax(axis=1)
        return (pred == yy).mean()

    # Adam optimizer
    t = 0
    best_tune_acc = -1.0
    best_params = None

    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(split)
        for s in range(0, split, BATCH):
            idx = order[s:s + BATCH]
            Xb = X_bytes[idx]
            yy = y[idx]
            logits, h, e_flat = forward(emb, W1, b1, W2, b2, Xb)
            grad_logits = softmax_grad(logits, yy)
            grads = backward(grad_logits, h, e_flat, Xb, W2)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
        # Evaluate on tuning split
        if len(idx_tune) > 0:
            tune_acc = evaluate(X_bytes[idx_tune], y[idx_tune])
            if tune_acc > best_tune_acc:
                best_tune_acc = tune_acc
                best_params = [p.copy() for p in params]
        # Early stop if we've seen enough samples
        if (epoch + 1) * split >= SAMPLES:
            break

    # Use best params on tuning split (or last if no tuning split)
    if best_params is not None:
        emb, W1, b1, W2, b2 = best_params

    # Final training on full data (optional, but we already trained on 80%; we can do a few more epochs on full)
    # For simplicity, we just use the best model from above. The rule says "choose settings on tuning split, then train on all of train".
    # We'll do a few more epochs on full data with the same hyperparameters.
    for epoch in range(5):  # few epochs on full data
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb = X_bytes[idx]
            yy = y[idx]
            logits, h, e_flat = forward(emb, W1, b1, W2, b2, Xb)
            grad_logits = softmax_grad(logits, yy)
            grads = backward(grad_logits, h, e_flat, Xb, W2)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    # magic, d, D, H, C
    header = struct.pack("<5i", 0x454D424C, d, D, H, C)
    # embeddings: (d, 256, D) -> flatten in order d, 256, D
    emb_bytes = emb.astype("<f4").tobytes()
    W1_bytes = W1.astype("<f4").tobytes()
    b1_bytes = b1.astype("<f4").tobytes()
    W2_bytes = W2.astype("<f4").tobytes()
    b2_bytes = b2.astype("<f4").tobytes()
    out = header + emb_bytes + W1_bytes + b1_bytes + W2_bytes + b2_bytes
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte class index (0..64).
 * Weights format: magic 0x454D424C, int32 d,D,H,C; float32 emb[d][256][D], W1[d*D][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || D <= 0 || H <= 0 || C <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * D + (long)d * D * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * D;
    const float *b1 = W1 + (long)d * D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * H);
    float *score = malloc(sizeof(float) * C);
    float *emb_row = malloc(sizeof(float) * D);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // acc = b1
        memcpy(acc, b1, H * sizeof(float));
        // For each position j
        for (int j = 0; j < d; j++) {
            unsigned char byte = xi[j];
            const float *e = emb + ((long)j * 256 + byte) * D;
            // acc += e * W1[j*D : (j+1)*D, :]
            const float *W1_j = W1 + (long)j * D * H;
            for (int k = 0; k < D; k++) {
                float ek = e[k];
                if (ek == 0.0f) continue;
                const float *w = W1_j + (long)k * H;
                for (int h = 0; h < H; h++) acc[h] += ek * w[h];
            }
        }
        // ReLU
        for (int h = 0; h < H; h++) if (acc[h] < 0.0f) acc[h] = 0.0f;
        // Output layer
        memcpy(score, b2, C * sizeof(float));
        for (int h = 0; h < H; h++) {
            float ah = acc[h];
            if (ah == 0.0f) continue;
            const float *w = W2 + (long)h * C;
            for (int c = 0; c < C; c++) score[c] += ah * w[c];
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```