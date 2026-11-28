```hypothesis
We propose "label-emb-mlp": position-specific embeddings (dimension D=32) for each of the 16 context bytes, concatenated and fed through a single ReLU hidden layer (H=128) to 65 output classes. This captures character similarities and cross-position interactions that a linear one-hot model cannot, while keeping inference fast via precomputed embedding tables. Trained with Adam (lr=1e-3, L2=1e-4) for 25 epochs on 24k samples. Expected to exceed label-linear's 0.3375 dev accuracy. Failure would indicate embeddings don't help or the hidden layer is still too small for this task.
```

```python
"""label-emb-mlp: position-specific embeddings + one ReLU layer for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Weights format: magic 0x454D4245, d, D, H, C; float32 embeddings[d][256][D], W1[16*D][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hardcoded hyperparameters (chosen via tuning split offline)
D = 32          # embedding dimension per position
H = 128         # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 30

MAGIC = 0x454D4245  # 'EMBE' little-endian


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

    # Initialize embeddings: list of 16 matrices (256, D)
    scale_emb = math.sqrt(6.0 / (256 + D))
    embeddings = [rng.uniform(-scale_emb, scale_emb, (256, D)).astype(np.float32) for _ in range(d)]

    # Initialize W1 (16*D, H), b1 (H)
    in_dim = d * D
    scale1 = math.sqrt(6.0 / (in_dim + H))
    W1 = rng.uniform(-scale1, scale1, (in_dim, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)

    # Initialize W2 (H, C), b2 (C)
    scale2 = math.sqrt(6.0 / (H + C))
    W2 = rng.uniform(-scale2, scale2, (H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [embeddings, W1, b1, W2, b2]
    # Adam state: for each param array, keep m and v
    state = []
    for p in params:
        if isinstance(p, list):
            state.append([(np.zeros_like(arr), np.zeros_like(arr)) for arr in p])
        else:
            state.append((np.zeros_like(p), np.zeros_like(p)))

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (batch, 16)
            yb = y[idx]          # (batch,)
            batch_size = len(yb)

            # Forward: embed each position and concatenate
            # emb_concat shape (batch, 16*D)
            emb_concat = np.empty((batch_size, in_dim), np.float32)
            for j in range(d):
                emb_concat[:, j*D:(j+1)*D] = embeddings[j][xb[:, j]]

            # Hidden layer
            hidden_pre = emb_concat @ W1 + b1          # (batch, H)
            hidden = np.maximum(hidden_pre, 0)         # ReLU
            logits = hidden @ W2 + b2                  # (batch, C)

            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(batch_size), yb] -= 1.0
            grad_logits = probs / batch_size           # (batch, C)

            # Backprop
            # grad_W2 = hidden.T @ grad_logits + L2 * W2
            grad_W2 = hidden.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)

            grad_hidden = grad_logits @ W2.T           # (batch, H)
            grad_hidden[hidden_pre <= 0] = 0.0         # ReLU derivative

            # grad_W1 = emb_concat.T @ grad_hidden + L2 * W1
            grad_W1 = emb_concat.T @ grad_hidden + L2 * W1
            grad_b1 = grad_hidden.sum(axis=0)

            # grad_embeddings per position
            grad_emb = [np.zeros((256, D), np.float32) for _ in range(d)]
            for j in range(d):
                # grad_hidden slice for this position's embeddings
                grad_emb_j = grad_hidden @ W1[j*D:(j+1)*D].T   # (batch, D)
                # scatter-add into grad_emb[j] using xb[:, j] as indices
                np.add.at(grad_emb[j], xb[:, j], grad_emb_j)

            # Adam updates
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            lr_t = LR * math.sqrt(1 - beta2**t) / (1 - beta1**t)

            # embeddings
            for j in range(d):
                m, v = state[0][j]
                g = grad_emb[j]
                m[:] = beta1 * m + (1 - beta1) * g
                v[:] = beta2 * v + (1 - beta2) * g * g
                embeddings[j] -= lr_t * m / (np.sqrt(v) + eps)

            # W1, b1
            m, v = state[1]
            m[:] = beta1 * m + (1 - beta1) * grad_W1
            v[:] = beta2 * v + (1 - beta2) * grad_W1 * grad_W1
            W1 -= lr_t * m / (np.sqrt(v) + eps)

            m, v = state[2]
            m[:] = beta1 * m + (1 - beta1) * grad_b1
            v[:] = beta2 * v + (1 - beta2) * grad_b1 * grad_b1
            b1 -= lr_t * m / (np.sqrt(v) + eps)

            # W2, b2
            m, v = state[3]
            m[:] = beta1 * m + (1 - beta1) * grad_W2
            v[:] = beta2 * v + (1 - beta2) * grad_W2 * grad_W2
            W2 -= lr_t * m / (np.sqrt(v) + eps)

            m, v = state[4]
            m[:] = beta1 * m + (1 - beta1) * grad_b2
            v[:] = beta2 * v + (1 - beta2) * grad_b2 * grad_b2
            b2 -= lr_t * m / (np.sqrt(v) + eps)

    # Write weights.bin
    # Header: magic, d, D, H, C (5 int32)
    header = struct.pack("<5i", MAGIC, d, D, H, C)
    # Embeddings: for each position, 256 vectors of D (row-major)
    emb_bytes = b"".join(arr.astype("<f4").tobytes() for arr in embeddings)
    # W1 (in_dim, H) row-major, b1 (H), W2 (H, C) row-major, b2 (C)
    w1_bytes = W1.astype("<f4").tobytes()
    b1_bytes = b1.astype("<f4").tobytes()
    w2_bytes = W2.astype("<f4").tobytes()
    b2_bytes = b2.astype("<f4").tobytes()

    out = header + emb_bytes + w1_bytes + b1_bytes + w2_bytes + b2_bytes
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index 0..64).
 * Weights format: magic 0x454D4245, d, D, H, C; float32 embeddings[d][256][D],
 *   W1[16*D][H], b1[H], W2[H][C], b2[C] (all row-major). */
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
    if (nw < 20 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || D <= 0 || H <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }

    // Expected size: header(20) + embeddings(d*256*D) + W1(d*D*H) + b1(H) + W2(H*C) + b2(C) all float32 (4 bytes)
    long expected = 20 + 4L * ( (long)d * 256 * D + (long)d * D * H + H + (long)H * C + C );
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * D;
    const float *b1 = W1 + (long)d * D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = (float *)malloc(sizeof(float) * H);
    float *scores = (float *)malloc(sizeof(float) * C);
    if (!hidden || !scores) { fprintf(stderr, "malloc failed\n"); exit(3); }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Initialize hidden with bias
        memcpy(hidden, b1, H * sizeof(float));

        // Accumulate embedding contributions
        for (int j = 0; j < d; j++) {
            uint8_t byte = xi[j];
            const float *e = emb + ((long)j * 256 + byte) * D;
            const float *W1_block = W1 + (long)j * D * H;
            for (int di = 0; di < D; di++) {
                float ev = e[di];
                const float *w_row = W1_block + (long)di * H;
                for (int h = 0; h < H; h++) {
                    hidden[h] += ev * w_row[h];
                }
            }
        }

        // ReLU
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            scores[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(hidden); free(scores);
    if (x) free(x);
    return 0;
}
```