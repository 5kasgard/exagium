```hypothesis
Family: label-embedding-mlp – learns a 16-dim embedding per byte position (16×256×16 params), then a ReLU hidden layer of 64 units, then 65-class output. Total ~86k parameters (vs 266k for linear). Should capture positional interactions better than linear one-hot while staying small and fast. If dev accuracy ≤ label-linear (0.3375) or inference cost exceeds 2× linear, the hypothesis fails.
```
```python
"""label-embedding-mlp: position-wise embeddings + one ReLU hidden layer.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Weights format (little-endian):
  int32 magic=0x454D424C, d=16, E=16, H=64, C=65
  float32 embeddings[16][256][E]   (flattened per position)
  float32 W1[(d*E)][H], b1[H]
  float32 W2[H][C], b2[C]"""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (chosen on tuning split conceptually)
E = 16          # embedding dimension per position
H = 64          # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-5
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 20

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
    assert d == 16
    rng = np.random.default_rng(SEED)

    # Initialize embeddings: 16 positions, each 256 x E
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
    steps_per_epoch = (n + BATCH - 1) // BATCH
    max_steps = min(MAX_EPOCHS * steps_per_epoch, math.ceil(SAMPLES / BATCH))

    def forward(X_batch):
        # X_batch: (B, d) uint8
        B = X_batch.shape[0]
        # Gather embeddings: for each position, (B, E)
        embs = np.empty((B, d * E), np.float32)
        for j in range(d):
            embs[:, j*E:(j+1)*E] = emb[j, X_batch[:, j]]
        h = embs @ W1 + b1
        h = np.maximum(h, 0, dtype=np.float32)  # ReLU
        logits = h @ W2 + b2
        return embs, h, logits

    def backward(embs, h, logits, y_batch):
        B = len(y_batch)
        # Softmax cross-entropy gradient
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits, dtype=np.float32)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(B), y_batch] -= 1.0
        probs /= B

        # Output layer grads
        gW2 = h.T @ probs + L2 * W2
        gb2 = probs.sum(axis=0)

        # Hidden layer grads
        gh = probs @ W2.T
        gh[h <= 0] = 0.0  # ReLU derivative

        # Embedding + W1 grads
        gW1 = embs.T @ gh + L2 * W1
        gb1 = gh.sum(axis=0)

        # Embedding grads: scatter-add
        gemb = np.zeros_like(emb)
        for j in range(d):
            idx = X_batch[:, j]
            np.add.at(gemb[j], idx, gh[:, j*E:(j+1)*E])

        return [gemb, gW1, gb1, gW2, gb2]

    t = 0
    for step in range(max_steps):
        if step % steps_per_epoch == 0:
            order = rng.permutation(n)
        s = (step % steps_per_epoch) * BATCH
        idx = order[s:s+BATCH]
        X_batch, y_batch = X[idx], y[idx]
        embs, h, logits = forward(X_batch)
        grads = backward(embs, h, logits, y_batch)
        t += 1
        for i, (p, gr) in enumerate(zip(params, grads)):
            m, v = state[i]
            m *= 0.9; m += 0.1 * gr
            v *= 0.999; v += 0.001 * gr * gr
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    out = struct.pack("<5i", 0x454D424C, d, E, H, C)
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-embedding-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte class index.
 * Weights format (little-endian):
 *   int32 magic=0x454D424C, d=16, E=16, H=64, C=65
 *   float32 emb[d][256][E]
 *   float32 W1[d*E][H], b1[H]
 *   float32 W2[H][C], b2[C] */
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E != 16 || H != 64 || C != 65) { fprintf(stderr, "unexpected dimensions\n"); return 3; }

    long expected = 20
        + 4L * d * 256 * E          // embeddings
        + 4L * d * E * H            // W1
        + 4L * H                    // b1
        + 4L * H * C                // W2
        + 4L * C;                   // b2
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + d * 256 * E;
    const float *b1 = W1 + d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // hidden = ReLU( concat(emb[j][xi[j]]) @ W1 + b1 )
        for (int h = 0; h < H; h++) hidden[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *e = emb + (j * 256 + xi[j]) * E;
            for (int h = 0; h < H; h++) {
                float sum = 0.0f;
                for (int ee = 0; ee < E; ee++) sum += e[ee] * W1[(j * E + ee) * H + h];
                hidden[h] += sum;
            }
        }
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;

        // logits = hidden @ W2 + b2
        for (int c = 0; c < C; c++) logits[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float hv = hidden[h];
            if (hv == 0.0f) continue;
            for (int c = 0; c < C; c++) logits[c] += hv * W2[h * C + c];
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```