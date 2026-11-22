```hypothesis
Propose "label-embed": a shallow network with per-position embeddings (16-dim) for the 16-byte context, concatenated, a 64-unit ReLU hidden layer, and a 65-class output. This captures positional character n-gram statistics more efficiently than one-hot linear models, should exceed label-linear's 0.3375 accuracy, and remains fast due to small size (~86k params). If dev accuracy ≤0.3375 or inference cost (Ir/item) exceeds label-linear's 7359 by >2×, the approach fails.
```

```python
"""label-embed: per-position embeddings + 1-hidden-layer ReLU MLP for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Weights format: magic 0x454D4245, int32 d, E, H, C; float32 embeddings[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
E = 16          # embedding dimension per position
H = 64          # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 256
EPOCHS = 30

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

    # Initialize parameters
    # embeddings: list of d arrays, each (256, E)
    embeddings = [(rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32) for _ in range(d)]
    # W1: (d*E, H)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # W2: (H, C)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = embeddings + [W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    def forward(X_batch):
        # X_batch: (B, d) uint8
        B = X_batch.shape[0]
        # concat embeddings: (B, d*E)
        concat = np.empty((B, d * E), np.float32)
        for j in range(d):
            concat[:, j*E:(j+1)*E] = embeddings[j][X_batch[:, j]]
        # hidden = ReLU(concat @ W1 + b1)
        hidden = concat @ W1 + b1
        np.maximum(hidden, 0, out=hidden)
        # logits = hidden @ W2 + b2
        logits = hidden @ W2 + b2
        return concat, hidden, logits

    def backward(concat, hidden, logits, y_batch, B):
        # logits: (B, C), y_batch: (B,)
        # softmax cross-entropy gradient
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits, dtype=np.float32)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(B), y_batch] -= 1.0
        probs /= B
        # grad W2 = hidden.T @ probs
        grad_W2 = hidden.T @ probs
        grad_b2 = probs.sum(axis=0)
        # grad hidden = probs @ W2.T
        grad_hidden = probs @ W2.T
        # ReLU mask
        grad_hidden[hidden == 0] = 0.0
        # grad W1 = concat.T @ grad_hidden
        grad_W1 = concat.T @ grad_hidden
        grad_b1 = grad_hidden.sum(axis=0)
        # grad embeddings
        grad_embeddings = []
        for j in range(d):
            g = np.zeros((256, E), np.float32)
            # grad_hidden slice for position j: grad_hidden * W1[j*E:(j+1)*E, :]^T?
            # Actually, concat[:, j*E:(j+1)*E] = embeddings[j][X_batch[:, j]]
            # So gradient w.r.t. embeddings[j] is scatter-add of grad_hidden @ W1[j*E:(j+1)*E, :].T
            # grad_concat_slice = grad_hidden @ W1[j*E:(j+1)*E, :].T  (B, E)
            grad_concat_slice = grad_hidden @ W1[j*E:(j+1)*E, :].T
            np.add.at(g, X_batch[:, j], grad_concat_slice)
            grad_embeddings.append(g)
        return grad_embeddings, grad_W1, grad_b1, grad_W2, grad_b2

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            B = len(yb)
            concat, hidden, logits = forward(Xb)
            grads = backward(concat, hidden, logits, yb, B)
            # Adam update
            t += 1
            all_grads = grads[0] + [grads[1], grads[2], grads[3], grads[4]]
            for i, (p, g) in enumerate(zip(params, all_grads)):
                g = g.astype(np.float32)
                if i < d:  # embeddings
                    g += L2 * p
                elif i == d:  # W1
                    g += L2 * p
                elif i == d+2:  # W2
                    g += L2 * p
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                m_hat = m[i] / (1 - 0.9**t)
                v_hat = v[i] / (1 - 0.999**t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    # header: magic, d, E, H, C
    out = struct.pack("<5i", 0x454D4245, d, E, H, C)
    # embeddings
    for j in range(d):
        out += embeddings[j].astype("<f4").tobytes()
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
/* label-embed inference: ./infer weights.bin in.rec N
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65) { fprintf(stderr, "unexpected d or C\n"); return 3; }
    long expected = 20L
        + 4L * d * 256 * E
        + 4L * (d * E) * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *row = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, row, E * sizeof(float));
        }

        // hidden = ReLU(concat * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int idx = 0; idx < d * E; idx++) sum += concat[idx] * W1[(long)idx * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // logits = hidden * W2 + b2
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