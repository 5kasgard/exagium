```hypothesis
Family: label-embed (embedding MLP with precomputed position-wise tables).
Why: Character-level LM benefits from dense embeddings capturing similarity between characters (e.g., 'a'/'b', 'A'/'a'), which one-hot linear models cannot. Precomputing Emb@W1 per position yields fast inference (O(d*H + H*C)) similar to label-linear but with better generalization. With E=32, H=128, we expect >0.35 accuracy at ~10k Ir/item.
Failure: If accuracy <= label-linear (0.3375) or instructions > 2x label-linear, the embedding advantage doesn't materialize for this data scale.
```
```python
"""label-embed: Embedding MLP for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Architecture: per-position embeddings (256 x E) -> concat (d*E) -> ReLU hidden (H) -> output (C).
Exported as precomputed tables T[d][256][H] = Emb_j @ W1_j, plus b1, W2, b2."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
E = 32          # embedding dimension per position
H = 128         # hidden layer size
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
    """Return gradient of cross-entropy w.r.t logits: (softmax - one_hot) / n."""
    n = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits, dtype=np.float32)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    return probs / n


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert d == 16

    # Load all training data
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # Emb: list of d arrays, each (256, E)
    Emb = [(rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32) for _ in range(d)]
    # W1: (d*E, H)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # W2: (H, C)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [Emb, W1, b1, W2, b2]
    # Adam state: m, v for each param array
    state = []
    for p in params:
        if isinstance(p, list):
            state.append([(np.zeros_like(arr), np.zeros_like(arr)) for arr in p])
        else:
            state.append((np.zeros_like(p), np.zeros_like(p)))

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb = X_bytes[idx]
            yb = y[idx]
            bs = len(yb)

            # Forward
            # x: (bs, d*E)
            x = np.zeros((bs, d * E), np.float32)
            for j in range(d):
                x[:, j*E:(j+1)*E] = Emb[j][Xb[:, j]]
            h = x @ W1 + b1
            h_relu = np.maximum(h, 0)
            logits = h_relu @ W2 + b2

            # Loss gradient
            dlogits = softmax_cross_entropy_grad(logits, yb)

            # Backward
            dW2 = h_relu.T @ dlogits
            db2 = dlogits.sum(axis=0)
            dh_relu = dlogits @ W2.T
            dh = dh_relu * (h > 0)
            dW1 = x.T @ dh
            db1 = dh.sum(axis=0)
            dx = dh @ W1.T

            # Embedding gradients
            dEmb = [np.zeros_like(Emb[j]) for j in range(d)]
            for j in range(d):
                np.add.at(dEmb[j], Xb[:, j], dx[:, j*E:(j+1)*E])

            # Add L2 regularization gradients (not on biases)
            for j in range(d):
                dEmb[j] += L2 * Emb[j]
            dW1 += L2 * W1
            dW2 += L2 * W2

            # Scale by batch size
            scale = 1.0 / bs
            for j in range(d):
                dEmb[j] *= scale
            dW1 *= scale; db1 *= scale
            dW2 *= scale; db2 *= scale

            # Adam update
            t += 1
            grads = [dEmb, dW1, db1, dW2, db2]
            for i, (p, gr) in enumerate(zip(params, grads)):
                if isinstance(p, list):
                    for j, (pij, grij) in enumerate(zip(p, gr)):
                        m, v = state[i][j]
                        m *= 0.9; m += 0.1 * grij
                        v *= 0.999; v += 0.001 * grij * grij
                        m_hat = m / (1 - 0.9 ** t)
                        v_hat = v / (1 - 0.999 ** t)
                        pij -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
                else:
                    m, v = state[i]
                    m *= 0.9; m += 0.1 * gr
                    v *= 0.999; v += 0.001 * gr * gr
                    m_hat = m / (1 - 0.9 ** t)
                    v_hat = v / (1 - 0.999 ** t)
                    p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export: compute T[j][v] = Emb[j][v] @ W1[j*E:(j+1)*E]
    T = np.zeros((d, 256, H), np.float32)
    for j in range(d):
        W1_j = W1[j*E:(j+1)*E, :]
        T[j] = Emb[j] @ W1_j

    # Pack weights
    # Header: magic, d, E, H, C (5 int32)
    MAGIC = 0x454D4231  # "EMB1"
    header = struct.pack("<5i", MAGIC, d, E, H, C)
    # T: d * 256 * H float32
    # b1: H float32
    # W2: H * C float32
    # b2: C float32
    out = header + T.astype("<f4").tobytes() + b1.astype("<f4").tobytes() + W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-embed: Embedding MLP with precomputed tables.
 * ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte class index. */
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
    if (nw < 20 || hd[0] != 0x454D4231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C < 1 || C > 256 || H < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *T = (const float *)(hd + 5);           // [d][256][H]
    const float *b1 = T + (long)d * 256 * H;            // [H]
    const float *W2 = b1 + H;                           // [H][C]
    const float *b2 = W2 + (long)H * C;                 // [C]

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * H);
    float *sc = malloc(sizeof(float) * C);
    if (!acc || !sc) { fprintf(stderr, "malloc failed\n"); exit(3); }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // acc = b1
        memcpy(acc, b1, H * sizeof(float));
        // Add T[j][xi[j]] for each position
        for (int j = 0; j < d; j++) {
            const float *tj = T + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) acc[h] += tj[h];
        }
        // ReLU
        for (int h = 0; h < H; h++) if (acc[h] < 0.0f) acc[h] = 0.0f;
        // Output layer: sc = b2 + acc @ W2
        memcpy(sc, b2, C * sizeof(float));
        for (int h = 0; h < H; h++) {
            float ah = acc[h];
            if (ah == 0.0f) continue;
            const float *w2h = W2 + (long)h * C;
            for (int c = 0; c < C; c++) sc[c] += ah * w2h[c];
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```