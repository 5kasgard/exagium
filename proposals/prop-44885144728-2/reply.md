```hypothesis
We propose a factorized linear model (label-factorized) that learns a low-dimensional embedding for each position and character, then combines them via a shared output matrix. This captures pairwise interactions through the embedding space, improving over the purely linear model while keeping inference fast (O(d*D + D*C) per item). The embedding dimension D is chosen on the tuning split (last 20% of train). If dev accuracy does not exceed label-linear's 0.3375, the approach fails.
```

```python
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (except D which is tuned)
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS_TUNE = 50
MAX_EPOCHS_FULL = 100
D_CANDIDATES = [16, 32, 64, 128]  # embedding dimensions to try


def read_records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_model(X, y, d, C, D, seed, lr, l2, batch, epochs):
    """Train factorized model: E[d,65,D], W[D,C], b[C]. Returns trained params."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    # Initialize parameters
    # He init for embeddings (input dim 65), Xavier for output (input dim D)
    E = (rng.normal(0, math.sqrt(2.0 / 65), (d, 65, D))).astype(np.float32)
    W = (rng.normal(0, math.sqrt(2.0 / D), (D, C))).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    params = [E, W, b]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb = X[idx]          # (batch, d)
            yb = y[idx]          # (batch,)
            bs = xb.shape[0]

            # Forward
            hidden = np.zeros((bs, D), dtype=np.float32)
            for j in range(d):
                hidden += E[j, xb[:, j], :]   # (bs, D)
            scores = hidden @ W + b            # (bs, C)

            # Softmax + cross-entropy gradient
            max_s = scores.max(axis=1, keepdims=True)
            exp_s = np.exp(scores - max_s)
            probs = exp_s / exp_s.sum(axis=1, keepdims=True)
            dscores = probs
            dscores[np.arange(bs), yb] -= 1.0
            dscores /= bs

            # Gradients
            dW = hidden.T @ dscores + l2 * W
            db = dscores.sum(axis=0)
            dhidden = dscores @ W.T
            dE = np.zeros_like(E)
            for j in range(d):
                np.add.at(dE[j], xb[:, j], dhidden)
            dE /= bs
            dE += l2 * E

            grads = [dE, dW, db]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return E, W, b


def evaluate(X, y, E, W, b):
    """Return accuracy on given data."""
    n = X.shape[0]
    d = X.shape[1]
    hidden = np.zeros((n, E.shape[2]), dtype=np.float32)
    for j in range(d):
        hidden += E[j, X[:, j], :]
    scores = hidden @ W + b
    preds = scores.argmax(axis=1)
    return (preds == y).mean()


def write_weights(path, d, D, C, mapping, E, W, b):
    """Write weights.bin with format:
       magic(4), d, D, C, mapping[256], E[d,65,D], W[D,C], b[C] all little-endian."""
    magic = 0x46414354  # 'TCAF' in little-endian
    header = struct.pack("<4i", magic, d, D, C)
    # mapping is 256 uint8
    map_bytes = mapping.astype(np.uint8).tobytes()
    # E: d * 65 * D floats
    e_bytes = E.astype("<f4").tobytes()
    # W: D * C floats
    w_bytes = W.astype("<f4").tobytes()
    # b: C floats
    b_bytes = b.astype("<f4").tobytes()
    with open(path, "wb") as f:
        f.write(header)
        f.write(map_bytes)
        f.write(e_bytes)
        f.write(w_bytes)
        f.write(b_bytes)


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    class_chars = card["class_chars"]

    # Build byte->class mapping (0..64)
    mapping = np.zeros(256, dtype=np.uint8)
    for i, ch in enumerate(class_chars):
        mapping[ord(ch)] = i

    # Load data
    X_bytes = np.frombuffer(b"".join(read_records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in read_records(sys.argv[3])], dtype=np.int64)
    # Map input bytes to class indices
    X = mapping[X_bytes]
    N = X.shape[0]

    # Tuning split: last 20%
    split = int(0.8 * N)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]

    # Grid search over D
    best_D = None
    best_acc = -1.0
    for D in D_CANDIDATES:
        E, W, b = train_model(X_train, y_train, d, C, D, seed=SEED + D,
                              lr=LR, l2=L2, batch=BATCH, epochs=MAX_EPOCHS_TUNE)
        acc = evaluate(X_tune, y_tune, E, W, b)
        print(f"D={D}, tuning accuracy={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_D = D

    print(f"Best D={best_D} (acc={best_acc:.4f}), retraining on full data")
    E, W, b = train_model(X, y, d, C, best_D, seed=42,
                          lr=LR, l2=L2, batch=BATCH, epochs=MAX_EPOCHS_FULL)

    write_weights(sys.argv[4], d, best_D, C, mapping, E, W, b)


if __name__ == "__main__":
    main()
```

```c
/* Factorized linear model (label-factorized) for next-char prediction.
 * ./infer weights.bin in.rec N
 * weights.bin format: magic(0x46414354), d, D, C, map[256], E[d][65][D], W[D][C], b[C] (all little-endian float32 except map).
 * Each input record: d bytes. Output: 1 byte class index. */
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
    uint32_t k = buf[*pos] | (buf[*pos+1]<<8) | (buf[*pos+2]<<16) | ((uint32_t)buf[*pos+3]<<24);
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *r = buf + *pos + 4;
    *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, (k>>8)&255, (k>>16)&255, (k>>24)&255};
    fwrite(h, 1, 4, f);
    fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wbuf = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wbuf;
    if (nw < 16 || hd[0] != 0x46414354) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C != 65 || D <= 0 || D > 256) { fprintf(stderr, "bad header dims\n"); return 3; }

    // Expected size: header(16) + map(256) + E(d*65*D*4) + W(D*C*4) + b(C*4)
    long expected = 16 + 256 + (long)d * 65 * D * 4 + (long)D * C * 4 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const unsigned char *map = wbuf + 16;
    const float *E = (const float *)(wbuf + 16 + 256);
    const float *W = E + (long)d * 65 * D;
    const float *b = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = (float *)malloc(sizeof(float) * D);
    float *scores = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // hidden = 0
        for (int u = 0; u < D; u++) hidden[u] = 0.0f;

        // accumulate embeddings
        for (int j = 0; j < d; j++) {
            unsigned char idx = map[xi[j]];
            const float *e = E + ((long)j * 65 + idx) * D;
            for (int u = 0; u < D; u++) hidden[u] += e[u];
        }

        // scores = b + hidden * W
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int u = 0; u < D; u++) {
            float h = hidden[u];
            const float *wu = W + (long)u * C;
            for (int c = 0; c < C; c++) scores[c] += h * wu[c];
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```