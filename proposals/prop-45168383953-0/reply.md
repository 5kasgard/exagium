```hypothesis
Family: label-embmlp. Learn a 4‑dim embedding for each of the 14 byte columns, concatenate (56 dims), then a one‑hidden‑layer ReLU MLP (32 units) to 2‑class softmax. Embeddings capture nonlinear similarities in categorical/ordinal bins; MLP adds feature interactions. Beats label‑linear (0.8524) by modeling interactions. If dev accuracy ≤ 0.8524 or inference cost > 1000 Ir/item, the hypothesis fails.
```

```python
"""label-embmlp: per-column embeddings + one hidden ReLU layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
E_DIM = 4          # embedding dimension per column
H = 32             # hidden units
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

MAGIC = 0x454D424C  # "LBME" in little-endian


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

    # Parameters
    # embeddings: (d, 256, E_DIM)
    emb = (rng.uniform(-1, 1, (d, 256, E_DIM)) * math.sqrt(6.0 / 256)).astype(np.float32)
    D = d * E_DIM
    W1 = (rng.uniform(-1, 1, (D, H)) * math.sqrt(6.0 / D)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    shapes = [p.shape for p in params]
    flat_sizes = [p.size for p in params]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    def forward(X_batch):
        """X_batch (B, d) uint8 -> logits (B, C), and cache for backward."""
        B = X_batch.shape[0]
        # Gather embeddings: (d, B, E_DIM) -> (B, d, E_DIM) -> (B, D)
        idx = X_batch.T  # (d, B)
        emb_batch = emb[np.arange(d)[:, None], idx]  # (d, B, E_DIM)
        Z = emb_batch.transpose(1, 0, 2).reshape(B, D)  # (B, D)
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
        return logits, Z, h, a

    def loss_and_grad(logits, y_batch):
        """Cross-entropy loss, return gradient w.r.t logits."""
        B = logits.shape[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(B), y_batch] -= 1.0
        return (probs / B).astype(np.float32)

    def backward(dlogits, Z, h, a, X_batch):
        """Compute gradients for all parameters."""
        B = dlogits.shape[0]
        # grad W2, b2
        gW2 = h.T @ dlogits + L2 * W2
        gb2 = dlogits.sum(axis=0)
        # grad h
        dh = dlogits @ W2.T
        # grad a (ReLU)
        da = dh * (a > 0)
        # grad W1, b1
        gW1 = Z.T @ da + L2 * W1
        gb1 = da.sum(axis=0)
        # grad Z
        dZ = da @ W1.T  # (B, D)
        # grad embeddings
        gemb = np.zeros_like(emb)
        dZ_split = dZ.reshape(B, d, E_DIM)  # (B, d, E_DIM)
        for j in range(d):
            idx = X_batch[:, j]  # (B,)
            # accumulate gradients for each category
            np.add.at(gemb[j], idx, dZ_split[:, j, :])
        # L2 on embeddings
        gemb += L2 * emb
        return [gemb, gW1, gb1, gW2, gb2]

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            X_batch, y_batch = X[idx], y[idx]
            logits, Z, h, a = forward(X_batch)
            dlogits = loss_and_grad(logits, y_batch)
            grads = backward(dlogits, Z, h, a, X_batch)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] *= 0.9; m[i] += 0.1 * gr
                v[i] *= 0.999; v[i] += 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    # header: magic, d, E_DIM, H, C (5 int32)
    header = struct.pack("<5i", MAGIC, d, E_DIM, H, C)
    # embeddings (d, 256, E_DIM) -> flat float32
    emb_flat = emb.astype("<f4").tobytes()
    W1_flat = W1.astype("<f4").tobytes()
    b1_flat = b1.astype("<f4").tobytes()
    W2_flat = W2.astype("<f4").tobytes()
    b2_flat = b2.astype("<f4").tobytes()
    out = header + emb_flat + W1_flat + b1_flat + W2_flat + b2_flat
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (one per column). Output: one byte class index. */
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
    int d = hd[1], E_DIM = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d < 1 || E_DIM < 1 || H < 0) { fprintf(stderr, "bad header values\n"); return 3; }
    long D = (long)d * E_DIM;
    long expected = 20
        + 4L * (long)d * 256 * E_DIM        /* embeddings */
        + 4L * D * H                         /* W1 */
        + 4L * H                             /* b1 */
        + 4L * (long)H * C                   /* W2 */
        + 4L * C;                            /* b2 */
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E_DIM;
    const float *b1 = W1 + D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * H);
    float *s = malloc(sizeof(float) * C);
    if (!acc || !s) { fprintf(stderr, "malloc failed\n"); exit(3); }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* Hidden layer: acc = b1 + sum_j emb[j][xi[j]] * W1_col */
        for (int h = 0; h < H; h++) acc[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E_DIM;
            for (int ed = 0; ed < E_DIM; ed++) {
                float ev = e[ed];
                const float *w = W1 + ((long)j * E_DIM + ed) * H;
                for (int h = 0; h < H; h++) acc[h] += ev * w[h];
            }
        }
        /* ReLU */
        for (int h = 0; h < H; h++) if (acc[h] < 0.0f) acc[h] = 0.0f;

        /* Output layer */
        for (int c = 0; c < C; c++) s[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float ah = acc[h];
            const float *w = W2 + (long)h * C;
            for (int c = 0; c < C; c++) s[c] += ah * w[c];
        }

        /* Argmax (first max wins) */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (s[c] > s[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```