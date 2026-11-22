```hypothesis
Family: label-emb-mlp (embedding + one hidden layer). Categorical features get learned embeddings (size 4) instead of one-hot, ordinal features are z-scored and kept as scalars. This reduces input dimension from ~2000 (one-hot) to 6 + 8*4 = 38, allowing a wider hidden layer (32 units) within similar parameter budget. It should capture categorical similarities and beat label-linear (0.8524) while staying faster than label-mlp16. Failure: dev accuracy ≤ 0.8524 or inference cost ≥ label-mlp16.
```

```python
"""label-emb-mlp: embedding + one hidden layer for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format (little-endian):
  int32 magic=0x454D424C, d, E, H, C, n_ord, n_cat
  float32 mu[n_ord], sigma[n_ord]
  for each categorical feature (in column order): float32 emb[256*E]
  float32 W1[D*H], b1[H], W2[H*C], b2[C]  where D = n_ord + n_cat*E
"""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
E = 4          # embedding size per categorical feature
H = 32         # hidden layer size


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
    C = card["classes"]
    types = card["input"]["types"]
    d = len(types)

    # Identify ordinal / categorical columns
    ord_idx = [j for j, t in enumerate(types) if t == "ordinal"]
    cat_idx = [j for j, t in enumerate(types) if t == "categorical"]
    n_ord, n_cat = len(ord_idx), len(cat_idx)
    D = n_ord + n_cat * E

    rng = np.random.default_rng(SEED)

    # Compute ordinal statistics on full training set
    mu = X_bytes[:, ord_idx].mean(axis=0).astype(np.float32)
    sigma = X_bytes[:, ord_idx].std(axis=0).astype(np.float32)
    sigma[sigma == 0] = 1.0

    # Initialize embeddings for each categorical feature: (256, E)
    cat_emb = []
    for _ in range(n_cat):
        emb = rng.uniform(-1, 1, (256, E)).astype(np.float32) * math.sqrt(6.0 / (256 + E))
        cat_emb.append(emb)

    # Initialize MLP weights
    W1 = rng.uniform(-1, 1, (D, H)).astype(np.float32) * math.sqrt(6.0 / D)
    b1 = np.zeros(H, np.float32)
    W2 = rng.uniform(-1, 1, (H, C)).astype(np.float32) * math.sqrt(6.0 / H)
    b2 = np.zeros(C, np.float32)

    params = [W1, b1, W2, b2]
    cat_emb_params = cat_emb  # list of arrays
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    cat_emb_state = [(np.zeros_like(e), np.zeros_like(e)) for e in cat_emb_params]

    n = X_bytes.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    def forward(X_batch):
        """Return (logits, hidden_pre, hidden_post, z) for backprop."""
        z_parts = []
        # ordinal part
        if n_ord:
            z_ord = ((X_batch[:, ord_idx].astype(np.float32) - mu) / sigma).astype(np.float32)
            z_parts.append(z_ord)
        # categorical embeddings
        for ci, cj in enumerate(cat_idx):
            z_parts.append(cat_emb_params[ci][X_batch[:, cj]])
        z = np.concatenate(z_parts, axis=1)  # (batch, D)
        a = z @ W1 + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
        return logits, a, h, z

    def softmax_grad(logits, y_batch):
        """Return gradient of cross-entropy w.r.t logits."""
        P = np.exp(logits - logits.max(axis=1, keepdims=True))
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(y_batch)), y_batch] -= 1.0
        return (P / len(y_batch)).astype(np.float32)

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_bytes[idx], y[idx]
            logits, a, h, z = forward(Xb)
            g_logits = softmax_grad(logits, yb)

            # Gradients for W2, b2
            g_W2 = h.T @ g_logits + L2 * W2
            g_b2 = g_logits.sum(axis=0)

            # Backprop to hidden layer
            g_h = g_logits @ W2.T
            g_a = g_h * (a > 0)

            # Gradients for W1, b1
            g_W1 = z.T @ g_a + L2 * W1
            g_b1 = g_a.sum(axis=0)

            # Backprop to embeddings and ordinal scaling (ordinal scaling is fixed, no grad)
            g_z = g_a @ W1.T  # (batch, D)

            # Split g_z into ordinal and categorical parts
            pos = 0
            if n_ord:
                pos += n_ord
            for ci in range(n_cat):
                g_emb = g_z[:, pos:pos + E]  # (batch, E)
                indices = Xb[:, cat_idx[ci]]  # (batch,)
                # Accumulate gradients for each embedding row
                grad_emb = np.zeros_like(cat_emb_params[ci])
                np.add.at(grad_emb, indices, g_emb)
                # Adam update for this embedding matrix
                m, v = cat_emb_state[ci]
                m *= 0.9; m += 0.1 * grad_emb
                v *= 0.999; v += 0.001 * grad_emb * grad_emb
                # Use global step counter for bias correction
                t = len(state)  # dummy, we'll use a global step
                cat_emb_params[ci] -= LR * m / (np.sqrt(v) + 1e-8)
                pos += E

            # Adam updates for MLP params
            grads = [g_W1, g_b1, g_W2, g_b2]
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * m / (np.sqrt(v) + 1e-8)

    # Pack weights
    magic = 0x454D424C  # 'EMBL'
    header = struct.pack("<6i", magic, d, E, H, C, n_ord, n_cat)
    body = header
    body += mu.astype("<f4").tobytes()
    body += sigma.astype("<f4").tobytes()
    for emb in cat_emb_params:
        body += emb.astype("<f4").tobytes()
    for p in params:
        body += p.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(body)


if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from header). Output: 1 byte class index.
 * Weights layout (little-endian):
 *   int32 magic=0x454D424C, d, E, H, C, n_ord, n_cat
 *   float32 mu[n_ord], sigma[n_ord]
 *   for each categorical feature: float32 emb[256*E]
 *   float32 W1[D*H], b1[H], W2[H*C], b2[C]  where D = n_ord + n_cat*E
 */
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
    if (nw < 28 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4], n_ord = hd[5], n_cat = hd[6];
    int D = n_ord + n_cat * E;
    long expected = 28L
        + 4L * n_ord * 2          // mu, sigma
        + 4L * n_cat * 256 * E    // embeddings
        + 4L * (D * H + H + H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mu = (const float *)(hd + 7);
    const float *sigma = mu + n_ord;
    const float *cat_emb = sigma + n_ord;
    const float *W1 = cat_emb + n_cat * 256 * E;
    const float *b1 = W1 + D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * D);
    float *a = malloc(sizeof(float) * H);
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build z vector
        int p = 0;
        for (int j = 0; j < d; j++) {
            // We need to know column types. Since we don't store them, we must reconstruct from training order.
            // The header does not store types array. We'll assume the same order as training:
            // ordinal columns first (in original column order), then categorical (in original column order).
            // But we don't have that mapping in weights. We must store it.
            // To keep it simple, we'll store a bitmap of ordinal columns in the header (but we didn't).
            // We'll change the format: store a 16-bit mask for ordinal columns (d <= 14).
            // Since we cannot change train.py now, we must infer from the fact that we know d=14 and the task.
            // This is a hack: we assume the first 6 columns are ordinal? Not safe.
            // We'll add a 16-bit mask after n_cat in the header. Let's adjust train.py and infer.c together.
            // But the answer must be consistent. We'll modify the header to include a uint16_t ord_mask.
            // We'll do that in the final code.
        }
    }
    return 0;
}
```

**Correction:** The C code above is incomplete because the header lacks the ordinal/categorical column mapping. We must store a bitmask in the header. Here is the complete, consistent pair:

```python
"""label-emb-mlp: embedding + one hidden layer for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format (little-endian):
  int32 magic=0x454D424C, d, E, H, C, n_ord, n_cat
  uint16 ord_mask (bit j set if column j is ordinal), uint16 padding
  float32 mu[n_ord], sigma[n_ord]
  for each categorical feature (in column order): float32 emb[256*E]
  float32 W1[D*H], b1[H], W2[H*C], b2[C]  where D = n_ord + n_cat*E
"""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
E = 4
H = 32


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
    C = card["classes"]
    types = card["input"]["types"]
    d = len(types)

    ord_idx = [j for j, t in enumerate(types) if t == "ordinal"]
    cat_idx = [j for j, t in enumerate(types) if t == "categorical"]
    n_ord, n_cat = len(ord_idx), len(cat_idx)
    D = n_ord + n_cat * E

    # Build ordinal mask (bit j = 1 if column j is ordinal)
    ord_mask = 0
    for j in ord_idx:
        ord_mask |= 1 << j

    rng = np.random.default_rng(SEED)

    mu = X_bytes[:, ord_idx].mean(axis=0).astype(np.float32)
    sigma = X_bytes[:, ord_idx].std(axis=0).astype(np.float32)
    sigma[sigma == 0] = 1.0

    cat_emb = []
    for _ in range(n_cat):
        emb = rng.uniform(-1, 1, (256, E)).astype(np.float32) * math.sqrt(6.0 / (256 + E))
        cat_emb.append(emb)

    W1 = rng.uniform(-1, 1, (D, H)).astype(np.float32) * math.sqrt(6.0 / D)
    b1 = np.zeros(H, np.float32)
    W2 = rng.uniform(-1, 1, (H, C)).astype(np.float32) * math.sqrt(6.0 / H)
    b2 = np.zeros(C, np.float32)

    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    cat_emb_state = [(np.zeros_like(e), np.zeros_like(e)) for e in cat_emb]

    n = X_bytes.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    def forward(X_batch):
        z_parts = []
        if n_ord:
            z_ord = ((X_batch[:, ord_idx].astype(np.float32) - mu) / sigma).astype(np.float32)
            z_parts.append(z_ord)
        for ci, cj in enumerate(cat_idx):
            z_parts.append(cat_emb[ci][X_batch[:, cj]])
        z = np.concatenate(z_parts, axis=1)
        a = z @ W1 + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
        return logits, a, h, z

    def softmax_grad(logits, y_batch):
        P = np.exp(logits - logits.max(axis=1, keepdims=True))
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(y_batch)), y_batch] -= 1.0
        return (P / len(y_batch)).astype(np.float32)

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_bytes[idx], y[idx]
            logits, a, h, z = forward(Xb)
            g_logits = softmax_grad(logits, yb)

            g_W2 = h.T @ g_logits + L2 * W2
            g_b2 = g_logits.sum(axis=0)

            g_h = g_logits @ W2.T
            g_a = g_h * (a > 0)

            g_W1 = z.T @ g_a + L2 * W1
            g_b1 = g_a.sum(axis=0)

            g_z = g_a @ W1.T
            pos = 0
            if n_ord:
                pos += n_ord
            for ci in range(n_cat):
                g_emb = g_z[:, pos:pos + E]
                indices = Xb[:, cat_idx[ci]]
                grad_emb = np.zeros_like(cat_emb[ci])
                np.add.at(grad_emb, indices, g_emb)
                m, v = cat_emb_state[ci]
                m *= 0.9; m += 0.1 * grad_emb
                v *= 0.999; v += 0.001 * grad_emb * grad_emb
                cat_emb[ci] -= LR * m / (np.sqrt(v) + 1e-8)
                pos += E

            grads = [g_W1, g_b1, g_W2, g_b2]
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * m / (np.sqrt(v) + 1e-8)

    # Pack weights
    magic = 0x454D424C
    header = struct.pack("<7iH", magic, d, E, H, C, n_ord, n_cat, ord_mask)
    body = header
    body += mu.astype("<f4").tobytes()
    body += sigma.astype("<f4").tobytes()
    for emb in cat_emb:
        body += emb.astype("<f4").tobytes()
    for p in params:
        body += p.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(body)


if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: 1 byte class index.
 * Weights layout (little-endian):
 *   int32 magic=0x454D424C, d, E, H, C, n_ord, n_cat
 *   uint16 ord_mask, uint16 padding
 *   float32 mu[n_ord], sigma[n_ord]
 *   for each categorical feature (in column order): float32 emb[256*E]
 *   float32 W1[D*H], b1[H], W2[H*C], b2[C]  where D = n_ord + n_cat*E
 */
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
    if (nw < 32) { fprintf(stderr, "weights too small\n"); return 3; }

    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x454D424C) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4], n_ord = hd[5], n_cat = hd[6];
    uint16_t ord_mask = *(uint16_t *)(hd + 7);
    int D = n_ord + n_cat * E;

    const float *mu = (const float *)(hd + 8);  // after 7 int32 + 1 uint16 + padding = 8*4 = 32 bytes
    const float *sigma = mu + n_ord;
    const float *cat_emb = sigma + n_ord;
    const float *W1 = cat_emb + n_cat * 256 * E;
    const float *b1 = W1 + D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + H * C;

    long expected = 32L + 4L * n_ord * 2 + 4L * n_cat * 256 * E + 4L * (D * H + H + H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * D);
    float *a = malloc(sizeof(float) * H);
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        int p = 0;
        for (int j = 0; j < d; j++) {
            if (ord_mask & (1 << j)) {
                z[p++] = (xi[j] - mu[j - __builtin_popcount(ord_mask & ((1 << j) - 1))]) / sigma[__builtin_popcount(ord_mask & ((1 << j) - 1))];
            } else {
                int cat_idx = __builtin_popcount(ord_mask & ((1 << j) - 1)) - (j - __builtin_popcount(ord_mask & ((1 << j) - 1)));
                const float *emb = cat_emb + cat_idx * 256 * E;
                for (int e = 0; e < E; e++) z[p++] = emb[xi[j] * E + e];
            }
        }

        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < D; v++) sum += z[v] * W1[v * H + u];
            a[u] = sum;
            h[u] = sum > 0 ? sum : 0;
        }

        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[u * C + c];
            logits[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```