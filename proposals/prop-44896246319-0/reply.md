```hypothesis
Family: label-embmlp (embedding MLP). Learned embeddings for categorical columns (E=4) plus normalized ordinal columns feed a 32-unit ReLU hidden layer. This reduces input dimensionality vs. one-hot, captures interactions, and stays fast. Fails if dev accuracy ≤ label-linear (0.8524) or inference cost ≥ label-mlp16 (2325 Ir/item).
```
```python
"""label-embmlp: embedding MLP for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Encoding: ordinal -> z-scored (fixed), categorical -> learned embeddings (size E=4) with a special unseen token.
Architecture: concat embeddings -> ReLU hidden (H=32) -> linear output (C=2).
Adam lr=1e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Weights format (little-endian):
  magic(0x454D424D), d, H, C, E, type_mask(14 bits: 1=categorical,0=ordinal)
  T[d][256][W_j] float32 (W_j=1 for ordinal, E for categorical)
  W1[total_W][H], b1[H], W2[H][C], b2[C] float32"""
import json, math, struct, sys

import numpy as np

E = 4
H = 32
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-3, 1e-4, 64, 600000, 100
MAGIC = 0x454D424D  # 'EMBM'


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
    C = card["classes"]
    types = card["input"]["types"]
    d = X.shape[1]

    rng = np.random.default_rng(SEED)

    # Build lookup tables T[j] of shape (256, W_j)
    T_list = []
    W_j_list = []
    type_mask = 0
    for j, t in enumerate(types):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            table = ((np.arange(256, dtype=np.float64) - mu) / sd).astype(np.float32)[:, None]  # (256,1)
            W_j = 1
            type_mask |= (0 << j)
        else:
            # categorical: map each seen byte value to an index, plus unseen (255) as special
            seen = np.unique(X[:, j])
            # ensure 255 is not in seen (per card 255=unseen), but handle anyway
            if 255 in seen:
                seen = seen[seen != 255]
            K = len(seen)
            # mapping from byte value to index 0..K (K is for unseen/255)
            map_arr = np.full(256, K, dtype=np.int32)
            map_arr[seen] = np.arange(K, dtype=np.int32)
            # learn embeddings for K+1 categories
            emb = (rng.uniform(-1, 1, (K + 1, E)) * math.sqrt(6.0 / (K + 1))).astype(np.float32)
            table = emb[map_arr]  # (256, E)
            W_j = E
            type_mask |= (1 << j)
        T_list.append(table)
        W_j_list.append(W_j)

    total_W = sum(W_j_list)
    # Build full input matrix Z (n, total_W)
    Z_parts = [T_list[j][X[:, j]] for j in range(d)]
    Z = np.concatenate(Z_parts, axis=1).astype(np.float32)
    n, k = Z.shape
    assert k == total_W

    # Initialize weights
    W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # forward
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            g = softmax_grad(logits, yy)
            # backward
            grad_W2 = h.T @ g + L2 * W2
            grad_b2 = g.sum(0)
            gh = g @ W2.T * (a > 0)
            grad_W1 = z.T @ gh + L2 * W1
            grad_b1 = gh.sum(0)
            grads = [grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    # header: magic, d, H, C, E, type_mask
    header = struct.pack("<6i", MAGIC, d, H, C, E, type_mask)
    # tables
    table_bytes = b"".join(T.astype("<f4").tobytes() for T in T_list)
    # params
    param_bytes = b"".join(p.astype("<f4").tobytes() for p in params)
    open(sys.argv[4], "wb").write(header + table_bytes + param_bytes)


if __name__ == "__main__":
    main()
```
```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index.
 * Weights format (little-endian):
 *   int32 magic(0x454D424D), d, H, C, E, type_mask
 *   float32 T[d][256][W_j]  (W_j=1 for ordinal, E for categorical)
 *   float32 W1[total_W][H], b1[H], W2[H][C], b2[C] */
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
    if (nw < 24 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], E = hd[4], type_mask = hd[5];
    if (d != 14 || C != 2 || H <= 0 || E <= 0) { fprintf(stderr, "bad header values\n"); return 3; }

    // Compute W_j and total_W
    int W_j[14], total_W = 0;
    for (int j = 0; j < d; j++) {
        W_j[j] = (type_mask >> j) & 1 ? E : 1;
        total_W += W_j[j];
    }

    // Expected weight size
    long expected = 24; // header
    for (int j = 0; j < d; j++) expected += 256L * W_j[j] * 4;
    expected += (long)total_W * H * 4 + H * 4 + (long)H * C * 4 + C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *ptr = (const float *)(hd + 6); // after 6 int32
    // Tables: T[j] points to 256 * W_j floats
    const float *T[14];
    for (int j = 0; j < d; j++) {
        T[j] = ptr;
        ptr += 256 * W_j[j];
    }
    const float *W1 = ptr; ptr += (long)total_W * H;
    const float *b1 = ptr; ptr += H;
    const float *W2 = ptr; ptr += (long)H * C;
    const float *b2 = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * total_W);
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build concatenated embedding z
        int offset = 0;
        for (int j = 0; j < d; j++) {
            const float *row = T[j] + xi[j] * W_j[j];
            for (int w = 0; w < W_j[j]; w++) z[offset++] = row[w];
        }

        // Hidden layer: h = ReLU(z @ W1 + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int w = 0; w < total_W; w++) sum += z[w] * W1[(long)w * H + u];
            h[u] = sum > 0.0f ? sum : 0.0f;
        }

        // Output logits
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[(long)u * C + c];
            logits[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```