```hypothesis
Family: label-poly2 (degree-2 polynomial with hashed cross features).
Why: The task predicts the next character from 16 previous bytes. A linear model (label-linear) ignores interactions between positions; a degree-2 polynomial captures pairwise interactions (e.g., bigrams) which are crucial for language. Hashing the cross features into a fixed-size table (M=4096) keeps the model small and inference fast while adding expressive power. We expect >0.35 accuracy with CPU cost slightly above label-linear.
Failure: If dev accuracy ≤0.34 or Ir/item >15000, the extra complexity does not pay off.
```

```python
"""label-poly2: linear + hashed degree-2 cross features for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, numpy only.
Weights format: magic 0x504F4C32, d, M, C; float32 T_lin[d][256][C], b_lin[C], W_cross[M][C]."""
import json, math, struct, sys

import numpy as np

M = 4096                 # hash buckets for cross features
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def hash_pair(i, j, b1, b2, M):
    # deterministic 32-bit hash, matches C implementation
    x = (i << 24) | (j << 16) | (b1 << 8) | b2
    x = ((x ^ (x >> 16)) * 0x45d9f3b) & 0xFFFFFFFF
    x = ((x ^ (x >> 16)) * 0x45d9f3b) & 0xFFFFFFFF
    x = (x ^ (x >> 16)) & 0xFFFFFFFF
    return x % M


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X_bytes.shape[0]

    rng = np.random.default_rng(SEED)

    # --- Linear part: per-position one-hot over observed values ---
    enc = []          # list of (256, num_vals) one-hot matrices
    vals_list = []    # list of unique values per position
    for j in range(d):
        vals = np.unique(X_bytes[:, j])
        vals_list.append(vals)
        E = np.zeros((256, len(vals)), dtype=np.float32)
        E[vals, np.arange(len(vals))] = 1.0
        enc.append(E)
    # total linear features
    total_lin = sum(E.shape[1] for E in enc)

    # --- Cross part: precompute hash indices for all pairs (i<j) ---
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    P = len(pairs)                     # 120 for d=16
    cross_idx = np.empty((n, P), dtype=np.int32)
    for idx, (i, j) in enumerate(pairs):
        cross_idx[:, idx] = [hash_pair(i, j, X_bytes[k, i], X_bytes[k, j], M) for k in range(n)]

    # --- Build linear design matrix Z (n, total_lin) ---
    Z = np.concatenate([E[X_bytes[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)

    # --- Parameters ---
    W_lin = (rng.uniform(-1, 1, (total_lin, C)) * math.sqrt(6.0 / total_lin)).astype(np.float32)
    b_lin = np.zeros(C, dtype=np.float32)
    W_cross = (rng.uniform(-1, 1, (M, C)) * math.sqrt(6.0 / M)).astype(np.float32)

    params = [W_lin, b_lin, W_cross]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # --- Training loop ---
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z = Z[idx]                      # (B, total_lin)
            cid = cross_idx[idx]            # (B, P)
            yy = y[idx]                     # (B,)

            # Forward
            S = z @ W_lin + b_lin           # (B, C)
            # Add cross features: gather and sum
            cross_contrib = W_cross[cid].sum(axis=1)   # (B, C)
            S += cross_contrib

            # Softmax gradient
            S_max = S.max(axis=1, keepdims=True)
            P = np.exp(S - S_max)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            grad_S = (P / len(yy)).astype(np.float32)   # (B, C)

            # Gradients
            grad_W_lin = z.T @ grad_S + L2 * W_lin
            grad_b_lin = grad_S.sum(axis=0)
            # Cross gradients: accumulate per bucket
            grad_W_cross = np.zeros_like(W_cross)
            flat_cid = cid.ravel()
            repeated_grad = np.repeat(grad_S, P, axis=0)   # (B*P, C)
            np.add.at(grad_W_cross, flat_cid, repeated_grad)
            grad_W_cross += L2 * W_cross

            grads = [grad_W_lin, grad_b_lin, grad_W_cross]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # --- Export weights ---
    # Precompute T_lin: for each position, (256, C) = E @ W_lin_slice
    T_lin = np.zeros((d, 256, C), dtype=np.float32)
    off = 0
    for j, E in enumerate(enc):
        T_lin[j] = E @ W_lin[off:off + E.shape[1]]
        off += E.shape[1]

    magic = 0x504F4C32  # "POL2"
    out = struct.pack("<4i", magic, d, M, C)
    out += T_lin.astype("<f4").tobytes()
    out += b_lin.astype("<f4").tobytes()
    out += W_cross.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-poly2 inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: one byte class index per record. */
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

/* Same hash as in train.py */
static uint32_t hash_pair(uint32_t i, uint32_t j, uint32_t b1, uint32_t b2, uint32_t M) {
    uint32_t x = (i << 24) | (j << 16) | (b1 << 8) | b2;
    x = ((x ^ (x >> 16)) * 0x45d9f3b) & 0xFFFFFFFF;
    x = ((x ^ (x >> 16)) * 0x45d9f3b) & 0xFFFFFFFF;
    x = (x ^ (x >> 16)) & 0xFFFFFFFF;
    return x % M;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x504F4C32) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], M = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * C + C + (long)M * C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *T_lin = (const float *)(hd + 4);
    const float *b_lin = T_lin + (long)d * 256 * C;
    const float *W_cross = b_lin + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(acc, b_lin, C * sizeof(float));

        // Linear part
        for (int j = 0; j < d; j++) {
            const float *t = T_lin + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) acc[c] += t[c];
        }

        // Cross part: all pairs i<j
        for (int ii = 0; ii < d; ii++) {
            for (int jj = ii + 1; jj < d; jj++) {
                uint32_t h = hash_pair(ii, jj, xi[ii], xi[jj], M);
                const float *w = W_cross + (long)h * C;
                for (int c = 0; c < C; c++) acc[c] += w[c];
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```