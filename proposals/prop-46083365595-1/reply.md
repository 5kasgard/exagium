```hypothesis
Family: label-poly2 — degree-2 polynomial features with signed feature hashing to 2048 dimensions, then linear classification (Adam, same schedule as label-linear). Why: 8×8 digit pixels have strong pairwise interactions (strokes, corners); hashing keeps inference a single linear pass over ~2k features (O(d²) feature compute + O(D) dot product). Beats linear if interactions matter; fails (≤ linear accuracy) if collisions drown signal or 2048 dims are too small.
```
```python
"""label-poly2: degree-2 polynomial with feature hashing to fixed width, then linear.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x504F4C32, d, D, C; float32 W[D][C], b[C]."""
import json, math, struct, sys, zlib

import numpy as np

D = 2048          # hash dimension
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def hash1(j, v, seed):
    # hash (feature_index, value) -> (index, sign)
    h = zlib.crc32(struct.pack("<HH", j, v) + struct.pack("<I", seed)) & 0xffffffff
    return (h % D, 1.0 if (h >> 31) == 0 else -1.0)

def hash2(j, k, vj, vk, seed):
    # hash (feature_index_j, feature_index_k, value_j, value_k) -> (index, sign)
    h = zlib.crc32(struct.pack("<HHHH", j, k, vj, vk) + struct.pack("<I", seed)) & 0xffffffff
    return (h % D, 1.0 if (h >> 31) == 0 else -1.0)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Precompute hashed feature indices and signs for linear and quadratic terms
    # Linear: for each feature j (0..d-1) and value v (0..16)
    lin_idx = np.zeros((d, 17), dtype=np.int32)
    lin_sign = np.zeros((d, 17), dtype=np.float32)
    for j in range(d):
        for v in range(17):
            idx, sign = hash1(j, v, SEED)
            lin_idx[j, v] = idx
            lin_sign[j, v] = sign

    # Quadratic: for each pair j<k and values vj, vk
    # We'll compute on the fly in batches to save memory, but precompute for speed
    # Since d=64, pairs = 2016. 2016*17*17 ≈ 583k entries - manageable.
    quad_idx = {}
    quad_sign = {}
    for j in range(d):
        for k in range(j+1, d):
            for vj in range(17):
                for vk in range(17):
                    idx, sign = hash2(j, k, vj, vk, SEED)
                    quad_idx[(j, k, vj, vk)] = idx
                    quad_sign[(j, k, vj, vk)] = sign

    # Build design matrix Z (n x D)
    Z = np.zeros((n, D), dtype=np.float32)
    for i in range(n):
        xi = X[i]
        # linear terms
        for j in range(d):
            v = xi[j]
            Z[i, lin_idx[j, v]] += lin_sign[j, v]
        # quadratic terms
        for j in range(d):
            vj = xi[j]
            for k in range(j+1, d):
                vk = xi[k]
                idx = quad_idx[(j, k, vj, vk)]
                Z[i, idx] += quad_sign[(j, k, vj, vk)]

    # Normalize each example to unit L2 norm (helps with hashing scale)
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Z /= norms

    # Train linear model on Z
    W = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(S, yy):
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            g = softmax_grad(z @ W + b, yy)
            grads = [z.T @ g + L2 * W, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Export weights
    out = struct.pack("<4i", 0x504F4C32, d, D, C)
    out += W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-poly2 inference: ./infer weights.bin in.rec N
 * Computes degree-2 hashed features on the fly, then linear classification.
 * Weights format: magic 0x504F4C32, d, D, C; float32 W[D][C], b[C]. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

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

static inline uint32_t hash_crc32(const void *buf, size_t len, uint32_t seed) {
    return crc32(seed, (const Bytef *)buf, len);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x504F4C32) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 64 || D != 2048 ||
        nw != 16 + 4L * ((long)D * C + C)) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *W = (const float *)(hd + 4);
    const float *b = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = calloc(D, sizeof(float));
    float *s = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memset(z, 0, D * sizeof(float));

        // Linear terms
        for (int j = 0; j < d; j++) {
            uint8_t v = xi[j];
            uint32_t h = hash_crc32(&(struct {uint16_t j; uint16_t v;}){j, v}, 4, 1);
            int idx = h % D;
            float sign = (h >> 31) ? -1.0f : 1.0f;
            z[idx] += sign;
        }

        // Quadratic terms
        for (int j = 0; j < d; j++) {
            uint8_t vj = xi[j];
            for (int k = j + 1; k < d; k++) {
                uint8_t vk = xi[k];
                uint32_t h = hash_crc32(&(struct {uint16_t j,k; uint16_t vj,vk;}){j,k,vj,vk}, 8, 1);
                int idx = h % D;
                float sign = (h >> 31) ? -1.0f : 1.0f;
                z[idx] += sign;
            }
        }

        // L2 normalize
        float norm = 0.0f;
        for (int j = 0; j < D; j++) norm += z[j] * z[j];
        if (norm > 0.0f) {
            float inv = 1.0f / sqrtf(norm);
            for (int j = 0; j < D; j++) z[j] *= inv;
        }

        // Linear classification
        for (int c = 0; c < C; c++) s[c] = b[c];
        for (int j = 0; j < D; j++) {
            float zj = z[j];
            if (zj == 0.0f) continue;
            const float *w = W + (long)j * C;
            for (int c = 0; c < C; c++) s[c] += zj * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (s[c] > s[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```