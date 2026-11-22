```hypothesis
Family: label-elm (Extreme Learning Machine). Fixed random projection to 128 ReLU units, then ridge-regression output layer. No back-prop, so training is a single deterministic linear solve. On optdigits the 64 ordinal pixels benefit from a non-linear expansion; 128 units gives enough capacity to beat label-linear (0.9613) while inference stays at one dense layer (64*128 + 128*10 mul-adds), faster than label-mlp16. Failure: dev accuracy ≤ 0.9613 or inference cost ≥ label-mlp16.
```
```python
"""label-elm: Extreme Learning Machine for fixed-size byte label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Architecture: input -> fixed random projection (ReLU) -> linear output.
Hidden size H=128, ridge lambda=1e-4. Weights file stores seed, H, C, W_out, b_out.
Random projection weights are regenerated from seed in infer.c."""
import json, math, struct, sys

import numpy as np

SEED = 1
H = 128
LAMBDA = 1e-4

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
    d = X.shape[1]

    # Encode ordinal features as z-scored floats (same as label-linear)
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals))); E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, k = Z.shape

    # Fixed random projection
    rng = np.random.default_rng(SEED)
    W_hidden = (rng.standard_normal((k, H)) * math.sqrt(2.0 / k)).astype(np.float32)  # He init
    b_hidden = np.zeros(H, np.float32)

    # Hidden activations
    H_act = np.maximum(Z @ W_hidden + b_hidden, 0, dtype=np.float32)  # ReLU

    # One-hot targets
    Y = np.zeros((n, C), np.float32)
    Y[np.arange(n), y] = 1.0

    # Ridge regression: (H^T H + lambda I) W_out = H^T Y
    A = H_act.T @ H_act + LAMBDA * np.eye(H, dtype=np.float32)
    B = H_act.T @ Y
    W_out = np.linalg.solve(A, B).astype(np.float32)  # H x C
    b_out = np.zeros(C, np.float32)  # bias absorbed in hidden layer; optional but kept for symmetry

    # Save weights.bin: magic, d, H, C, seed, lambda, W_out, b_out
    magic = 0x454C4D31  # "ELM1"
    out = struct.pack("<5If", magic, d, H, C, SEED, LAMBDA)
    out += W_out.astype("<f4").tobytes()
    out += b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-elm inference: ./infer weights.bin in.rec N
 * Input record: d bytes (64). Output record: 1 byte (class 0..9).
 * Weights format: magic(4), d(4), H(4), C(4), seed(8), lambda(4), W_out[H*C](float32), b_out[C](float32).
 * Random projection weights generated on the fly from seed using splitmix64 -> Box-Muller. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static float randn(uint64_t *state) {
    // Box-Muller transform, two independent normals per two uint64
    static int has_spare = 0;
    static float spare;
    if (has_spare) {
        has_spare = 0;
        return spare;
    }
    uint64_t u64 = splitmix64(state);
    uint64_t v64 = splitmix64(state);
    double u = (u64 >> 11) * (1.0 / 9007199254740992.0); // 53 bits
    double v = (v64 >> 11) * (1.0 / 9007199254740992.0);
    double r = sqrt(-2.0 * log(u + 1e-12));
    double theta = 2.0 * M_PI * v;
    spare = (float)(r * sin(theta));
    has_spare = 1;
    return (float)(r * cos(theta));
}

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
    if (nw < 24) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x454C4D31) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    uint64_t seed = *(uint64_t *)(wb + 16);
    float lambda = *(float *)(wb + 24);
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }
    size_t expected = 28 + (size_t)H * C * 4 + (size_t)C * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }
    const float *W_out = (const float *)(wb + 28);
    const float *b_out = W_out + (size_t)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    // Generate fixed random projection weights
    uint64_t rng_state = seed;
    float *W_hidden = malloc((size_t)d * H * sizeof(float));
    float *b_hidden = malloc((size_t)H * sizeof(float));
    float scale = sqrtf(2.0f / d);
    for (int i = 0; i < d * H; i++) W_hidden[i] = randn(&rng_state) * scale;
    for (int i = 0; i < H; i++) b_hidden[i] = 0.0f;

    float *hidden = malloc((size_t)H * sizeof(float));
    float *scores = malloc((size_t)C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Hidden layer: ReLU(x * W_hidden + b_hidden)
        for (int h = 0; h < H; h++) {
            float sum = b_hidden[h];
            for (int j = 0; j < d; j++) sum += xi[j] * W_hidden[(size_t)j * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b_out[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W_out[(size_t)h * C + c];
            scores[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(W_hidden); free(b_hidden); free(hidden); free(scores);
    if (x) free(x);
    free(wb);
    return 0;
}
```