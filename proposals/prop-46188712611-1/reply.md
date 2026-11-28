```hypothesis
Family: label-patch. Learns 16 non-overlapping 2×2 patch embeddings (4 pixels each, binned to 4 levels → 256 codes/patch) with a linear output layer. Captures local spatial structure critical for 8×8 digits while using ~4× fewer inference adds than label-linear (160 vs 640). Expects dev accuracy > 0.9613 at lower Ir/item. Fails if dev accuracy ≤ 0.9613 or instructions/item ≥ label-linear.
```
```python
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 100
NUM_PATCHES = 16
PATCH_PIXELS = 4
BINS = 4
CODES = BINS ** PATCH_PIXELS  # 256


def bin_pixel(v: int) -> int:
    if v == 0:
        return 0
    elif v <= 5:
        return 1
    elif v <= 11:
        return 2
    else:
        return 3


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
    n, d = X_bytes.shape
    assert d == 64

    patch_indices = []
    for p in range(NUM_PATCHES):
        pi, pj = p // 4, p % 4
        idx = [
            (2 * pi) * 8 + 2 * pj,
            (2 * pi) * 8 + 2 * pj + 1,
            (2 * pi + 1) * 8 + 2 * pj,
            (2 * pi + 1) * 8 + 2 * pj + 1,
        ]
        patch_indices.append(idx)

    X_binned = np.vectorize(bin_pixel)(X_bytes).astype(np.uint8)
    X_patch = np.zeros((n, NUM_PATCHES), dtype=np.uint8)
    for p, idx in enumerate(patch_indices):
        code = (X_binned[:, idx[0]] * 64 +
                X_binned[:, idx[1]] * 16 +
                X_binned[:, idx[2]] * 4 +
                X_binned[:, idx[3]])
        X_patch[:, p] = code

    rng = np.random.default_rng(SEED)
    E = (rng.normal(0, 0.01, (NUM_PATCHES, CODES, C))).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    params = [E, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    num_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(num_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xp, yy = X_patch[idx], y[idx]
            acc = np.broadcast_to(b, (len(yy), C)).copy()
            for p in range(NUM_PATCHES):
                acc += E[p, xp[:, p], :]
            acc_max = acc.max(axis=1, keepdims=True)
            exp_acc = np.exp(acc - acc_max)
            probs = exp_acc / exp_acc.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad_acc = (probs / len(yy)).astype(np.float32)

            grad_E = np.zeros_like(E)
            for p in range(NUM_PATCHES):
                np.add.at(grad_E[p], xp[:, p], grad_acc)
            grad_E += L2 * E
            grad_b = grad_acc.sum(axis=0) + L2 * b
            grads = [grad_E, grad_b]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    magic = 0x50415443  # 'PATC'
    header = struct.pack("<6i", magic, d, NUM_PATCHES, PATCH_PIXELS, BINS, C)
    out = header + E.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
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

static inline int bin_pixel(unsigned char v) {
    if (v == 0) return 0;
    else if (v <= 5) return 1;
    else if (v <= 11) return 2;
    else return 3;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 24 || hd[0] != 0x50415443) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], num_patches = hd[2], patch_pixels = hd[3], bins = hd[4], C = hd[5];
    if (d != 64 || num_patches != 16 || patch_pixels != 4 || bins != 4 || C < 1 || C > 256) {
        fprintf(stderr, "bad weights header\n"); return 3;
    }
    long expected = 24L + 4L * num_patches * 256 * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *E = (const float *)(hd + 6);
    const float *b = E + (long)num_patches * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * C);
    float *sc = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) acc[c] = b[c];

        for (int p = 0; p < num_patches; p++) {
            int pi = p / 4, pj = p % 4;
            int idx[4] = {
                (2 * pi) * 8 + 2 * pj,
                (2 * pi) * 8 + 2 * pj + 1,
                (2 * pi + 1) * 8 + 2 * pj,
                (2 * pi + 1) * 8 + 2 * pj + 1
            };
            int code = (bin_pixel(xi[idx[0]]) * 64 +
                        bin_pixel(xi[idx[1]]) * 16 +
                        bin_pixel(xi[idx[2]]) * 4 +
                        bin_pixel(xi[idx[3]]));
            const float *ep = E + ((long)p * 256 + code) * C;
            for (int c = 0; c < C; c++) acc[c] += ep[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```