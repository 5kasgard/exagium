```hypothesis
Factorization Machine (label-fm) with per-class embeddings (k=3) on the 14 quantized fields.
Captures pairwise interactions missing in label-linear; 14 fields × 256 values keeps lookup tables small.
Target: ~0.858 dev accuracy (beats label-linear's 0.8524) at ~1500 Ir/item (between label-linear and label-mlp16).
Failure if accuracy ≤0.853 or Ir/item ≥2500.
```
```python
"""Factorization Machine for label tasks with fixed-size byte items (14 fields, 256 values each).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, Adam, minibatch 64, epochs min(100, ceil(600000/n)).
Weights: magic 0x464D2D4C, d, k, C; float32 linear[C][d][256], emb[C][d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 3
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


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
    d, C = card["input"]["bytes"], card["classes"]
    assert d == 14 and C == 2

    rng = np.random.default_rng(SEED)
    # linear: [C, d, 256], emb: [C, d, 256, K]
    linear = (rng.normal(0, 0.01, (C, d, 256))).astype(np.float32)
    emb = (rng.normal(0, 0.01, (C, d, 256, K))).astype(np.float32)
    params = [linear, emb]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    pairs = [(j1, j2) for j1 in range(d) for j2 in range(j1 + 1, d)]
    n_pairs = len(pairs)

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # forward: logits[b, c]
            logits = np.zeros((bsz, C), np.float32)
            for c in range(C):
                # linear term
                for j in range(d):
                    logits[:, c] += linear[c, j, xb[:, j]]
                # pairwise interactions
                for j1, j2 in pairs:
                    e1 = emb[c, j1, xb[:, j1]]   # [bsz, K]
                    e2 = emb[c, j2, xb[:, j2]]   # [bsz, K]
                    logits[:, c] += (e1 * e2).sum(axis=1)

            # softmax + cross-entropy grad
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yb] -= 1.0
            grad_logits = (probs / bsz).astype(np.float32)  # [bsz, C]

            # gradients
            g_linear = np.zeros_like(linear)
            g_emb = np.zeros_like(emb)

            for c in range(C):
                gl = grad_logits[:, c]  # [bsz]
                # linear grad
                for j in range(d):
                    np.add.at(g_linear[c, j], xb[:, j], gl)
                # emb grad
                for j1, j2 in pairs:
                    v1 = xb[:, j1]
                    v2 = xb[:, j2]
                    e1 = emb[c, j1, v1]  # [bsz, K]
                    e2 = emb[c, j2, v2]  # [bsz, K]
                    # dL/de1 = gl * e2, dL/de2 = gl * e1
                    g1 = gl[:, None] * e2
                    g2 = gl[:, None] * e1
                    np.add.at(g_emb[c, j1], v1, g1)
                    np.add.at(g_emb[c, j2], v2, g2)

            # L2 regularization
            g_linear += L2 * linear
            g_emb += L2 * emb

            grads = [g_linear, g_emb]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights: magic, d, k, C, then linear[C*d*256], emb[C*d*256*K] (row-major)
    out = struct.pack("<4i", 0x464D2D4C, d, K, C)
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine for label tasks: ./infer weights.bin in.rec N
 * Input: d bytes per record (d=14). Output: 1 byte class index (argmax).
 * Weights: magic 0x464D2D4C, int32 d, K, C; float32 linear[C][d][256], emb[C][d][256][K]. */
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
    if (nw < 16 || hd[0] != 0x464D2D4C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 14) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * (C * d * 256 + C * d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *linear = (const float *)(hd + 4);
    const float *emb = linear + C * d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float scores[2];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = 0.0f;

        // linear term
        for (int c = 0; c < C; c++) {
            const float *lin_c = linear + c * d * 256;
            for (int j = 0; j < d; j++) scores[c] += lin_c[j * 256 + xi[j]];
        }

        // pairwise interactions
        for (int c = 0; c < C; c++) {
            const float *emb_c = emb + c * d * 256 * K;
            for (int j1 = 0; j1 < d; j1++) {
                const float *e1 = emb_c + j1 * 256 * K + xi[j1] * K;
                for (int j2 = j1 + 1; j2 < d; j2++) {
                    const float *e2 = emb_c + j2 * 256 * K + xi[j2] * K;
                    float dot = 0.0f;
                    for (int f = 0; f < K; f++) dot += e1[f] * e2[f];
                    scores[c] += dot;
                }
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```