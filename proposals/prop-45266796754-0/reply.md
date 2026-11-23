```hypothesis
Factorization Machine (FM) with per-feature embeddings (K=4) captures pairwise interactions crucial for tabular data (e.g., education×occupation). It should exceed label-linear's 0.8524 dev accuracy while keeping inference cost low (~400 FLOPs/item). If dev accuracy ≤ 0.8524 or latency > label-linear, the hypothesis fails.
```

```python
"""Factorization Machine for tabular label tasks: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy. Exports: magic 0x464D3130, d, K, C; float32 per-class linear[d][256], bias[C], emb[C][d][256][K]."""
import json, math, struct, sys

import numpy as np

K = 4
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
    d = X.shape[1]
    C = card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # observed values per feature (for zeroing unseen later)
    seen = np.zeros((d, 256), dtype=bool)
    for j in range(d):
        seen[j, X[:, j]] = True

    # parameters per class: linear[d,256], bias, emb[d,256,K]
    linear = [np.zeros((d, 256), np.float32) for _ in range(C)]
    bias = [np.zeros((), np.float32) for _ in range(C)]
    emb = [rng.normal(0, 0.01, (d, 256, K)).astype(np.float32) for _ in range(C)]

    # Adam state
    params = []
    for c in range(C):
        params.append(linear[c])
        params.append(bias[c])
        params.append(emb[c])
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # forward
            logits = np.zeros((B, C), np.float32)
            sum_emb = [np.zeros((B, K), np.float32) for _ in range(C)]
            sum_emb_sq = [np.zeros((B, K), np.float32) for _ in range(C)]
            for c in range(C):
                # linear term
                lin = bias[c]
                for j in range(d):
                    lin += linear[c][j, xb[:, j]]
                # embeddings
                for j in range(d):
                    e = emb[c][j, xb[:, j]]          # (B, K)
                    sum_emb[c] += e
                    sum_emb_sq[c] += e * e
                # interaction
                inter = 0.5 * np.sum(sum_emb[c] * sum_emb[c] - sum_emb_sq[c], axis=1)
                logits[:, c] = lin + inter

            # softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = probs / B                    # (B, C)

            # gradients
            grads = []
            for c in range(C):
                gl = grad_logits[:, c]                 # (B,)
                # bias
                g_bias = np.array([gl.sum()], np.float32)
                # linear
                g_linear = np.zeros((d, 256), np.float32)
                for j in range(d):
                    np.add.at(g_linear[j], xb[:, j], gl)
                g_linear += L2 * linear[c]
                # emb
                g_emb = np.zeros((d, 256, K), np.float32)
                # d_interaction/de_j = sum_emb - e_j
                for j in range(d):
                    e = emb[c][j, xb[:, j]]            # (B, K)
                    diff = (sum_emb[c] - e) * gl[:, None]   # (B, K)
                    np.add.at(g_emb[j], xb[:, j], diff)
                g_emb += L2 * emb[c]

                grads.append(g_linear)
                grads.append(g_bias)
                grads.append(g_emb)

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # zero out unseen values (including 255 for categorical)
    for c in range(C):
        for j in range(d):
            unseen = ~seen[j]
            linear[c][j, unseen] = 0.0
            emb[c][j, unseen] = 0.0

    # pack weights: magic, d, K, C; then per class: linear[d*256], bias, emb[d*256*K]
    out = struct.pack("<4i", 0x464D3130, d, K, C)
    for c in range(C):
        out += linear[c].astype("<f4").tobytes()
        out += bias[c].astype("<f4").tobytes()
        out += emb[c].astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index (argmax). */
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1 || K < 1) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16 + 4L * C * ((long)d * 256 + 1 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    const float **linear = malloc(C * sizeof(float*));
    const float *bias = malloc(C * sizeof(float));
    const float **emb = malloc(C * sizeof(float*));
    for (int c = 0; c < C; c++) {
        linear[c] = p; p += d * 256;
        bias[c] = *p++;
        emb[c] = p; p += d * 256 * K;
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(C * K * sizeof(float));
    float *sum_emb_sq = malloc(C * K * sizeof(float));
    float *logits = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            logits[c] = bias[c];
            for (int f = 0; f < K; f++) { sum_emb[c*K+f] = 0.0f; sum_emb_sq[c*K+f] = 0.0f; }
        }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            for (int c = 0; c < C; c++) {
                logits[c] += linear[c][(long)j * 256 + v];
                const float *e = &emb[c][((long)j * 256 + v) * K];
                for (int f = 0; f < K; f++) {
                    float val = e[f];
                    sum_emb[c*K+f] += val;
                    sum_emb_sq[c*K+f] += val * val;
                }
            }
        }

        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int f = 0; f < K; f++) {
                float s = sum_emb[c*K+f];
                inter += 0.5f * (s * s - sum_emb_sq[c*K+f]);
            }
            logits[c] += inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```