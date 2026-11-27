```hypothesis
Factorization Machine (degree-2) with k=4 latent factors per class. Captures pairwise feature interactions crucial for Adult (e.g., education×occupation, marital-status×relationship) while maintaining near-linear inference speed via the FM trick (O(d·k) not O(d²)). If dev accuracy ≤ label-linear (0.8524) or Ir/item > 2× label-linear, the family fails.
```
```python
"""Factorization Machine (degree-2) for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Weights: magic 0x464D3231, d, k, C; float32 L[d][256][C], b[C], V[d][256][C][k]."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 100


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
    d, C = X.shape[1], card["classes"]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # params: L[d,256,C], V[d,256,C,K], b[C]
    L = (rng.normal(0, 0.01, (d, 256, C))).astype(np.float32)
    V = (rng.normal(0, 0.01, (d, 256, C, K))).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [L, V, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(idx)

            # Forward
            # Linear: sum_j L[j, xb[:,j], :] -> (B, C)
            lin = np.zeros((B, C), np.float32)
            for j in range(d):
                lin += L[j, xb[:, j], :]

            # FM interaction per class
            inter = np.zeros((B, C), np.float32)
            for c in range(C):
                emb_sum = np.zeros((B, K), np.float32)
                emb_sq_sum = np.zeros((B, K), np.float32)
                for j in range(d):
                    v = V[j, xb[:, j], c, :]  # (B, K)
                    emb_sum += v
                    emb_sq_sum += v * v
                inter[:, c] = 0.5 * np.sum(emb_sum * emb_sum - emb_sq_sum, axis=1)

            scores = b + lin + inter  # (B, C)

            # Softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_scores = (probs / B).astype(np.float32)  # (B, C)

            # Gradients
            grad_L = np.zeros_like(L)
            grad_V = np.zeros_like(V)
            grad_b = grad_scores.sum(axis=0)

            # Linear gradient: scatter-add
            for j in range(d):
                np.add.at(grad_L[j], xb[:, j], grad_scores)

            # FM gradient
            for c in range(C):
                # emb_sum per sample for class c: (B, K)
                emb_sum_c = np.zeros((B, K), np.float32)
                for j in range(d):
                    emb_sum_c += V[j, xb[:, j], c, :]
                for j in range(d):
                    v = V[j, xb[:, j], c, :]  # (B, K)
                    # dInter/dV = grad_scores[:,c:c+1] * (emb_sum_c - v)
                    gv = grad_scores[:, c:c+1] * (emb_sum_c - v)  # (B, K)
                    np.add.at(grad_V[j, :, c, :], xb[:, j], gv)

            grads = [grad_L, grad_V, grad_b]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * (m_hat / (np.sqrt(v_hat) + 1e-8) + L2 * p)

    # Pack weights: magic, d, k, C; L[d*256*C]; b[C]; V[d*256*C*K]
    out = struct.pack("<4i", 0x464D3231, d, K, C)
    out += L.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine (degree-2) inference: ./infer weights.bin in.rec N
 * Input record: d bytes (feature values 0-255). Output: 1 byte class index. */
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
    if (nw < 16 || hd[0] != 0x464D3231) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 64 || d < 1 || d > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * C + C + (long)d * 256 * C * K);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *L = (const float *)(hd + 4);
    const float *b = L + (long)d * 256 * C;
    const float *V = b + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *emb_sum = malloc(sizeof(float) * K);
    float *emb_sq_sum = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = b[c];

        // Linear part
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            const float *lj = L + ((long)j * 256 + v) * C;
            for (int c = 0; c < C; c++) scores[c] += lj[c];
        }

        // FM interaction per class
        for (int c = 0; c < C; c++) {
            for (int f = 0; f < K; f++) { emb_sum[f] = 0.0f; emb_sq_sum[f] = 0.0f; }
            for (int j = 0; j < d; j++) {
                int v = xi[j];
                const float *vj = V + (((long)j * 256 + v) * C + c) * K;
                for (int f = 0; f < K; f++) {
                    float val = vj[f];
                    emb_sum[f] += val;
                    emb_sq_sum[f] += val * val;
                }
            }
            float inter = 0.0f;
            for (int f = 0; f < K; f++) inter += 0.5f * (emb_sum[f] * emb_sum[f] - emb_sq_sum[f]);
            scores[c] += inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```