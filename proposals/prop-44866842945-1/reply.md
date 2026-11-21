```hypothesis
Propose "label-fm": a Factorization Machine adding pairwise interactions via low-rank embeddings to a linear model.
The 8x8 digit pixels have strong local correlations (strokes, loops) that a linear model misses; a degree-2 FM with k=8 captures these with only ~1.5M parameters.
Inference stays fast: O(C·d·k) ≈ 5k ops/item. If dev accuracy ≤ 0.9613 (label-linear) or not competitive with label-mlp16 (0.9770), the family fails.
```
```python
"""label-fm: Factorization Machine for fixed-size byte items (linear + pairwise interactions via k-dim embeddings).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Exports: int32 magic 0x464D5032, d, k, C; float32 T[d][256][C], V[C][d][256][k], b[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 0.01, 1e-4, 64, 600000, 100
K = 8                     # FM embedding dimension
MAGIC = 0x464D5032        # 'FMP2'

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

    # Parameters: linear T[d,256,C], FM embeddings V[C,d,256,K], bias b[C]
    T = np.zeros((d, 256, C), dtype=np.float32)
    V = (rng.normal(0, 0.01, (C, d, 256, K))).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    params = [T, V, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            B = len(yb)

            # ---- forward ----
            lin = np.zeros((B, C), dtype=np.float32)
            for j in range(d):
                lin += T[j, xb[:, j], :]
            lin += b

            fm = np.zeros((B, C), dtype=np.float32)
            for c in range(C):
                a = np.zeros((B, K), dtype=np.float32)
                for j in range(d):
                    a += V[c, j, xb[:, j], :]
                sum_a2 = np.sum(a * a, axis=1)
                sum_v2 = 0.0
                for j in range(d):
                    v = V[c, j, xb[:, j], :]
                    sum_v2 += np.sum(v * v, axis=1)
                fm[:, c] = 0.5 * (sum_a2 - sum_v2)

            logits = lin + fm
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            # ---- loss gradient ----
            grad_logits = probs
            grad_logits[np.arange(B), yb] -= 1.0
            grad_logits /= B

            # ---- backward ----
            grad_b = grad_logits.sum(axis=0)

            grad_T = np.zeros_like(T)
            for j in range(d):
                for c in range(C):
                    np.add.at(grad_T[j, :, c], xb[:, j], grad_logits[:, c])

            grad_V = np.zeros_like(V)
            for c in range(C):
                a = np.zeros((B, K), dtype=np.float32)
                for j in range(d):
                    a += V[c, j, xb[:, j], :]
                for j in range(d):
                    v = V[c, j, xb[:, j], :]
                    g = (a - v) * grad_logits[:, c:c+1]   # (B, K)
                    for f in range(K):
                        np.add.at(grad_V[c, j, :, f], xb[:, j], g[:, f])

            # L2 regularization
            grad_T += L2 * T
            grad_V += L2 * V
            grad_b += L2 * b

            grads = [grad_T, grad_V, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # ---- export ----
    out = struct.pack("<4i", MAGIC, d, K, C)
    out += T.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels 0..16). Output: one byte class index.
 * Weights format: magic 0x464D5032, int32 d, K, C; float32 T[d][256][C], V[C][d][256][K], b[C]. */
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

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D5032) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 32 || d < 1) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * C + (long)C * d * 256 * K + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *V = T + (long)d * 256 * C;
    const float *b = V + (long)C * d * 256 * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_f = malloc(sizeof(float) * K);
    unsigned char best;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float best_score = -1e30f; best = 0;
        for (int c = 0; c < C; c++) {
            float score = b[c];
            for (int j = 0; j < d; j++) {
                score += T[((long)j * 256 + xi[j]) * C + c];
            }
            memset(sum_f, 0, sizeof(float) * K);
            float sum_v2 = 0.0f;
            for (int j = 0; j < d; j++) {
                const float *v = V + (((long)c * d + j) * 256 + xi[j]) * K;
                for (int f = 0; f < K; f++) {
                    sum_f[f] += v[f];
                    sum_v2 += v[f] * v[f];
                }
            }
            float fm = 0.0f;
            for (int f = 0; f < K; f++) fm += sum_f[f] * sum_f[f];
            fm = 0.5f * (fm - sum_v2);
            score += fm;
            if (score > best_score) { best_score = score; best = (unsigned char)c; }
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```