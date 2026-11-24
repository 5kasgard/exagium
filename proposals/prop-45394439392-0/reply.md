```hypothesis
Family: label-fm (factorization machine with per-value embeddings).
Why: Adult has mixed ordinal/categorical features; pairwise interactions capture combinatorial effects (e.g., education × occupation) that linear models miss. FM computes interactions in O(d·K) via embeddings, keeping inference fast.
Failure: If dev accuracy ≤ 0.8524 (label-linear) or CPU cost > 2× label-linear, the model does not justify its complexity.
```

```python
"""Factorization Machine for tabular label tasks: per-byte linear weights and embeddings.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Exports magic 0x464D3132, d, C, K; float32 W[d][256][C], E[d][256][C][K], b[C]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-3, 1e-4, 256, 600000, 50

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
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Parameters
    W = np.zeros((d, 256, C), dtype=np.float32)          # linear weights
    E = (rng.normal(0, 0.01, (d, 256, C, K))).astype(np.float32)  # embeddings
    b = np.zeros(C, dtype=np.float32)

    # Adam state
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_E, v_E = np.zeros_like(E), np.zeros_like(E)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    t = 0
    beta1, beta2 = 0.9, 0.999

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(idx)

            # Forward
            lin = np.tile(b, (B, 1))
            for j in range(d):
                lin += W[j, xb[:, j], :]

            emb_sum = np.zeros((B, C, K), dtype=np.float32)
            emb_sq_sum = np.zeros((B, C, K), dtype=np.float32)
            for j in range(d):
                e = E[j, xb[:, j], :, :]  # (B, C, K)
                emb_sum += e
                emb_sq_sum += e * e

            interaction = 0.5 * np.sum(emb_sum * emb_sum - emb_sq_sum, axis=2)  # (B, C)
            scores = lin + interaction

            # Softmax gradient
            max_s = np.max(scores, axis=1, keepdims=True)
            exp_s = np.exp(scores - max_s)
            probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_scores = probs / B  # (B, C)

            # Gradients
            grad_b = np.sum(grad_scores, axis=0)
            grad_W = np.zeros_like(W)
            grad_E = np.zeros_like(E)

            for j in range(d):
                # Linear gradient
                for c in range(C):
                    np.add.at(grad_W[j, :, c], xb[:, j], grad_scores[:, c])
                # Embedding gradient
                for c in range(C):
                    for k in range(K):
                        g = grad_scores[:, c]
                        es = emb_sum[:, c, k]
                        e_vals = E[j, xb[:, j], c, k]
                        diff = es - e_vals
                        np.add.at(grad_E[j, :, c, k], xb[:, j], diff * g)

            # L2 regularization
            grad_W += L2 * W
            grad_E += L2 * E
            grad_b += L2 * b

            # Adam update
            t += 1
            for param, grad, m, v in ((W, grad_W, m_W, v_W), (E, grad_E, m_E, v_E), (b, grad_b, m_b, v_b)):
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x464D3132
    expected_floats = d * 256 * C * (1 + K) + C
    out = struct.pack("<4i", magic, d, C, K)
    out += W.astype("<f4").tobytes()
    out += E.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    if len(out) != 16 + 4 * expected_floats:
        sys.stderr.write(f"Weight size mismatch: {len(out)} vs {16 + 4 * expected_floats}\n")
        sys.exit(1)
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: one byte class index (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], K = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected_floats = (long)d * 256 * C * (1 + K) + C;
    if (nw != 16 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W = (const float *)(hd + 4);
    const float *E = W + (long)d * 256 * C;
    const float *b = E + (long)d * 256 * C * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * C * K);
    float *emb_sq_sum = malloc(sizeof(float) * C * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int c = 0; c < C; c++) {
            for (int kk = 0; kk < K; kk++) {
                emb_sum[c * K + kk] = 0.0f;
                emb_sq_sum[c * K + kk] = 0.0f;
            }
        }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            long base = ((long)j * 256 + v) * C;
            for (int c = 0; c < C; c++) {
                scores[c] += W[base + c];
                const float *e = E + (base + c) * K;
                for (int kk = 0; kk < K; kk++) {
                    float ek = e[kk];
                    emb_sum[c * K + kk] += ek;
                    emb_sq_sum[c * K + kk] += ek * ek;
                }
            }
        }

        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int kk = 0; kk < K; kk++) {
                float s = emb_sum[c * K + kk];
                inter += 0.5f * (s * s - emb_sq_sum[c * K + kk]);
            }
            scores[c] += inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(emb_sum); free(emb_sq_sum); free(scores);
    return 0;
}
```