```hypothesis
Family: label-fm (Factorization Machine with shared embeddings).
Reason: Adult has strong pairwise interactions (e.g., education×occupation, marital-status×sex). A factorization machine captures all pairwise interactions via low-rank embeddings, improving over linear models while staying faster than GBDT. Shared embeddings + per-class linear terms keep parameters small (~36k) and inference O(14·k).
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or inference cost exceeds label-gbdt, the FM does not justify its complexity here.
```

```python
"""label-fm: Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x464D0001, d, k, C; float32 linear[C][d][256], emb[d][256][k], bias[C]."""
import json, math, struct, sys

import numpy as np

K = 8
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
    rng = np.random.default_rng(SEED)

    # parameters
    linear = np.zeros((C, d, 256), dtype=np.float32)
    emb = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)
    bias = np.zeros(C, dtype=np.float32)

    # Adam state
    m_linear, v_linear = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_bias, v_bias = np.zeros_like(bias), np.zeros_like(bias)

    n = X.shape[0]
    steps = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(steps):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # forward
            # linear part: (C, B)
            lin_part = np.zeros((C, B), dtype=np.float32)
            for c in range(C):
                for j in range(d):
                    lin_part[c] += linear[c, j, xb[:, j]]

            # interaction part: same for both classes
            # compute per-dimension: sum_emb, sum_emb_sq
            sum_emb = np.zeros((B, K), dtype=np.float32)
            sum_emb_sq = np.zeros((B, K), dtype=np.float32)
            for j in range(d):
                v = xb[:, j]                 # (B,)
                sum_emb += emb[j, v, :]      # (B, K)
                sum_emb_sq += emb[j, v, :] ** 2
            inter = 0.5 * np.sum(sum_emb ** 2 - sum_emb_sq, axis=1)  # (B,)

            logits = lin_part + inter + bias[:, None]  # (C, B)
            logits -= logits.max(axis=0, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=0, keepdims=True)  # (C, B)

            # gradient wrt logits
            grad_logits = probs
            grad_logits[yb, np.arange(B)] -= 1.0
            grad_logits /= B  # (C, B)

            # gradients
            # linear
            grad_linear = np.zeros_like(linear)
            for c in range(C):
                for j in range(d):
                    np.add.at(grad_linear[c, j], xb[:, j], grad_logits[c])

            # bias
            grad_bias = grad_logits.sum(axis=1)

            # embeddings (shared)
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                v = xb[:, j]                 # (B,)
                # for each embedding dimension k:
                # dL/demb[j, v, k] = sum_c grad_logits[c, b] * (sum_emb[b, k] - emb[j, v, k])
                diff = sum_emb - emb[j, v, :]  # (B, K)
                for c in range(C):
                    g = grad_logits[c][:, None] * diff  # (B, K)
                    np.add.at(grad_emb[j, v], np.arange(K), g.T)  # add over batch

            # Adam update with L2
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

            # linear
            grad_linear += L2 * linear
            m_linear = beta1 * m_linear + (1 - beta1) * grad_linear
            v_linear = beta2 * v_linear + (1 - beta2) * grad_linear * grad_linear
            m_hat = m_linear / (1 - beta1 ** t)
            v_hat = v_linear / (1 - beta2 ** t)
            linear -= LR * m_hat / (np.sqrt(v_hat) + eps)

            # emb
            grad_emb += L2 * emb
            m_emb = beta1 * m_emb + (1 - beta1) * grad_emb
            v_emb = beta2 * v_emb + (1 - beta2) * grad_emb * grad_emb
            m_hat = m_emb / (1 - beta1 ** t)
            v_hat = v_emb / (1 - beta2 ** t)
            emb -= LR * m_hat / (np.sqrt(v_hat) + eps)

            # bias
            grad_bias += L2 * bias
            m_bias = beta1 * m_bias + (1 - beta1) * grad_bias
            v_bias = beta2 * v_bias + (1 - beta2) * grad_bias * grad_bias
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # write weights
    # magic, d, k, C
    out = struct.pack("<4i", 0x464D0001, d, K, C)
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class (argmax of two logits).
 * Weights layout: magic(0x464D0001), d, k, C; float32 linear[C][d][256], emb[d][256][k], bias[C]. */
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
    if (nw < 16 || hd[0] != 0x464D0001) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d < 1 || K < 1 || d > 256 || K > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }

    size_t expected = 16
        + (size_t)C * d * 256 * sizeof(float)
        + (size_t)d * 256 * K * sizeof(float)
        + (size_t)C * sizeof(float);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *linear = (const float *)(hd + 4);
    const float *emb = linear + (size_t)C * d * 256;
    const float *bias = emb + (size_t)d * 256 * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logits[2] = {bias[0], bias[1]};
        float sum_emb[256];       // K <= 256, but we only need up to K
        float sum_emb_sq[256];
        for (int f = 0; f < K; f++) { sum_emb[f] = 0.0f; sum_emb_sq[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            // linear part
            for (int c = 0; c < C; c++)
                logits[c] += linear[(c * d + j) * 256 + v];
            // accumulate embeddings
            const float *e = emb + ((size_t)j * 256 + v) * K;
            for (int f = 0; f < K; f++) {
                float val = e[f];
                sum_emb[f] += val;
                sum_emb_sq[f] += val * val;
            }
        }

        // interaction term (same for both classes)
        float inter = 0.0f;
        for (int f = 0; f < K; f++) {
            float s = sum_emb[f];
            inter += 0.5f * (s * s - sum_emb_sq[f]);
        }
        logits[0] += inter;
        logits[1] += inter;

        unsigned char best = (logits[1] > logits[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```