```hypothesis
Family: label-fm (Factorization Machine on the 14 raw byte fields, embedding size k=8).
Why: Adult has strong pairwise interactions (e.g., education × occupation, marital-status × relationship). FM captures all pairwise interactions with O(d·k) parameters and O(d²·k) inference, which is tiny for d=14, k=8. It should beat linear (0.852) and approach GBDT (0.866) at a fraction of the CPU cost.
Failure signal: dev accuracy ≤ 0.852 or Ir/item > 2000.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Weights format (little-endian):
  int32 magic=0x464D4C31 ('FML1'), d, k, C
  float32 w0
  float32 w[d*256]
  float32 V[d*256*k]  (row-major: feature, category, factor)
"""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 64, 600000, 10


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
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.float32)  # 0/1
    d = X.shape[1]
    C = card["classes"]
    assert C == 2
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    w0 = np.float32(0.0)
    w = np.zeros(d * 256, dtype=np.float32)
    V = (rng.normal(0, 0.01, size=(d * 256, K))).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    # Precompute indices for each example: (d,) feature indices
    idx = X.astype(np.int32) + np.arange(d) * 256  # shape (n, d)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            batch_idx = order[s:s + BATCH]
            xi = idx[batch_idx]          # (B, d)
            yi = y[batch_idx]            # (B,)

            # Linear part
            w_x = w[xi].sum(axis=1)      # (B,)
            lin = w0 + w_x

            # FM interaction part: 0.5 * ( (sum v_i)^2 - sum v_i^2 )
            V_xi = V[xi]                 # (B, d, K)
            sum_V = V_xi.sum(axis=1)     # (B, K)
            sum_V2 = (V_xi * V_xi).sum(axis=1)  # (B, K)
            inter = 0.5 * ((sum_V * sum_V).sum(axis=1) - sum_V2.sum(axis=1))

            logits = lin + inter
            # Sigmoid and gradient
            preds = 1.0 / (1.0 + np.exp(-logits))
            grad = (preds - yi).astype(np.float32)  # (B,)

            # Gradients
            # w0
            gw0 = grad.mean()
            # w
            gw = np.zeros_like(w)
            np.add.at(gw, xi.ravel(), np.repeat(grad, d) / BATCH)
            # V
            gV = np.zeros_like(V)
            # For each example: grad * (sum_V - V_xi)  [broadcast over K]
            # sum_V shape (B,K), V_xi shape (B,d,K) -> (B,d,K)
            gV_xi = grad[:, None, None] * (sum_V[:, None, :] - V_xi) / BATCH
            np.add.at(gV, xi.ravel(), gV_xi.reshape(-1, K))

            # L2
            gw += L2 * w
            gV += L2 * V

            t += 1
            # Adam updates
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * gw0 * gw0
            w0 -= LR * (m_w0 / (1 - beta1 ** t)) / (math.sqrt(v_w0 / (1 - beta2 ** t)) + eps)

            m_w = beta1 * m_w + (1 - beta1) * gw
            v_w = beta2 * v_w + (1 - beta2) * gw * gw
            w -= LR * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)

            m_V = beta1 * m_V + (1 - beta1) * gV
            v_V = beta2 * v_V + (1 - beta2) * gV * gV
            V -= LR * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)

    # Write weights
    magic = 0x464D4C31
    out = struct.pack("<4i", magic, d, K, C)
    out += np.array([w0], dtype=np.float32).tobytes()
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d from weights). Output: one byte class index (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K <= 0 || d <= 0) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16L + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *V = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_V = malloc(sizeof(float) * K);
    float *sum_V2 = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float lin = w0;
        memset(sum_V, 0, K * sizeof(float));
        memset(sum_V2, 0, K * sizeof(float));

        for (int j = 0; j < d; j++) {
            int cat = xi[j];
            long base = ((long)j * 256 + cat) * K;
            lin += w[j * 256 + cat];
            for (int f = 0; f < K; f++) {
                float v = V[base + f];
                sum_V[f] += v;
                sum_V2[f] += v * v;
            }
        }

        float inter = 0.0f;
        for (int f = 0; f < K; f++) inter += sum_V[f] * sum_V[f] - sum_V2[f];
        inter *= 0.5f;

        float logit = lin + inter;
        float prob = 1.0f / (1.0f + expf(-logit));
        unsigned char pred = (prob > 0.5f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```