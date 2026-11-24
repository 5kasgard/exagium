```hypothesis
Factorization Machine (FM) captures pairwise feature interactions crucial for tabular data like Adult
(education×occupation, marital×sex, etc.) with only O(k·d) extra params (k=8, d=14). Inference costs
~14·k=112 FLOPs/item, far below label-linear's 719 Ir/item. Deterministic Adam training on all 26k
samples. If dev accuracy ≤ 0.8524 (label-linear), the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split conceptually, hardcoded for determinism)
K = 8
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 256
EPOCHS = 20
MAGIC = 0x464D3131  # 'FM11'

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
    d = card["input"]["bytes"]          # 14
    C = card["classes"]                 # 2
    n_feat = d * 256                    # 3584
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Parameters
    w0 = 0.0
    w = np.zeros(n_feat, np.float32)
    V = (rng.normal(0, 0.01, (n_feat, K))).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)

    t = 0
    # Precompute feature index offsets per column
    col_off = np.arange(d, dtype=np.int32) * 256  # (14,)

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]                 # (B, 14) uint8
            yb = y[idx]                 # (B,) int64
            B = len(idx)

            # Active feature indices: (B, 14)
            feat_idx = xb.astype(np.int32) + col_off

            # Linear term
            lin = w0 + np.sum(w[feat_idx], axis=1)  # (B,)

            # Interaction term
            V_act = V[feat_idx]                     # (B, 14, K)
            sum_v = np.sum(V_act, axis=1)           # (B, K)
            sum_v2 = np.sum(V_act * V_act, axis=1)  # (B, K)
            inter = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)  # (B,)

            score = lin + inter
            p = 1.0 / (1.0 + np.exp(-score))        # sigmoid
            g = (p - yb.astype(np.float32)) / B     # (B,) average gradient

            # Gradients
            dw0 = np.sum(g) + L2 * w0

            dw = np.zeros_like(w)
            np.add.at(dw, feat_idx.ravel(), np.repeat(g, d))
            dw += L2 * w

            # dV: g * (sum_v - V_act) per active feature
            sum_v_exp = sum_v[:, np.newaxis, :]     # (B, 1, K)
            dV_act = g[:, np.newaxis, np.newaxis] * (sum_v_exp - V_act)  # (B, 14, K)
            dV = np.zeros_like(V)
            np.add.at(dV, feat_idx.ravel(), dV_act.reshape(-1, K))
            dV += L2 * V

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

            m_w0 = beta1 * m_w0 + (1 - beta1) * dw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * dw0 * dw0
            w0 -= LR * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)

            m_w = beta1 * m_w + (1 - beta1) * dw
            v_w = beta2 * v_w + (1 - beta2) * dw * dw
            w -= LR * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)

            m_V = beta1 * m_V + (1 - beta1) * dV
            v_V = beta2 * v_V + (1 - beta2) * dV * dV
            V -= LR * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)

    # Write weights.bin
    # Header: magic, d, K, C
    # Then: w0 (float32), w[n_feat] (float32), V[n_feat*K] (float32)
    hdr = struct.pack("<4i", MAGIC, d, K, C)
    body = struct.pack("<f", w0) + w.astype("<f4").tobytes() + V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(hdr + body)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks with fixed-size byte items.
 * ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte class (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D3131) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 14 || C != 2 || K != 8) { fprintf(stderr, "bad weights dims\n"); return 3; }
    int n_feat = d * 256;                /* 3584 */
    long expected = 16 + 4L + 4L * n_feat + 4L * (long)n_feat * K;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    const float *ptr = (const float *)(hd + 4);
    float w0 = ptr[0];
    const float *w = ptr + 1;
    const float *V = ptr + 1 + n_feat;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_v[8], sum_v2[8];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = w0;
        for (int j = 0; j < d; j++) {
            int idx = j * 256 + xi[j];
            score += w[idx];
        }
        for (int f = 0; f < K; f++) { sum_v[f] = 0.0f; sum_v2[f] = 0.0f; }
        for (int j = 0; j < d; j++) {
            int idx = j * 256 + xi[j];
            const float *v = V + (long)idx * K;
            for (int f = 0; f < K; f++) {
                float vf = v[f];
                sum_v[f] += vf;
                sum_v2[f] += vf * vf;
            }
        }
        for (int f = 0; f < K; f++) {
            score += 0.5f * (sum_v[f] * sum_v[f] - sum_v2[f]);
        }
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```