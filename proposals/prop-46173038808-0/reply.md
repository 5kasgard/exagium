```hypothesis
Family: label-fm (Factorization Machine with per-field embeddings).
Why: Adult income has strong pairwise interactions (e.g., education×occupation, marital×sex) that linear models miss.
FM captures all pairwise interactions via low-rank embeddings (k=8) while keeping inference at O(d·k) ~ 100 FLOPs/item.
If dev accuracy ≤ 0.8524 (label-linear) or latency ≥ label-linear, the hypothesis fails.
```

```python
"""label-fm: Factorization Machine for tabular byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Exports: magic(0x464D4C31), d, k, C, w0, w[d][256], v[d][256][k] as float32 LE."""
import json, math, struct, sys

import numpy as np

SEED = 1
K = 8                 # embedding dimension
LR = 0.01
L2 = 1e-4
BATCH = 256
EPOCHS = 50
SAMPLES = 600000

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
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)   # 0 / 1
    d = X.shape[1]
    n = X.shape[0]
    C = card["classes"]
    assert C == 2

    rng = np.random.default_rng(SEED)

    # parameters
    w0 = np.float32(0.0)
    w  = np.zeros((d, 256), dtype=np.float32)          # linear weights per field per value
    v  = (rng.normal(0, 0.01, (d, 256, K)).astype(np.float32))  # embeddings

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_v, v_v = np.zeros_like(v), np.zeros_like(v)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    max_epochs = min(EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]                 # (b, d) uint8
            yb = y[idx]                 # (b,) int64
            b = xb.shape[0]

            # ---- forward ----
            # linear term
            lin = w0 + np.sum(w[np.arange(d), xb.T], axis=0)   # (b,)
            # interaction term: 0.5 * sum_k ( (sum_f v_f)^2 - sum_f v_f^2 )
            emb = v[np.arange(d), xb.T]      # (d, b, k)
            sum_emb = emb.sum(axis=0)        # (b, k)
            sum_emb_sq = (emb * emb).sum(axis=0)
            inter = 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=1)   # (b,)
            logits = lin + inter             # (b,)

            # binary cross-entropy gradient w.r.t logits
            p = 1.0 / (1.0 + np.exp(-logits))
            grad_logits = (p - yb).astype(np.float32) / b    # (b,)

            # ---- backward ----
            # w0
            g_w0 = grad_logits.sum()
            # w
            g_w = np.zeros_like(w)
            np.add.at(g_w, (np.arange(d), xb.T), grad_logits)   # (d, 256)
            g_w += L2 * w
            # v
            # d(inter)/d v_f = (sum_emb - v_f) * grad_logits[:,None]
            g_v = np.zeros_like(v)
            for f in range(d):
                vf = emb[f]                     # (b, k)
                g_vf = (sum_emb - vf) * grad_logits[:, None]    # (b, k)
                np.add.at(g_v[f], xb[:, f], g_vf)
            g_v += L2 * v

            # ---- Adam update ----
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            w0 -= LR * m_w0_hat / (np.sqrt(v_w0_hat) + eps)
            # w
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            w -= LR * m_w_hat / (np.sqrt(v_w_hat) + eps)
            # v
            m_v = beta1 * m_v + (1 - beta1) * g_v
            v_v = beta2 * v_v + (1 - beta2) * g_v * g_v
            m_v_hat = m_v / (1 - beta1 ** t)
            v_v_hat = v_v / (1 - beta2 ** t)
            v -= LR * m_v_hat / (np.sqrt(v_v_hat) + eps)

    # ---- write weights.bin ----
    magic = 0x464D4C31  # 'FML1'
    header = struct.pack("<4i", magic, d, K, C)
    body = struct.pack("<f", w0) + w.astype("<f4").tobytes() + v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte class (0 or 1).
 * Weights format: int32 magic(0x464D4C31), d, k, C; float32 w0, w[d][256], v[d][256][k]. */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d <= 0 || k <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L + 4L * d * 256 + 4L * d * 256 * k;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *ptr = (const float *)(hd + 4);
    float w0 = *ptr++;
    const float *w = ptr; ptr += d * 256;
    const float *v = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(k * sizeof(float));
    float *sum_emb_sq = malloc(k * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        float lin = w0;
        for (int f = 0; f < k; f++) { sum_emb[f] = 0.0f; sum_emb_sq[f] = 0.0f; }

        for (int f = 0; f < d; f++) {
            unsigned char val = xi[f];
            lin += w[f * 256 + val];
            const float *vf = v + (f * 256 + val) * k;
            for (int fk = 0; fk < k; fk++) {
                float ev = vf[fk];
                sum_emb[fk] += ev;
                sum_emb_sq[fk] += ev * ev;
            }
        }
        float inter = 0.0f;
        for (int fk = 0; fk < k; fk++) {
            float s = sum_emb[fk];
            inter += s * s - sum_emb_sq[fk];
        }
        inter *= 0.5f;
        float logit = lin + inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(sum_emb); free(sum_emb_sq);
    return 0;
}
```