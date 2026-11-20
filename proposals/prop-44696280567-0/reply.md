```hypothesis
Factorization Machine (FM) with field-wise embeddings (k=8). FM models pairwise feature interactions efficiently, which is crucial for Adult where combinations like education×occupation are predictive. It adds minimal overhead over linear models (O(d·k) inference) while capturing non-linearities. If dev accuracy ≤ label-linear (0.8524) or inference cost ≥ label-mlp16 (2325 Ir/item), the approach fails.
```

```python
"""FM family for tabular label tasks: field-aware pairwise interactions via embeddings.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x464D3130, d, k, C=1; float32 bias, lin[d][256], emb[d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 8
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
    d = X.shape[1]
    C = card["classes"]
    assert C == 2, "FM binary only"
    rng = np.random.default_rng(SEED)

    # Initialize parameters
    bias = np.float32(0.0)
    lin = np.zeros((d, 256), dtype=np.float32)
    emb = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_lin, v_lin = np.zeros_like(lin), np.zeros_like(lin)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)

    n = X.shape[0]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d) uint8
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Gather parameters for this batch
            lin_b = lin[xb]                 # (B, d)
            emb_b = emb[xb]                 # (B, d, K)

            # Forward
            sum_emb = emb_b.sum(axis=1)                 # (B, K)
            sum_emb2 = (emb_b * emb_b).sum(axis=1)      # (B, K)
            interaction = 0.5 * ((sum_emb * sum_emb - sum_emb2).sum(axis=1))  # (B,)
            logit = bias + lin_b.sum(axis=1) + interaction
            prob = 1.0 / (1.0 + np.exp(-logit))

            # Loss gradient (BCE)
            dlogit = (prob - yb).astype(np.float32)     # (B,)

            # Gradients
            dbias = dlogit.mean()
            dlin_b = dlogit[:, None]                    # (B, 1) -> broadcast to (B, d)
            # d_interaction/d_emb = sum_emb[:, None, :] - emb_b  (B, d, K)
            demb_b = dlogit[:, None, None] * (sum_emb[:, None, :] - emb_b)

            # Accumulate gradients into parameter gradients (scatter add)
            # We use np.add.at for unsorted indices
            # Linear weights gradient
            grad_lin = np.zeros_like(lin)
            np.add.at(grad_lin, (np.arange(d)[None, :].repeat(B, axis=0).ravel(), xb.ravel()), dlin_b.ravel())
            # Embedding gradient
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], demb_b[:, j, :])

            # L2 regularization
            grad_lin += L2 * lin
            grad_emb += L2 * emb

            # Adam update
            t += 1
            # bias
            m_bias = 0.9 * m_bias + 0.1 * dbias
            v_bias = 0.999 * v_bias + 0.001 * dbias * dbias
            bias -= LR * (m_bias / (1 - 0.9 ** t)) / (np.sqrt(v_bias / (1 - 0.999 ** t)) + 1e-8)
            # lin
            m_lin = 0.9 * m_lin + 0.1 * grad_lin
            v_lin = 0.999 * v_lin + 0.001 * grad_lin * grad_lin
            lin -= LR * (m_lin / (1 - 0.9 ** t)) / (np.sqrt(v_lin / (1 - 0.999 ** t)) + 1e-8)
            # emb
            m_emb = 0.9 * m_emb + 0.1 * grad_emb
            v_emb = 0.999 * v_emb + 0.001 * grad_emb * grad_emb
            emb -= LR * (m_emb / (1 - 0.9 ** t)) / (np.sqrt(v_emb / (1 - 0.999 ** t)) + 1e-8)

    # Write weights
    magic = 0x464D3130  # "FM10"
    out = struct.pack("<4i", magic, d, K, 1)
    out += np.array([bias], dtype="<f4").tobytes()
    out += lin.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* FM inference for label tasks: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: one byte class index (0/1). */
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
    if (C != 1) { fprintf(stderr, "expected C=1\n"); return 3; }
    long expected = 16 + 4L + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *lin = p + 1;
    const float *emb = lin + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = calloc(K, sizeof(float));
    float *sum_v2 = calloc(K, sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float logit = bias;
        memset(sum_v, 0, K * sizeof(float));
        memset(sum_v2, 0, K * sizeof(float));
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            logit += lin[(long)j * 256 + v];
            const float *e = emb + ((long)j * 256 + v) * K;
            for (int u = 0; u < K; u++) {
                float val = e[u];
                sum_v[u] += val;
                sum_v2[u] += val * val;
            }
        }
        float inter = 0.0f;
        for (int u = 0; u < K; u++) inter += sum_v[u] * sum_v[u] - sum_v2[u];
        logit += 0.5f * inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(sum_v);
    free(sum_v2);
    return 0;
}
```