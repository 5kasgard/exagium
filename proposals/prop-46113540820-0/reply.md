```hypothesis
Family: label-fm (Factorization Machine with k=4 embeddings per feature position).
Why: Adult has mixed ordinal/categorical features where pairwise interactions (e.g., education × occupation, marital-status × relationship) are known to be predictive. FM captures all pairwise interactions via low-rank embeddings (k=4), adding only ~18K parameters. The per-byte lookup table structure matches label-linear's fast inference (table lookups + vector dot products), avoiding branching. Expected dev accuracy >0.855 (beating label-linear's 0.8524) at ~1500 Ir/item (faster than label-mlp16's 2325). Failure mode: if interactions are weak or k=4 is too small, accuracy may not exceed linear.
```

```python
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 256, 600000, 100


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
    assert C == 2
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    w0 = np.float32(0.0)
    L = np.zeros((d, 256), dtype=np.float32)
    E = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    m_w0, v_w0 = np.float32(0.0), np.float32(0.0)
    m_L, v_L = np.zeros_like(L), np.zeros_like(L)
    m_E, v_E = np.zeros_like(E), np.zeros_like(E)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            B = len(idx)

            linear = np.zeros(B, np.float32)
            emb_sum = np.zeros((B, K), np.float32)
            emb_sq_sum = np.zeros((B, K), np.float32)
            for j in range(d):
                xj = xb[:, j]
                linear += L[j, xj]
                Ej = E[j, xj]
                emb_sum += Ej
                emb_sq_sum += Ej * Ej
            fm = 0.5 * np.sum(emb_sum * emb_sum - emb_sq_sum, axis=1)
            logit = w0 + linear + fm

            max_logit = np.maximum(logit, 0)
            log_exp = np.log(np.exp(-max_logit) + np.exp(logit - max_logit)) + max_logit
            sigmoid = np.exp(logit - log_exp)
            dlogit = (sigmoid - yb).astype(np.float32) / B

            dw0 = np.sum(dlogit)
            dL = np.zeros_like(L)
            dE = np.zeros_like(E)
            for j in range(d):
                xj = xb[:, j]
                np.add.at(dL[j], xj, dlogit)
                Ej = E[j, xj]
                grad_E = dlogit[:, None] * (emb_sum - Ej)
                np.add.at(dE[j], xj, grad_E)

            dL += L2 * L
            dE += L2 * E
            dw0 += L2 * w0

            t += 1
            m_w0 = beta1 * m_w0 + (1 - beta1) * dw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * dw0 * dw0
            w0 -= LR * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)

            m_L = beta1 * m_L + (1 - beta1) * dL
            v_L = beta2 * v_L + (1 - beta2) * dL * dL
            L -= LR * (m_L / (1 - beta1 ** t)) / (np.sqrt(v_L / (1 - beta2 ** t)) + eps)

            m_E = beta1 * m_E + (1 - beta1) * dE
            v_E = beta2 * v_E + (1 - beta2) * dE * dE
            E -= LR * (m_E / (1 - beta1 ** t)) / (np.sqrt(v_E / (1 - beta2 ** t)) + eps)

    out = struct.pack("<4i", 0x21314D46, d, K, C)
    out += np.array([w0], dtype=np.float32).tobytes()
    out += L.astype("<f4").tobytes()
    out += E.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
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
    if (nw < 16 || hd[0] != 0x21314D46) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K < 1 || K > 16 || d != 14) { fprintf(stderr, "bad weights config\n"); return 3; }
    long expected = 16 + 4 + (long)d * 256 * 4 + (long)d * 256 * K * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *L = p + 1;
    const float *E = L + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float emb_sum[16], emb_sq_sum[16];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = w0;
        for (int u = 0; u < K; u++) { emb_sum[u] = 0.0f; emb_sq_sum[u] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            linear += L[(long)j * 256 + v];
            const float *Ej = E + ((long)j * 256 + v) * K;
            for (int u = 0; u < K; u++) {
                float e = Ej[u];
                emb_sum[u] += e;
                emb_sq_sum[u] += e * e;
            }
        }

        float fm = 0.0f;
        for (int u = 0; u < K; u++) {
            float s = emb_sum[u];
            fm += s * s - emb_sq_sum[u];
        }
        fm *= 0.5f;
        float logit = linear + fm;

        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```