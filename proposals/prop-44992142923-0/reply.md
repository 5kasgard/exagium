```hypothesis
Family: label-fm (Factorization Machine on 14 raw byte fields).
Why: Adult has strong pairwise feature interactions (education×occupation, marital×relationship) that linear models miss. FM captures all pairwise interactions via low-rank embeddings (k=4) with only ~72KB weights. It adds ~380 mult-adds/item vs label-linear's ~28, but should gain 0.5-1% accuracy, approaching label-mlp16 at lower cost.
Failure: If dev accuracy ≤ 0.8524 (label-linear) or Ir/item > 2000, the family does not justify its complexity.
```

```python
import json, math, struct, sys
import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 256, 400000, 50


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

    bias = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    e = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_e, v_e = np.zeros_like(e), np.zeros_like(e)

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            B = len(idx)

            e_gather = [e[j, xb[:, j]] for j in range(d)]
            w_gather = [w[j, xb[:, j]] for j in range(d)]

            logits = np.full(B, bias, dtype=np.float32)
            for j in range(d):
                logits += w_gather[j]
            for j in range(d):
                for l in range(j + 1, d):
                    logits += np.sum(e_gather[j] * e_gather[l], axis=1)

            p = 1.0 / (1.0 + np.exp(-logits))
            dlogit = (p - yb).astype(np.float32)

            g_bias = np.mean(dlogit)
            g_w = [np.zeros(256, np.float32) for _ in range(d)]
            g_e = [np.zeros((256, K), np.float32) for _ in range(d)]

            total_e = np.zeros((B, K), np.float32)
            for j in range(d):
                total_e += e_gather[j]

            for j in range(d):
                vals = xb[:, j]
                np.add.at(g_w[j], vals, dlogit)
                sum_other = total_e - e_gather[j]
                for kk in range(K):
                    np.add.at(g_e[j, :, kk], vals, dlogit * sum_other[:, kk])

            g_bias /= B
            for j in range(d):
                g_w[j] /= B
                g_e[j] /= B
                g_w[j] += L2 * w[j]
                g_e[j] += L2 * e[j]

            t += 1
            m_bias = 0.9 * m_bias + 0.1 * g_bias
            v_bias = 0.999 * v_bias + 0.001 * g_bias * g_bias
            bias -= LR * (m_bias / (1 - 0.9**t)) / (np.sqrt(v_bias / (1 - 0.999**t)) + 1e-8)

            for j in range(d):
                mw, vw = m_w[j], v_w[j]
                gw = g_w[j]
                mw[:] = 0.9 * mw + 0.1 * gw
                vw[:] = 0.999 * vw + 0.001 * gw * gw
                w[j] -= LR * (mw / (1 - 0.9**t)) / (np.sqrt(vw / (1 - 0.999**t)) + 1e-8)

                me, ve = m_e[j], v_e[j]
                ge = g_e[j]
                me[:] = 0.9 * me + 0.1 * ge
                ve[:] = 0.999 * ve + 0.001 * ge * ge
                e[j] -= LR * (me / (1 - 0.9**t)) / (np.sqrt(ve / (1 - 0.999**t)) + 1e-8)

    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    out += struct.pack("<f", bias)
    out += w.astype("<f4").tobytes()
    out += e.astype("<f4").tobytes()
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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 20 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K < 1 || K > 16) { fprintf(stderr, "bad weights params\n"); return 3; }

    long expected = 20 + 4L * d * 256 * (1 + K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    float *ptr = (float *)(wb + 16);
    float bias = *ptr++;
    float *w = ptr; ptr += d * 256;
    float *e = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb = malloc(sizeof(float) * d * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logit = bias;
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            logit += w[j * 256 + v];
            float *ej = e + (j * 256 + v) * K;
            float *emb_j = emb + j * K;
            for (int kk = 0; kk < K; kk++) emb_j[kk] = ej[kk];
        }
        for (int j = 0; j < d; j++) {
            float *ej = emb + j * K;
            for (int l = j + 1; l < d; l++) {
                float *el = emb + l * K;
                for (int kk = 0; kk < K; kk++) logit += ej[kk] * el[kk];
            }
        }
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```