```hypothesis
Family: label-fm (factorization machine). It adds pairwise feature interactions via low-rank embeddings (K=8) to a linear model, capturing non-linearities in tabular data while staying efficient. Trained with Adam on logistic loss, deterministic. Expected to beat label-linear (0.8524) on dev accuracy (target >0.855) with CPU cost ~2000-3000 Ir/item. If accuracy ≤ label-linear or cost exceeds label-mlp16, the approach fails.
```

```python
import json, math, struct, sys
import numpy as np

MAGIC = 0x464D4C31
SEED = 1
K = 8
LR = 0.01
L2 = 1e-4
BATCH = 256
EPOCHS = 20

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
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
    bias = np.float32(0.0)
    linear = rng.uniform(-0.01, 0.01, (d, 256)).astype(np.float32)
    emb = rng.uniform(-0.01, 0.01, (d, 256, K)).astype(np.float32)
    m_bias, v_bias = 0.0, 0.0
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    t = 0
    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]
            yb = y[idx]
            B = len(yb)
            lin_part = bias + np.sum(linear[np.arange(d), xb.T], axis=0)
            emb_vals = emb[np.arange(d), xb.T].transpose(1,0,2)
            sum_emb = np.sum(emb_vals, axis=1)
            sum_emb_sq = np.sum(emb_vals**2, axis=1)
            inter_part = 0.5 * np.sum(sum_emb**2 - sum_emb_sq, axis=1)
            logits = lin_part + inter_part
            p = 1.0 / (1.0 + np.exp(-logits))
            grad = (p - yb) / B
            g_bias = np.sum(grad)
            g_lin = np.zeros_like(linear)
            for j in range(d):
                np.add.at(g_lin[j], xb[:, j], grad)
            g_emb = np.zeros_like(emb)
            for j in range(d):
                vals = xb[:, j]
                for kk in range(K):
                    diff = sum_emb[:, kk] - emb[j, vals, kk]
                    np.add.at(g_emb[j, :, kk], vals, grad * diff)
            g_bias += L2 * bias
            g_lin += L2 * linear
            g_emb += L2 * emb
            t += 1
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            bias -= LR * (m_bias / (1 - beta1**t)) / (np.sqrt(v_bias / (1 - beta2**t)) + eps)
            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            linear -= LR * (m_lin / (1 - beta1**t)) / (np.sqrt(v_lin / (1 - beta2**t)) + eps)
            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            emb -= LR * (m_emb / (1 - beta1**t)) / (np.sqrt(v_emb / (1 - beta2**t)) + eps)
    out = struct.pack("<4i", MAGIC, d, K, C)
    out += np.array([bias], dtype=np.float32).tobytes()
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K < 1 || K > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    float *bias = (float *)(wb + 16);
    float *linear = bias + 1;
    float *emb = linear + d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(K * sizeof(float));
    float *sum_emb_sq = malloc(K * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float logit = *bias;
        for (int kk = 0; kk < K; kk++) {
            sum_emb[kk] = 0.0f;
            sum_emb_sq[kk] = 0.0f;
        }
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            logit += linear[j * 256 + v];
            float *e = emb + (j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = e[kk];
                sum_emb[kk] += val;
                sum_emb_sq[kk] += val * val;
            }
        }
        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            float s = sum_emb[kk];
            inter += s * s - sum_emb_sq[kk];
        }
        logit += 0.5f * inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```