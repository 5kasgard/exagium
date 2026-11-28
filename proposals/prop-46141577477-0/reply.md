```hypothesis
Factorization machine (FM) with latent dimension K=4. FMs model pairwise feature interactions via low-rank dot products, which are crucial for tabular data like Adult. This should surpass the linear baseline (0.8524) and approach the MLP16 (0.8591) while being cheaper than MLP16 because interactions cost only 91*K=364 multiply-adds per item. If dev accuracy falls below 0.853, the model does not justify its extra complexity.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
K = 4

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
    C = card["classes"]
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)

    linear_weight = (rng.normal(0, 0.01, (C, d, 256))).astype(np.float32)
    latent = (rng.normal(0, 0.01, (C, d, 256, K))).astype(np.float32)
    bias = np.zeros(C, dtype=np.float32)

    params = [linear_weight, latent, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]
            yb = y[idx]
            bsize = xb.shape[0]

            logits = np.zeros((bsize, C), dtype=np.float32)
            for c in range(C):
                lin = bias[c] + np.sum(linear_weight[c, np.arange(d), xb.T], axis=0)
                lat = latent[c, np.arange(d), xb.T].transpose(1,0,2)
                sum_lat = lat.sum(axis=1)
                sum_sq = (lat**2).sum(axis=1)
                inter = 0.5 * ((sum_lat**2).sum(axis=1) - sum_sq.sum(axis=1))
                logits[:, c] = lin + inter

            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bsize), yb] -= 1.0
            grad_logits = probs / bsize

            grad_linear_weight = np.zeros_like(linear_weight)
            grad_latent = np.zeros_like(latent)
            grad_bias = np.zeros_like(bias)

            for c in range(C):
                g = grad_logits[:, c]
                grad_bias[c] = g.sum()
                for j in range(d):
                    vals = xb[:, j]
                    np.add.at(grad_linear_weight[c, j], vals, g)
                lat = latent[c, np.arange(d), xb.T].transpose(1,0,2)
                sum_lat = lat.sum(axis=1)
                for j in range(d):
                    vals = xb[:, j]
                    grad_lat = g[:, None] * (sum_lat - lat[:, j, :])
                    for u in range(K):
                        np.add.at(grad_latent[c, j, :, u], vals, grad_lat[:, u])

            grad_linear_weight += L2 * linear_weight
            grad_latent += L2 * latent

            t += 1
            grads = [grad_linear_weight, grad_latent, grad_bias]
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    for c in range(C):
        out += bias[c:c+1].astype("<f4").tobytes()
        out += linear_weight[c].astype("<f4").tobytes()
        out += latent[c].astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 14) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + C * (4L + (long)d * 256 * (1 + K) * 4);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *linear_sum = malloc(C * sizeof(float));
    float *latent_sum = malloc(C * K * sizeof(float));
    float *latent_sq_sum = malloc(C * K * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        const float *ptr = p;
        for (int c = 0; c < C; c++) {
            float bias = *ptr++;
            linear_sum[c] = bias;
            for (int u = 0; u < K; u++) {
                latent_sum[c*K + u] = 0.0f;
                latent_sq_sum[c*K + u] = 0.0f;
            }
            const float *lin_w = ptr; ptr += d * 256;
            const float *lat = ptr; ptr += d * 256 * K;

            for (int j = 0; j < d; j++) {
                int v = xi[j];
                linear_sum[c] += lin_w[j * 256 + v];
                const float *lat_vec = lat + (j * 256 + v) * K;
                for (int u = 0; u < K; u++) {
                    float val = lat_vec[u];
                    latent_sum[c*K + u] += val;
                    latent_sq_sum[c*K + u] += val * val;
                }
            }
        }

        float best_logit = -1e30f;
        unsigned char best_class = 0;
        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int u = 0; u < K; u++) {
                float s = latent_sum[c*K + u];
                float sq = latent_sq_sum[c*K + u];
                inter += s * s - sq;
            }
            inter *= 0.5f;
            float logit = linear_sum[c] + inter;
            if (c == 0 || logit > best_logit) {
                best_logit = logit;
                best_class = (unsigned char)c;
            }
        }
        put_rec(stdout, &best_class, 1);
    }
    return 0;
}
```