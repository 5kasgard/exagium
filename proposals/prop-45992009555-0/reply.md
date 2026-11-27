```hypothesis
Family: label-fm (Factorization Machine with shared latent vectors).
Why: Adult has strong feature interactions (e.g., education×occupation, marital-status×relationship) that linear models miss. FM captures all pairwise interactions via k=8 latent factors per feature value, sharing the interaction term across classes for efficiency. Adds only ~36K params vs linear's ~7K, inference ~2× cost.
Failure: If dev accuracy ≤ 0.8524 (label-linear), interactions don't help or overfit.
```
```python
import json, math, struct, sys
import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 64, 600000, 100

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
    d, C = card["input"]["bytes"], card["classes"]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    bias = np.zeros(C, np.float32)
    linear = rng.normal(0, 0.01, (d, 256, C)).astype(np.float32)
    latent = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)

    params = [bias, linear, latent]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            bsize = len(yb)

            lin_sum = np.zeros((bsize, C), np.float32)
            lat_sum = np.zeros((bsize, K), np.float32)
            lat_sq_sum = np.zeros(bsize, np.float32)

            for j in range(d):
                vals = Xb[:, j]
                lin_sum += linear[j, vals]
                lv = latent[j, vals]
                lat_sum += lv
                lat_sq_sum += np.sum(lv * lv, axis=1)

            inter = 0.5 * (np.sum(lat_sum * lat_sum, axis=1) - lat_sq_sum)
            scores = bias + lin_sum + inter[:, None]

            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(bsize), yb] -= 1.0
            dL_dscores = (probs / bsize).astype(np.float32)

            dbias = dL_dscores.sum(axis=0)
            dlinear = np.zeros_like(linear)
            dlatent = np.zeros_like(latent)
            dL_dinter = dL_dscores.sum(axis=1)

            for j in range(d):
                vals = Xb[:, j]
                for c in range(C):
                    np.add.at(dlinear[j, :, c], vals, dL_dscores[:, c])
                lv = latent[j, vals]
                grad_lat = dL_dinter[:, None] * (lat_sum - lv)
                for u in range(K):
                    np.add.at(dlatent[j, :, u], vals, grad_lat[:, u])

            dbias += L2 * bias
            dlinear += L2 * linear
            dlatent += L2 * latent

            grads = [dbias, dlinear, dlatent]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    out = struct.pack("<4i", 0x464D0001, d, K, C)
    out += bias.astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += latent.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x464D0001) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 256 || d < 1 || d > 256) {
        fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * (C + (long)d * 256 * C + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + C;
    const float *latent = linear + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *lat_sum = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int u = 0; u < K; u++) lat_sum[u] = 0.0f;
        float lat_sq_sum = 0.0f;

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lin_jv = linear + ((long)j * 256 + v) * C;
            for (int c = 0; c < C; c++) scores[c] += lin_jv[c];

            const float *lat_jv = latent + ((long)j * 256 + v) * K;
            for (int u = 0; u < K; u++) {
                float val = lat_jv[u];
                lat_sum[u] += val;
                lat_sq_sum += val * val;
            }
        }

        float inter = 0.0f;
        for (int u = 0; u < K; u++) inter += lat_sum[u] * lat_sum[u];
        inter = 0.5f * (inter - lat_sq_sum);

        for (int c = 0; c < C; c++) scores[c] += inter;

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```