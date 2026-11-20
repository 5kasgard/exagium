```hypothesis
Family: label-fm (factorization machine on raw byte features).
Why: Adult has 14 byte-wise features (quantiled numeric + categorical) with strong pairwise interactions (e.g., education × occupation). A factorization machine models all pairwise interactions via low-rank embeddings (k=8) with only ~32K parameters, adding negligible compute over linear. It should exceed label-linear (0.8524) and approach label-mlp16 (0.8591) at ~1000 Ir/item.
Failure signal: dev accuracy ≤ 0.852 (no gain over linear).
```
```python
"""Factorization machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Exports: int32 magic 0x464D4C31, d, k, C; float32 bias, linear[d*256], emb[d*256*k]."""
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
    d = X.shape[1]
    C = card["classes"]
    assert C == 2, "FM binary only"
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Parameters: bias, linear[d*256], emb[d*256, K]
    bias = np.float32(0.0)
    linear = rng.normal(0, 0.01, (d * 256,)).astype(np.float32)
    emb = rng.normal(0, 0.01, (d * 256, K)).astype(np.float32)

    params = [bias, linear, emb]
    # Adam state: (m, v) for each param
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)

            # Forward
            # linear part: bias + sum_j linear[j*256 + xb[:,j]]
            lin = bias + np.sum(linear[xb + np.arange(d) * 256], axis=1)
            # FM interaction: 0.5 * ( (sum v)^2 - sum v^2 )
            v = emb[xb + np.arange(d) * 256]  # (B, d, K)
            sum_v = v.sum(axis=1)             # (B, K)
            sum_v2 = (v * v).sum(axis=1)      # (B, K)
            inter = 0.5 * ((sum_v * sum_v).sum(axis=1) - sum_v2.sum(axis=1))
            logit = lin + inter                # (B,)

            # Binary cross entropy gradient w.r.t logit
            # loss = log(1 + exp(-y*logit)) with y in {0,1} -> use logits for y=1, -logits for y=0
            # Let target t = 2*y - 1 in {-1, 1}, then loss = log(1 + exp(-t*logit))
            # dloss/dlogit = -t * sigmoid(-t*logit) = sigmoid(logit) - y
            p = 1.0 / (1.0 + np.exp(-logit))   # sigmoid
            grad_logit = (p - yb).astype(np.float32)  # (B,)

            # Gradients
            # bias
            g_bias = grad_logit.sum()
            # linear
            g_linear = np.zeros_like(linear)
            np.add.at(g_linear, xb + np.arange(d) * 256, grad_logit[:, None])
            g_linear += L2 * linear
            # emb
            g_emb = np.zeros_like(emb)
            # d inter / d v_{b,j,k} = sum_v[b,k] - v[b,j,k]
            # so g_emb[idx] += grad_logit[b] * (sum_v[b] - v[b,j])
            # vectorized: for each batch item b, feature j
            diff = sum_v[:, None, :] - v  # (B, d, K)
            # multiply by grad_logit[:, None, None]
            g_vals = grad_logit[:, None, None] * diff  # (B, d, K)
            np.add.at(g_emb, xb + np.arange(d) * 256, g_vals)
            g_emb += L2 * emb

            grads = [np.float32(g_bias), g_linear.astype(np.float32), g_emb.astype(np.float32)]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    MAGIC = 0x464D4C31  # 'FML1'
    out = struct.pack("<4i", MAGIC, d, K, C)
    out += np.array([bias], dtype="<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization machine inference for label-fm family.
 * ./infer weights.bin in.rec N
 * Weights format: int32 magic(0x464D4C31), d, k, C; float32 bias, linear[d*256], emb[d*256*k].
 * Input record: d bytes. Output: 1 byte class (argmax of 2 logits: 0 and logit). */
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
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || k <= 0 || d <= 0) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * k);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *linear = p + 1;
    const float *emb = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * k);

    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }

        float logit = bias;
        for (int u = 0; u < k; u++) sum_v[u] = 0.0f;

        for (int j = 0; j < d; j++) {
            int idx = j * 256 + xi[j];
            logit += linear[idx];
            const float *e = emb + (long)idx * k;
            for (int u = 0; u < k; u++) sum_v[u] += e[u];
        }

        float sum_sq = 0.0f, sq_sum = 0.0f;
        for (int u = 0; u < k; u++) {
            float v = sum_v[u];
            sum_sq += v * v;
            sq_sum += v;
        }
        sq_sum = sq_sum * sq_sum;
        logit += 0.5f * (sq_sum - sum_sq);

        // Two-class scores: class 0 = 0, class 1 = logit
        unsigned char best = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```