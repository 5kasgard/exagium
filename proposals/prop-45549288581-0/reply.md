```hypothesis
Factorization Machine (FM) with rank 4 captures pairwise feature interactions crucial for tabular data like Adult, where combinations (e.g., education×occupation) are predictive. The model adds only ~18k parameters (d=14, k=4) and computes interactions in O(d·k) time, so it should beat label-linear (0.8524) while staying faster than label-mlp16 (2325 Ir/item). If dev accuracy ≤ 0.8524, the interactions are not sufficiently exploited or rank is too low.
```

```python
"""Factorization Machine (rank k) for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: int32 magic 0x464D31, d, k, C; float32 w0; float32 lin[d][256]; float32 V[d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-3, 1e-4, 64, 600000, 100


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
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)  # 0 or 1
    d = X.shape[1]
    C = card["classes"]
    assert C == 2, "FM binary only"
    rng = np.random.default_rng(SEED)

    # Parameters
    w0 = np.array(0.0, dtype=np.float32)
    lin = np.zeros((d, 256), dtype=np.float32)
    V = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    params = [w0, lin, V]
    shapes = [p.shape for p in params]
    flat_params = np.concatenate([p.ravel() for p in params])
    flat_grads = np.zeros_like(flat_params)
    m = np.zeros_like(flat_params)
    v = np.zeros_like(flat_params)
    t = 0

    def unpack(flat):
        idx = 0
        out = []
        for shp in shapes:
            sz = int(np.prod(shp))
            out.append(flat[idx:idx + sz].reshape(shp))
            idx += sz
        return out

    def pack(arrs):
        return np.concatenate([a.ravel() for a in arrs])

    def forward(X_batch):
        # X_batch: (n, d) uint8
        n = X_batch.shape[0]
        linear = np.zeros(n, dtype=np.float32)
        sum_v = np.zeros((n, K), dtype=np.float32)
        sum_v_sq = np.zeros((n, K), dtype=np.float32)
        for j in range(d):
            xj = X_batch[:, j]
            linear += lin[j, xj]
            vj = V[j, xj]          # (n, K)
            sum_v += vj
            sum_v_sq += vj * vj
        inter = 0.5 * np.sum(sum_v * sum_v - sum_v_sq, axis=1)
        logits = w0 + linear + inter
        return logits

    def loss_grad(logits, y_batch):
        # binary cross-entropy with logits
        # p = sigmoid(logits), loss = -[y log p + (1-y) log(1-p)]
        # dL/dlogits = p - y
        max_logits = np.maximum(logits, 0)
        log_exp = np.log1p(np.exp(-np.abs(logits))) + max_logits
        p = 1.0 / (1.0 + np.exp(-logits))
        diff = p - y_batch.astype(np.float32)  # (n,)
        # gradients
        gw0 = np.mean(diff)
        n = len(diff)
        glin = np.zeros_like(lin)
        gV = np.zeros_like(V)
        for j in range(d):
            xj = X_batch[:, j]
            # linear gradient
            np.add.at(glin[j], xj, diff)
            # V gradient: diff * (sum_v - vj)
            vj = V[j, xj]          # (n, K)
            factor = (diff[:, None] * (sum_v - vj)).astype(np.float32)  # (n, K)
            np.add.at(gV[j], xj, factor)
        # L2 regularization
        glin += L2 * lin
        gV += L2 * V
        return pack([gw0, glin, gV])

    n = X.shape[0]
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            # forward
            logits = forward(Xb)
            # gradient
            grad = loss_grad(logits, yb)
            # Adam update
            t += 1
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad * grad
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            flat_params -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
            w0, lin, V = unpack(flat_params)

    # Write weights
    magic = 0x464D31  # "FM1"
    out = struct.pack("<4i", magic, d, K, C)
    out += w0.astype("<f4").tobytes()
    out += lin.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (one per feature). Output: one byte class (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K <= 0 || d <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *lin = p + 1;
    const float *V = lin + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_v[16];  // K <= 16
    float sum_v_sq[16];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float linear = 0.0f;
        for (int kk = 0; kk < K; kk++) { sum_v[kk] = 0.0f; sum_v_sq[kk] = 0.0f; }
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            linear += lin[(long)j * 256 + v];
            const float *Vj = V + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = Vj[kk];
                sum_v[kk] += val;
                sum_v_sq[kk] += val * val;
            }
        }
        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            float diff = sum_v[kk] * sum_v[kk] - sum_v_sq[kk];
            inter += diff;
        }
        inter *= 0.5f;
        float logit = w0 + linear + inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```