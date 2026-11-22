```hypothesis
Family: label-fm (Factorization Machine with rank-8 pairwise interactions).
Why: Adult income has known feature interactions (e.g., education × occupation, marital-status × relationship).
Linear models miss these; FM captures all pairwise interactions with O(d·K) params. Rank K=8 balances
expressivity and overfitting on 26k samples. Inference is O(d·K) per item, ~3× linear cost.
Failure signal: If dev accuracy ≤ 0.853 (≤ label-linear), interactions aren't helping or optimization failed.
```
```python
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 256, 500000, 100


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
    y_bin = np.where(y == 0, -1.0, 1.0).astype(np.float32)
    C = card["classes"]
    assert C == 2
    rng = np.random.default_rng(SEED)
    num_cols = card["input"]["bytes"]
    enc = []
    enc_shapes = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None].astype(np.float32)
            enc.append(E)
            enc_shapes.append(1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
            enc_shapes.append(len(vals))
    D = sum(enc_shapes)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, d = Z.shape
    assert d == D
    w0 = np.float32(0.0)
    w = rng.normal(0, 0.01, d).astype(np.float32)
    V = rng.normal(0, 0.01, (d, K)).astype(np.float32)
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z = Z[idx]
            yy = y_bin[idx]
            b = len(yy)
            linear = w0 + z @ w
            zV = z @ V
            z2V2 = (z * z) @ (V * V)
            inter = 0.5 * np.sum(zV * zV - z2V2, axis=1)
            logit = linear + inter
            y_logit = yy * logit
            sig = 1.0 / (1.0 + np.exp(y_logit))
            grad_logit = -yy * sig
            grad_w0 = np.mean(grad_logit)
            grad_w = (z.T @ grad_logit) / b + L2 * w
            grad_V = (z.T @ (grad_logit[:, None] * zV)) / b - ((z * z).T @ (grad_logit[:, None] * V)) / b + L2 * V
            t += 1
            m_w0 = beta1 * m_w0 + (1 - beta1) * grad_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * grad_w0 * grad_w0
            w0 -= LR * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)
            m_w = beta1 * m_w + (1 - beta1) * grad_w
            v_w = beta2 * v_w + (1 - beta2) * grad_w * grad_w
            w -= LR * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)
            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * grad_V * grad_V
            V -= LR * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)
    T_linear = np.zeros((num_cols, 256), np.float32)
    T_factor = np.zeros((num_cols, 256, K), np.float32)
    off = 0
    for j in range(num_cols):
        dim_j = enc_shapes[j]
        E = enc[j]
        w_seg = w[off:off + dim_j]
        V_seg = V[off:off + dim_j]
        T_linear[j] = E @ w_seg
        T_factor[j] = E @ V_seg
        off += dim_j
    magic = 0x464D0001
    hdr = struct.pack("<4i", magic, num_cols, K, C)
    out = hdr + struct.pack("<f", w0) + T_linear.astype("<f4").tobytes() + T_factor.astype("<f4").tobytes()
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
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wb[0] | wb[1]<<8 | wb[2]<<16 | wb[3]<<24;
    if (magic != 0x464D0001) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    int num_cols = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || num_cols <= 0 || K <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4 + (long)num_cols * 256 * 4 * (1 + K);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }
    const float *ptr = (const float *)(wb + 16);
    float w0 = *ptr++;
    const float *T_linear = ptr; ptr += (long)num_cols * 256;
    const float *T_factor = ptr;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_f = malloc(sizeof(float) * K);
    float *sum_sq_f = malloc(sizeof(float) * K);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != num_cols) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, num_cols); return 3; }
        float linear = w0;
        for (int f = 0; f < K; f++) { sum_f[f] = 0.0f; sum_sq_f[f] = 0.0f; }
        for (int j = 0; j < num_cols; j++) {
            unsigned char byte = xi[j];
            linear += T_linear[(long)j * 256 + byte];
            const float *tf = T_factor + ((long)j * 256 + byte) * K;
            for (int f = 0; f < K; f++) {
                float v = tf[f];
                sum_f[f] += v;
                sum_sq_f[f] += v * v;
            }
        }
        float inter = 0.0f;
        for (int f = 0; f < K; f++) inter += (sum_f[f] * sum_f[f] - sum_sq_f[f]) * 0.5f;
        float logit = linear + inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```