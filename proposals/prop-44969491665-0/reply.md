```hypothesis
Factorization Machine (FM) with latent dimension 8 captures pairwise feature interactions
efficiently, which are crucial for tabular data like Adult. It adds only ~32K parameters over
linear (vs. 100K+ for MLP) and inference costs ~14*8=112 multiply-adds per item. If dev
accuracy does not exceed label-linear (0.8524) or Ir/item exceeds ~1500, the family fails.
```
```python
"""Factorization Machine for tabular label tasks: binary FM with k=8.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: int32 magic 0x464D3130, d, k, C=1; float32 w0, w[d][256], v[d][256][k]."""
import json, math, struct, sys
import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-3, 1e-4, 64, 600000, 100
MAGIC = 0x464D3130  # 'FM10'

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.float32)  # 0.0 or 1.0
    d = X.shape[1]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # parameters
    w0 = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    v = (rng.normal(0, 0.01, (d, 256, K)) * np.sqrt(2.0 / K)).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_v, v_v = np.zeros_like(v), np.zeros_like(v)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)

            # forward
            linear = w0 + np.sum(w[np.arange(d), xb.T], axis=0)  # (B,)
            v_sum = np.sum(v[np.arange(d), xb.T], axis=0)        # (B, K)
            v_sq_sum = np.sum(v[np.arange(d), xb.T]**2, axis=0)  # (B, K)
            inter = 0.5 * np.sum(v_sum**2 - v_sq_sum, axis=1)    # (B,)
            score = linear + inter
            p = 1.0 / (1.0 + np.exp(-score))
            grad = p - yb  # (B,)  dL/dscore

            # gradients
            g_w0 = np.mean(grad)
            # g_w[f][val] = mean(grad_i for i where xb[i,f]==val)
            g_w = np.zeros_like(w)
            g_v = np.zeros_like(v)
            for f in range(d):
                vals = xb[:, f]
                for val in range(256):
                    mask = (vals == val)
                    if np.any(mask):
                        g_w[f, val] = np.mean(grad[mask])
                        # dv = grad * (v_sum - v[f,val])
                        g_v[f, val] = np.mean(grad[mask][:, None] * (v_sum[mask] - v[f, val]), axis=0)

            # L2 regularization on w and v
            g_w += L2 * w
            g_v += L2 * v

            # Adam update
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            m_w0_hat = m_w0 / (1 - beta1**t)
            v_w0_hat = v_w0 / (1 - beta2**t)
            w0 -= LR * m_w0_hat / (np.sqrt(v_w0_hat) + eps)

            # w
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            m_w_hat = m_w / (1 - beta1**t)
            v_w_hat = v_w / (1 - beta2**t)
            w -= LR * m_w_hat / (np.sqrt(v_w_hat) + eps)

            # v
            m_v = beta1 * m_v + (1 - beta1) * g_v
            v_v = beta2 * v_v + (1 - beta2) * g_v * g_v
            m_v_hat = m_v / (1 - beta1**t)
            v_v_hat = v_v / (1 - beta2**t)
            v -= LR * m_v_hat / (np.sqrt(v_v_hat) + eps)

    # write weights
    with open(sys.argv[4], "wb") as f:
        f.write(struct.pack("<4i", MAGIC, d, K, 1))
        f.write(struct.pack("<f", w0))
        f.write(w.astype("<f4").tobytes())
        f.write(v.astype("<f4").tobytes())

if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Binary FM with k=8. Output: one byte per record (0 or 1). */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 1) { fprintf(stderr, "expected C=1\n"); return 3; }
    size_t expected = 16 + 4 + 4L*d*256 + 4L*d*256*K;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *v = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *v_sum = malloc(K * sizeof(float));
    float *v_sq_sum = malloc(K * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = w0;
        for (int u = 0; u < K; u++) { v_sum[u] = 0.0f; v_sq_sum[u] = 0.0f; }

        for (int f = 0; f < d; f++) {
            unsigned char val = xi[f];
            score += w[(long)f * 256 + val];
            const float *vf = v + ((long)f * 256 + val) * K;
            for (int u = 0; u < K; u++) {
                float vu = vf[u];
                v_sum[u] += vu;
                v_sq_sum[u] += vu * vu;
            }
        }
        float inter = 0.0f;
        for (int u = 0; u < K; u++) {
            float s = v_sum[u];
            inter += 0.5f * (s * s - v_sq_sum[u]);
        }
        score += inter;
        unsigned char pred = (score > 0.0f) ? 1 : 0;  // sigmoid(score) > 0.5 <=> score > 0
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```