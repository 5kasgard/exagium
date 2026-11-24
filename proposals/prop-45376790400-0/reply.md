```hypothesis
Family: label-fm (Factorization Machine with shared latent vectors).
Why: Adult has strong feature interactions (e.g., education × occupation, marital-status × relationship) that linear models miss. A factorization machine captures all pairwise interactions via low-rank embeddings (k=8) with only O(d·k) extra inference work. It should beat label-linear (0.8524) while staying far cheaper than label-mlp16 (2325 Ir/item).
Failure signal: dev accuracy ≤ 0.853 or Ir/item > 1500.
```

```python
"""label-fm: Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Format: magic 0x464D4C31, d, k, C; float32 w0[C], w[d][256][C], v[d][256][k]."""
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
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Parameters: w0[C], w[d,256,C], v[d,256,K]
    w0 = np.zeros(C, np.float32)
    w = np.zeros((d, 256, C), np.float32)
    v = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    m_w0, v_w0 = np.zeros_like(w0), np.zeros_like(w0)
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_v, v_v = np.zeros_like(v), np.zeros_like(v)
    t = 0

    def forward_batch(Xb):
        # Xb: (b, d) uint8
        b = Xb.shape[0]
        # First-order term
        fo = w0 + np.sum(w[np.arange(d), Xb.T, :], axis=0)  # (b, C)
        # Second-order term (shared v)
        v_sum = np.sum(v[np.arange(d), Xb.T, :], axis=0)    # (b, K)
        v_sq_sum = np.sum(v[np.arange(d), Xb.T, :] ** 2, axis=0)  # (b, K)
        interaction = 0.5 * np.sum(v_sum ** 2 - v_sq_sum, axis=1, keepdims=True)  # (b, 1)
        return fo + interaction  # (b, C)

    def softmax_grad(logits, yb):
        b = logits.shape[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(b), yb] -= 1.0
        return probs / b  # (b, C)

    for epoch in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            b = len(yb)

            # Forward
            v_sel = v[np.arange(d), Xb.T]          # (d, b, K)
            v_sum = np.sum(v_sel, axis=0)          # (b, K)
            v_sq_sum = np.sum(v_sel ** 2, axis=0)  # (b, K)
            inter = 0.5 * (v_sum ** 2 - v_sq_sum)  # (b, K)
            inter_sum = np.sum(inter, axis=1, keepdims=True)  # (b, 1)

            fo = w0 + np.sum(w[np.arange(d), Xb.T, :], axis=0)  # (b, C)
            logits = fo + inter_sum
            grad_out = softmax_grad(logits, yb)                 # (b, C)

            # Gradients
            g_w0 = grad_out.sum(axis=0)                         # (C,)
            g_w = np.zeros_like(w)
            for j in range(d):
                g_w[j, Xb[:, j], :] += grad_out                 # (b, C) accumulates
            # v gradient: dL/dv = sum_c (grad_out[:,c]) * (v_sum - v_sel_j)
            # grad_out summed over classes -> (b,)
            g_inter = grad_out.sum(axis=1)                      # (b,)
            g_v = np.zeros_like(v)
            for j in range(d):
                # (v_sum - v_sel_j) is (b, K)
                diff = v_sum - v_sel[j]
                g_v[j, Xb[:, j], :] += g_inter[:, None] * diff

            # L2 regularization
            g_w += L2 * w
            g_v += L2 * v

            t += 1
            beta1, beta2 = 0.9, 0.999
            # Adam updates
            for param, grad, m, v_adam in [(w0, g_w0, m_w0, v_w0),
                                           (w, g_w, m_w, v_w),
                                           (v, g_v, m_v, v_v)]:
                m *= beta1; m += (1 - beta1) * grad
                v_adam *= beta2; v_adam += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v_adam / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    out += w0.astype("<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index (argmax). */
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
    if (C < 1 || C > 256 || K < 1) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16L + 4L * (C + (long)d * 256 * C + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *w0 = (const float *)(hd + 4);
    const float *w  = w0 + C;
    const float *v  = w  + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *v_sum = malloc(sizeof(float) * K);
    float *v_sq_sum = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* first-order */
        for (int c = 0; c < C; c++) scores[c] = w0[c];
        for (int j = 0; j < d; j++) {
            const float *wj = w + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += wj[c];
        }

        /* second-order (shared v) */
        memset(v_sum, 0, sizeof(float) * K);
        memset(v_sq_sum, 0, sizeof(float) * K);
        for (int j = 0; j < d; j++) {
            const float *vj = v + ((long)j * 256 + xi[j]) * K;
            for (int u = 0; u < K; u++) {
                float val = vj[u];
                v_sum[u] += val;
                v_sq_sum[u] += val * val;
            }
        }
        float inter = 0.0f;
        for (int u = 0; u < K; u++) inter += 0.5f * (v_sum[u] * v_sum[u] - v_sq_sum[u]);
        for (int c = 0; c < C; c++) scores[c] += inter;

        /* argmax */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```