```hypothesis
Family: label-fm (factorization machine with learned embeddings). 
Why: Adult has 14 mixed-type features; a second-order FM captures pairwise interactions cheaply by sharing a low-rank embedding per feature value. 
We keep the linear term (like label-linear) and add a rank-8 interaction term, adding only ~16k parameters. 
Inference cost stays near label-linear (O(d·k) with k=8). 
Failure signal: dev accuracy ≤ label-linear (0.8524) or Ir/item > 1000.
```

```python
"""label-fm: Factorization Machine with learned embeddings for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Format: int32 magic 0x464D4C31, d, k, C; float32 emb[d][256][k], W_lin[d*k][C], b_lin[C], W_int[k][C], b_int[C]."""
import json, math, struct, sys

import numpy as np

K = 8                     # embedding rank
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


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
    d, C = X.shape[1], card["classes"]
    rng = np.random.default_rng(SEED)

    # embeddings: (d, 256, K)
    emb = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)
    W_lin = (rng.normal(0, 0.01, (d * K, C))).astype(np.float32)
    b_lin = np.zeros(C, np.float32)
    W_int = (rng.normal(0, 0.01, (K, C))).astype(np.float32)
    b_int = np.zeros(C, np.float32)

    params = [emb, W_lin, b_lin, W_int, b_int]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    steps = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return (probs / len(yy)).astype(np.float32)

    for _ in range(steps):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            # gather embeddings
            e = emb[:, xb.T, :].transpose(1, 0, 2)          # (batch, d, K)
            e_flat = e.reshape(len(yb), d * K)              # (batch, d*K)

            # linear logits
            lin_logits = e_flat @ W_lin + b_lin

            # FM interaction term: 0.5 * ((sum e)^2 - sum(e^2))
            sum_e = e.sum(axis=1)                           # (batch, K)
            sum_e_sq = (e * e).sum(axis=1)
            inter = 0.5 * (sum_e * sum_e - sum_e_sq)        # (batch, K)
            int_logits = inter @ W_int + b_int

            logits = lin_logits + int_logits
            g = softmax_grad(logits, yb)                    # (batch, C)

            # gradients
            g_lin = e_flat.T @ g + L2 * W_lin               # (d*K, C)
            g_b_lin = g.sum(axis=0)                         # (C,)

            # interaction gradients
            g_int = inter.T @ g + L2 * W_int                # (K, C)
            g_b_int = g.sum(axis=0)                         # (C,)

            # backprop to embeddings
            # dL/de_flat = g @ W_lin.T   -> (batch, d*K)
            g_flat = g @ W_lin.T                            # (batch, d*K)
            g_emb = g_flat.reshape(len(yb), d, K)           # (batch, d, K)

            # dL/dinter = g @ W_int.T  -> (batch, K)
            g_inter = g @ W_int.T                           # (batch, K)
            # dinter/de = e - (sum_e broadcast) ? Actually:
            # inter = 0.5*(sum_e^2 - sum(e^2))
            # dinter/de_i = sum_e - e_i
            # So for each feature i: g_emb[:, i] += g_inter * (sum_e - e[:, i])
            for i in range(d):
                g_emb[:, i] += g_inter * (sum_e - e[:, i])

            g_emb += L2 * emb[:, xb.T, :].transpose(1, 0, 2)

            grads = [g_emb.transpose(1, 0, 2), g_lin, g_b_lin, g_int, g_b_int]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # pack weights
    magic = 0x464D4C31  # "FML1"
    out = struct.pack("<4i", magic, d, K, C)
    out += emb.astype("<f4").tobytes()
    out += W_lin.astype("<f4").tobytes()
    out += b_lin.astype("<f4").tobytes()
    out += W_int.astype("<f4").tobytes()
    out += b_int.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from header). Output: 1 byte class index. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }
    long expected = 16L
        + 4L * d * 256 * K           /* emb */
        + 4L * d * K * C             /* W_lin */
        + 4L * C                     /* b_lin */
        + 4L * K * C                 /* W_int */
        + 4L * C;                    /* b_int */
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W_lin = emb + d * 256 * K;
    const float *b_lin = W_lin + d * K * C;
    const float *W_int = b_lin + C;
    const float *b_int = W_int + K * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *e = malloc(sizeof(float) * d * K);
    float *sum_e = malloc(sizeof(float) * K);
    float *sum_e_sq = malloc(sizeof(float) * K);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // gather embeddings
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * K;
            float *dst = e + j * K;
            memcpy(dst, src, K * sizeof(float));
        }

        // linear logits
        for (int c = 0; c < C; c++) logits[c] = b_lin[c];
        for (int j = 0; j < d * K; j++) {
            float ev = e[j];
            if (ev != 0.0f) {
                for (int c = 0; c < C; c++) logits[c] += ev * W_lin[(long)j * C + c];
            }
        }

        // interaction term
        memset(sum_e, 0, K * sizeof(float));
        memset(sum_e_sq, 0, K * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *ej = e + j * K;
            for (int u = 0; u < K; u++) {
                float v = ej[u];
                sum_e[u] += v;
                sum_e_sq[u] += v * v;
            }
        }
        for (int c = 0; c < C; c++) logits[c] += b_int[c];
        for (int u = 0; u < K; u++) {
            float inter = 0.5f * (sum_e[u] * sum_e[u] - sum_e_sq[u]);
            if (inter != 0.0f) {
                for (int c = 0; c < C; c++) logits[c] += inter * W_int[(long)u * C + c];
            }
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```