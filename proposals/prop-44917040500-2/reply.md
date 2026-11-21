```hypothesis
Family: label-fm (Factorization Machine). Character-level language modeling benefits from pairwise interactions between nearby bytes (e.g., bigrams). A linear model misses these; a full MLP is heavy. FM adds a low-rank interaction term (k=8) on top of the linear per-position embeddings, adding only ~32k parameters. It should beat label-linear (0.3375) with minimal CPU cost increase. If dev accuracy ≤ 0.35, the FM interactions are not capturing useful structure.
```

```python
"""label-fm: Factorization Machine for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Model: linear per-position one-hot weights + FM pairwise interactions with rank k.
Exported as flat float32 arrays: magic, d, C, k, L[d][256][C], E[d][256][k], W_fm[C][k]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (fixed, no tuning split used)
K = 8
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 100
SAMPLES_PER_EPOCH = 600000  # same as label-linear


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def softmax_cross_entropy_grad(logits, y):
    """Return gradient of softmax cross-entropy w.r.t logits, shape (batch, C)."""
    # logits: (batch, C), y: (batch,) int
    batch = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(batch), y] -= 1.0
    return probs / batch


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Parameters
    L = (rng.normal(0, 0.01, (d, 256, C))).astype(np.float32)
    E = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)
    W_fm = (rng.normal(0, 0.01, (C, K))).astype(np.float32)
    params = [L, E, W_fm]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES_PER_EPOCH / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (batch, d)
            yb = y[idx]          # (batch,)

            # Forward
            batch = xb.shape[0]
            linear = np.zeros((batch, C), dtype=np.float32)
            sum_e = np.zeros((batch, K), dtype=np.float32)
            sum_sq = np.zeros((batch, K), dtype=np.float32)
            for j in range(d):
                v = xb[:, j]                     # (batch,)
                linear += L[j, v, :]             # (batch, C)
                e = E[j, v, :]                   # (batch, K)
                sum_e += e
                sum_sq += e * e
            I = 0.5 * (sum_e * sum_e - sum_sq)   # (batch, K)
            fm = I @ W_fm.T                      # (batch, C)
            logits = linear + fm
            grad_logits = softmax_cross_entropy_grad(logits, yb).astype(np.float32)

            # Backward
            # dW_fm = I.T @ grad_logits  -> (K, C)
            dW_fm = I.T @ grad_logits
            # dI = grad_logits @ W_fm    -> (batch, K)
            dI = grad_logits @ W_fm
            # dsum_e = dI * sum_e
            dsum_e = dI * sum_e
            # dsum_sq = -0.5 * dI
            dsum_sq = -0.5 * dI

            dL = np.zeros_like(L)
            dE = np.zeros_like(E)
            for j in range(d):
                v = xb[:, j]
                # dL: for each sample, add grad_logits to dL[j, v, :]
                np.add.at(dL[j], v, grad_logits)
                # dE: dE = dsum_e + 2 * E * dsum_sq
                e = E[j, v, :]
                de = dsum_e + 2.0 * e * dsum_sq
                np.add.at(dE[j], v, de)

            # L2 regularization gradients
            dL += L2 * L
            dE += L2 * E
            dW_fm += L2 * W_fm

            grads = [dL, dE, dW_fm]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    magic = 0x464D4C31  # 'FML1' little-endian
    header = struct.pack("<4i", magic, d, C, K)
    # Flatten in order: L (d*256*C), E (d*256*K), W_fm (C*K)
    L_bytes = L.astype("<f4").tobytes()
    E_bytes = E.astype("<f4").tobytes()
    W_fm_bytes = W_fm.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + L_bytes + E_bytes + W_fm_bytes)


if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte class index.
 * Weights format: magic(0x464D4C31), d, C, k, then float32 arrays:
 *   L[d][256][C], E[d][256][k], W_fm[C][k] (all row-major). */
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
    int d = hd[1], C = hd[2], K = hd[3];
    if (d != 16 || C < 1 || C > 256 || K < 1) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16L
        + 4L * d * 256 * C
        + 4L * d * 256 * K
        + 4L * C * K;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *L = (const float *)(hd + 4);
    const float *E = L + (long)d * 256 * C;
    const float *W_fm = E + (long)d * 256 * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *linear = malloc(sizeof(float) * C);
    float *sum_e = malloc(sizeof(float) * K);
    float *sum_sq = malloc(sizeof(float) * K);
    float *I = malloc(sizeof(float) * K);
    float *fm = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memset(linear, 0, C * sizeof(float));
        memset(sum_e, 0, K * sizeof(float));
        memset(sum_sq, 0, K * sizeof(float));

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *Lj = L + ((long)j * 256 + v) * C;
            for (int c = 0; c < C; c++) linear[c] += Lj[c];
            const float *Ej = E + ((long)j * 256 + v) * K;
            for (int f = 0; f < K; f++) {
                float e = Ej[f];
                sum_e[f] += e;
                sum_sq[f] += e * e;
            }
        }

        for (int f = 0; f < K; f++) {
            I[f] = 0.5f * (sum_e[f] * sum_e[f] - sum_sq[f]);
        }

        memset(fm, 0, C * sizeof(float));
        for (int f = 0; f < K; f++) {
            const float *Wc = W_fm + (long)C * f;
            float If = I[f];
            for (int c = 0; c < C; c++) fm[c] += If * Wc[c];
        }

        unsigned char best = 0;
        for (int c = 0; c < C; c++) {
            float score = linear[c] + fm[c];
            if (score > linear[best] + fm[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```