```hypothesis
Family: label-fm (Factorization Machine with linear term). Captures pairwise feature interactions via low-rank embeddings (K=8), crucial for Adult where combinations like education×occupation and marital-status×relationship strongly predict income. Uses same per-byte lookup tables as label-linear for fast inference. If dev accuracy ≤ label-linear (0.8524), interactions aren't learnable at this scale or K is too small.
```
```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Encoding: ordinal z-scored, categorical one-hot (255=unseen). Linear + FM(K=8) per class.
Adam lr=1e-2, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)). Seed=1.
Weights: magic 0x464D4C31, d, K, C; float32 T_lin[d][256][C], b_lin[C], T_fm[d][256][K][C]."""
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
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # Build encoders per feature
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])          # (256,1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)                                               # (256, n_vals)

    # Concatenated design matrix
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, k = Z.shape

    # Parameters: W_lin [k,C], V_fm [k,K,C], b_lin [C]
    W_lin = np.zeros((k, C), np.float32)
    V_fm  = (rng.uniform(-0.01, 0.01, (k, K, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b_lin = np.zeros(C, np.float32)
    params = [W_lin, V_fm, b_lin]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            m = len(yy)

            # Forward
            lin = z @ W_lin + b_lin                                    # [m,C]
            # FM: for each class c, factor f: 0.5*((z@v)^2 - z^2@v^2)
            z2 = z * z
            fm = np.zeros((m, C), np.float32)
            for c in range(C):
                for f in range(K):
                    v = V_fm[:, f, c]                                  # [k]
                    v2 = v * v
                    sum_v  = z  @ v
                    sum_v2 = z2 @ v2
                    fm[:, c] += 0.5 * (sum_v * sum_v - sum_v2)
            logits = lin + fm

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(m), yy] -= 1.0
            grad_logits = (probs / m).astype(np.float32)               # [m,C]

            # Backprop
            grad_W_lin = z.T @ grad_logits + L2 * W_lin                # [k,C]
            grad_b_lin = grad_logits.sum(axis=0)                       # [C]
            grad_V_fm = np.zeros_like(V_fm)
            for c in range(C):
                g = grad_logits[:, c]                                  # [m]
                for f in range(K):
                    v = V_fm[:, f, c]
                    sum_v = z @ v
                    # d(fm)/dv = z * sum_v - z^2 * v
                    grad_v = (z.T * sum_v - z2.T * v) @ g / m + L2 * v
                    grad_V_fm[:, f, c] = grad_v.astype(np.float32)

            grads = [grad_W_lin, grad_V_fm, grad_b_lin]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m1, v1 = state[i]
                m1 *= 0.9; m1 += 0.1 * gr
                v1 *= 0.999; v1 += 0.001 * gr * gr
                p -= LR * (m1 / (1 - 0.9 ** t)) / (np.sqrt(v1 / (1 - 0.999 ** t)) + 1e-8)

    # Build per-byte lookup tables
    # T_lin[j][v][c] = sum_k enc[j][v,k] * W_lin[offset+k, c]
    # T_fm [j][v][f][c] = sum_k enc[j][v,k] * V_fm[offset+k, f, c]
    off = 0
    T_lin_list, T_fm_list = [], []
    for j, E in enumerate(enc):
        kj = E.shape[1]
        Wj = W_lin[off:off+kj]                      # [kj, C]
        Vj = V_fm[off:off+kj]                       # [kj, K, C]
        T_lin_list.append((E.astype(np.float32) @ Wj).astype(np.float32))          # [256, C]
        # E @ Vj: [256, kj] @ [kj, K, C] -> [256, K, C]
        T_fm_list.append(np.tensordot(E.astype(np.float32), Vj, axes=([1],[0])).astype(np.float32))
        off += kj

    T_lin = np.stack(T_lin_list)                    # [d, 256, C]
    T_fm  = np.stack(T_fm_list)                     # [d, 256, K, C]

    d = X.shape[1]
    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    out += T_lin.astype("<f4").tobytes()
    out += b_lin.astype("<f4").tobytes()
    out += T_fm.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine (linear + pairwise low-rank) for label tasks.
 * ./infer weights.bin in.rec N
 * Input record: d bytes. Output: 1 byte class index (argmax, first max wins).
 * Weights format: magic 0x464D4C31, int32 d, K, C;
 *   float32 T_lin[d][256][C], b_lin[C], T_fm[d][256][K][C] */
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
    if (C < 1 || C > 256 || K < 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * C + C + (long)d * 256 * K * C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T_lin = (const float *)(hd + 4);
    const float *b_lin = T_lin + (long)d * 256 * C;
    const float *T_fm  = b_lin + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *score = malloc(sizeof(float) * C);
    float *sum_v  = malloc(sizeof(float) * K * C);
    float *sum_v2 = malloc(sizeof(float) * K * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) score[c] = b_lin[c];
        for (int f = 0; f < K; f++)
            for (int c = 0; c < C; c++)
                sum_v[f*C+c] = sum_v2[f*C+c] = 0.0f;

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lin_row = T_lin + ((long)j * 256 + v) * C;
            const float *fm_row  = T_fm  + ((long)j * 256 + v) * K * C;
            for (int c = 0; c < C; c++) score[c] += lin_row[c];
            for (int f = 0; f < K; f++) {
                const float *fm_f = fm_row + f * C;
                for (int c = 0; c < C; c++) {
                    float val = fm_f[c];
                    sum_v[f*C+c]  += val;
                    sum_v2[f*C+c] += val * val;
                }
            }
        }

        for (int c = 0; c < C; c++) {
            float fm = 0.0f;
            for (int f = 0; f < K; f++) {
                float sv = sum_v[f*C+c], sv2 = sum_v2[f*C+c];
                fm += 0.5f * (sv * sv - sv2);
            }
            score[c] += fm;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++)
            if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```