```hypothesis
Family: label-fm (factorization machine with rank-8 latent vectors per class).
Why: Adult has strong feature interactions (e.g., education × occupation); linear models miss these. A factorization machine captures pairwise interactions efficiently via low-rank embeddings, adding only O(d*256*C*k) parameters. Inference computes interactions in O(d*C*k) using the (sum^2 - sum of squares) trick, keeping latency low.
Failure: If dev accuracy ≤ label-linear (0.8524) or instructions/item exceed label-mlp16 (2325), the model is not an improvement.
```

```python
"""Factorization Machine for tabular label tasks: fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Encoding: ordinal -> z-scored scalar, categorical -> one-hot over training values.
Model: per-class linear weights + rank-k latent vectors (k=8). Adam, L2, deterministic."""
import json, math, struct, sys

import numpy as np

K_RANK = 8
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
    d = X.shape[1]
    rng = np.random.default_rng(SEED)

    # Build per-feature encoding matrices E_j (256 x dim_j)
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None]          # (256,1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0                # (256, n_cat)
        enc.append(E.astype(np.float32))

    # Concatenated design matrix Z (n x K)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1)
    n, K = Z.shape

    # Parameters: W_lin (K, C), b_lin (C), W_lat (K, C, K_RANK)
    W_lin = (rng.normal(0, math.sqrt(2.0 / K), (K, C))).astype(np.float32)
    b_lin = np.zeros(C, np.float32)
    W_lat = (rng.normal(0, math.sqrt(2.0 / K), (K, C, K_RANK))).astype(np.float32)
    params = [W_lin, b_lin, W_lat]

    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]                     # (b, K), (b,)
            bsz = len(yy)

            # Forward
            lin = z @ W_lin + b_lin                    # (b, C)
            logits = lin.copy()
            for c in range(C):
                v = W_lat[:, c, :]                     # (K, k)
                zv = z @ v                             # (b, k)
                zv_sq = z @ (v * v)                    # (b, k)
                inter = 0.5 * np.sum(zv * zv - zv_sq, axis=1)
                logits[:, c] += inter

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yy] -= 1.0
            grad_logits = (probs / bsz).astype(np.float32)          # (b, C)

            # Gradients
            grad_W_lin = z.T @ grad_logits + L2 * W_lin
            grad_b_lin = grad_logits.sum(axis=0)
            grad_W_lat = np.zeros_like(W_lat)
            for c in range(C):
                g = grad_logits[:, c]                         # (b,)
                v = W_lat[:, c, :]                            # (K, k)
                zv = z @ v                                    # (b, k)
                # dL/dv = z.T @ (g[:,None] * zv) - (z.T @ g)[:,None] * v
                grad_W_lat[:, c, :] = z.T @ (g[:, None] * zv) - np.outer(z.T @ g, v)

            grad_W_lat += L2 * W_lat
            grads = [grad_W_lin, grad_b_lin, grad_W_lat]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Bake per-byte lookup tables
    # T_lin[j][v] = enc_j[v] @ W_lin   -> (d, 256, C)
    # T_lat[j][v] = enc_j[v] @ W_lat   -> (d, 256, C, K_RANK)
    T_lin = np.zeros((d, 256, C), np.float32)
    T_lat = np.zeros((d, 256, C, K_RANK), np.float32)
    for j, E in enumerate(enc):
        T_lin[j] = E @ W_lin
        T_lat[j] = E @ W_lat.reshape(K, C * K_RANK).reshape(C, K_RANK)

    # Write weights: magic, d, k, C; bias[C]; T_lin[d*256*C]; T_lat[d*256*C*k]
    magic = 0x464D4C31  # "FML1"
    out = struct.pack("<4i", magic, d, K_RANK, C)
    out += b_lin.astype("<f4").tobytes()
    out += T_lin.astype("<f4").tobytes()
    out += T_lat.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: 1 byte class index.
 * Weights layout: magic(4), d, k, C; bias[C]; T_lin[d*256*C]; T_lat[d*256*C*k] (all float32 LE). */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (C + (long)d*256*C + (long)d*256*C*K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *T_lin = bias + C;
    const float *T_lat = T_lin + (long)d*256*C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(C * sizeof(float));
    float *sum_lat = malloc(C * K * sizeof(float));
    float *sum_lat_sq = malloc(C * K * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(scores, bias, C * sizeof(float));
        memset(sum_lat, 0, C * K * sizeof(float));
        memset(sum_lat_sq, 0, C * K * sizeof(float));

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lin_ptr = T_lin + ((long)j*256 + v) * C;
            const float *lat_ptr = T_lat + ((long)j*256 + v) * C * K;
            for (int c = 0; c < C; c++) {
                scores[c] += lin_ptr[c];
                for (int l = 0; l < K; l++) {
                    float val = lat_ptr[c*K + l];
                    sum_lat[c*K + l] += val;
                    sum_lat_sq[c*K + l] += val * val;
                }
            }
        }

        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int l = 0; l < K; l++) {
                float s = sum_lat[c*K + l];
                inter += s*s - sum_lat_sq[c*K + l];
            }
            scores[c] += 0.5f * inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```