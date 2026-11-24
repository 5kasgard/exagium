```hypothesis
Family: label-fm (Factorization Machine with linear part). 
Why: Adult has 14 mixed-type features; pairwise interactions capture combinatorial effects (e.g., education × occupation) that a linear model misses, while a tiny latent dimension (k=4) keeps inference fast. 
Failure signal: dev accuracy ≤ label-linear (0.8524) or inference cost > 2× label-linear.
```
```python
"""Factorization Machine (degree 2) with linear part for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encodes ordinal as z-scored, categorical as one-hot (like label-linear).
Learns linear weights on concatenated embeddings + per-class FM embeddings (k=4).
Exports: magic 0x464D3131, d, k, C; T[d][256][C] (linear), b[C]; V[C][d][256][k] (FM)."""
import json, math, struct, sys

import numpy as np

K_FM = 4
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
    C = card["classes"]
    d = X.shape[1]
    rng = np.random.default_rng(SEED)

    # Build fixed encodings E_j (256, D_j) as in label-linear
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    Ds = [E.shape[1] for E in enc]
    K = sum(Ds)

    # Concatenated design matrix Z (n, K)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n = Z.shape[0]

    # Parameters: linear W (K, C), b (C); FM V_c (C, d, 256, k)
    W = (rng.uniform(-1, 1, (K, C)) * math.sqrt(6.0 / K)).astype(np.float32)
    b = np.zeros(C, np.float32)
    V = (rng.uniform(-1, 1, (C, d, 256, K_FM)) * math.sqrt(6.0 / (256 * K_FM))).astype(np.float32)

    params = [W, b, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    # Precompute offsets for linear part
    offs = np.cumsum([0] + Ds)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z = Z[idx]                     # (B, K)
            xb = X[idx]                    # (B, d)
            yy = y[idx]                    # (B,)

            # Forward
            logits = z @ W + b             # (B, C)
            # FM part
            for c in range(C):
                S = np.zeros((len(idx), K_FM), dtype=np.float32)
                S2 = np.zeros((len(idx), K_FM), dtype=np.float32)
                for j in range(d):
                    v = V[c, j, xb[:, j]]  # (B, k)
                    S += v
                    S2 += v * v
                fm = 0.5 * np.sum(S * S - S2, axis=1)
                logits[:, c] += fm

            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad_logits = (probs / len(yy)).astype(np.float32)  # (B, C)

            # Linear gradients
            grad_W = z.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
            grad_z = grad_logits @ W.T  # (B, K) for FM backprop through linear? No, FM doesn't use Z.

            # FM gradients
            grad_V = np.zeros_like(V)
            for c in range(C):
                g = grad_logits[:, c]  # (B,)
                # Compute S for this class
                S = np.zeros((len(idx), K_FM), dtype=np.float32)
                for j in range(d):
                    S += V[c, j, xb[:, j]]
                for j in range(d):
                    v = V[c, j, xb[:, j]]  # (B, k)
                    # dFM/dv = S - v
                    dv = (S - v) * g[:, None]  # (B, k)
                    # Accumulate per value
                    np.add.at(grad_V[c, j], xb[:, j], dv)

            grad_V += L2 * V

            grads = [grad_W, grad_b, grad_V]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export weights
    # Linear lookup tables T[j][val] = E_j[val] @ W  (C,)
    T = np.zeros((d, 256, C), dtype=np.float32)
    for j in range(d):
        T[j] = enc[j] @ W[offs[j]:offs[j+1]]
    # Flatten order: T (d*256*C), b (C), V (C*d*256*k)
    magic = 0x464D3131
    out = struct.pack("<4i", magic, d, K_FM, C)
    out += T.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine (degree 2) + linear for label tasks.
 * ./infer weights.bin in.rec N
 * Input record: d bytes. Output: 1 byte class index.
 * Weights format: magic 0x464D3131, int32 d, k, C;
 *   float32 T[d][256][C] (linear), b[C];
 *   float32 V[C][d][256][k] (FM embeddings). */
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
    if (nw < 16 || hd[0] != 0x464D3131) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C < 1 || C > 256 || k < 1 || k > 32) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * C + C + (long)C * d * 256 * k);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *b = T + (long)d * 256 * C;
    const float *V = b + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(sizeof(float) * C);
    float *S = malloc(sizeof(float) * C * k);
    float *S2 = malloc(sizeof(float) * C * k);

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        /* Linear part: sum_j T[j][xi[j]] + b */
        for (int c = 0; c < C; c++) logits[c] = b[c];
        for (int j = 0; j < d; j++) {
            const float *tj = T + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) logits[c] += tj[c];
        }

        /* FM part: for each class, 0.5 * sum_f (S_cf^2 - S2_cf) */
        for (int c = 0; c < C; c++) {
            float *Sc = S + c * k;
            float *S2c = S2 + c * k;
            for (int f = 0; f < k; f++) { Sc[f] = 0.0f; S2c[f] = 0.0f; }
            for (int j = 0; j < d; j++) {
                const float *v = V + ((long)c * d * 256 * k + (long)j * 256 * k + xi[j] * k);
                for (int f = 0; f < k; f++) {
                    float val = v[f];
                    Sc[f] += val;
                    S2c[f] += val * val;
                }
            }
            float fm = 0.0f;
            for (int f = 0; f < k; f++) fm += Sc[f] * Sc[f] - S2c[f];
            logits[c] += 0.5f * fm;
        }

        /* Argmax */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```