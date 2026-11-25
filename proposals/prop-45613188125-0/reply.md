```hypothesis
Family: label-fm (factorization machine with 8 latent dimensions).
Why: FMs model all pairwise feature interactions in O(k·d) time/space, which is crucial for tabular data like Adult where interactions (e.g., education×occupation) are predictive. With k=8 the model stays small and fast while adding expressive power beyond linear.
Failure: If dev accuracy ≤ 0.8524 (label-linear) or inference cost > 2× label-linear.
```
```python
"""label-fm: Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only.
Encoding: ordinal -> z-scored, categorical -> one-hot over training values.
FM: per-class bias, linear weights, and k=8 latent vectors.
Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Exports per-byte lookup tables for linear, vector, and squared-vector contributions.
Weights format: magic=0x464D4C31, d, k, C; then per-class: bias, T_lin[d][256], T_vec[d][256][k], T_vec2[d][256][k] (all float32)."""
import json, math, struct, sys

import numpy as np

K = 8
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
    d_orig = card["input"]["bytes"]
    types = card["input"]["types"]

    rng = np.random.default_rng(SEED)

    # Build per-column encoding matrices enc[j] of shape (256, C_j)
    enc = []
    for j, t in enumerate(types):
        col = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = col.mean(), col.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])  # (256, 1)
        else:
            vals = np.unique(col)
            E = np.zeros((256, len(vals)))
            E[vals.astype(int), np.arange(len(vals))] = 1.0
            enc.append(E)
    # Offsets for each column in the full feature vector
    offs = [0]
    for E in enc:
        offs.append(offs[-1] + E.shape[1])
    D = offs[-1]  # total feature dimension

    # Build full feature matrix Z (n x D)
    n = X.shape[0]
    Z = np.empty((n, D), dtype=np.float32)
    for j, E in enumerate(enc):
        Z[:, offs[j]:offs[j+1]] = E[X[:, j]].astype(np.float32)

    # Initialize per-class parameters
    params = []  # list of (w, V, b) for each class
    for _ in range(C):
        w = (rng.uniform(-1, 1, D) * math.sqrt(6.0 / D)).astype(np.float32)
        V = (rng.uniform(-1, 1, (D, K)) * math.sqrt(6.0 / (D + K))).astype(np.float32)
        b = np.array(0.0, dtype=np.float32)
        params.append([w, V, b])

    # Adam state
    state = []
    for p_list in params:
        s = []
        for p in p_list:
            s.append((np.zeros_like(p), np.zeros_like(p)))
        state.append(s)

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z = Z[idx]          # (B, D)
            yy = y[idx]         # (B,)
            B = z.shape[0]

            # Forward for all classes
            scores = np.zeros((B, C), dtype=np.float32)
            for c in range(C):
                w, V, b = params[c]
                linear = z @ w + b
                ZV = z @ V                      # (B, K)
                Z2 = z * z
                V2 = V * V
                inter = 0.5 * ((ZV * ZV).sum(axis=1) - (Z2 @ V2).sum(axis=1))
                scores[:, c] = linear + inter

            # Softmax cross-entropy gradient
            max_s = scores.max(axis=1, keepdims=True)
            exp_s = np.exp(scores - max_s)
            probs = exp_s / exp_s.sum(axis=1, keepdims=True)
            probs[np.arange(B), yy] -= 1.0
            grads_score = (probs / B).astype(np.float32)   # (B, C)

            # Backward per class
            for c in range(C):
                w, V, b = params[c]
                g = grads_score[:, c]                     # (B,)
                # grad b
                g_b = g.sum()
                # grad w
                g_w = z.T @ g + L2 * w
                # grad V
                ZV = z @ V
                G = g[:, None] * ZV                       # (B, K)
                g_V1 = z.T @ G                            # (D, K)
                Z2 = z * z
                H = (Z2.T @ g)[:, None] * V               # (D, K)
                g_V = g_V1 - H + L2 * V

                # Adam update
                p_list = [w, V, b]
                g_list = [g_w, g_V, g_b]
                for i, (p, gr) in enumerate(zip(p_list, g_list)):
                    m, v = state[c][i]
                    m *= 0.9; m += 0.1 * gr
                    v *= 0.999; v += 0.001 * gr * gr
                    t += 1
                    m_hat = m / (1 - 0.9 ** t)
                    v_hat = v / (1 - 0.999 ** t)
                    p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Build lookup tables per class
    T_all = []  # list per class of (bias, T_lin, T_vec, T_vec2)
    for c in range(C):
        w, V, b = params[c]
        T_lin = np.zeros((d_orig, 256), dtype=np.float32)
        T_vec = np.zeros((d_orig, 256, K), dtype=np.float32)
        T_vec2 = np.zeros((d_orig, 256, K), dtype=np.float32)
        for j in range(d_orig):
            E = enc[j]
            off, nxt = offs[j], offs[j+1]
            w_slice = w[off:nxt]
            V_slice = V[off:nxt]
            V2_slice = V_slice * V_slice
            T_lin[j] = E @ w_slice
            T_vec[j] = E @ V_slice
            T_vec2[j] = (E * E) @ V2_slice
        T_all.append((b, T_lin, T_vec, T_vec2))

    # Write weights.bin
    magic = 0x464D4C31  # 'FML1'
    header = struct.pack("<4i", magic, d_orig, K, C)
    chunks = [header]
    for c in range(C):
        bias, T_lin, T_vec, T_vec2 = T_all[c]
        chunks.append(struct.pack("<f", bias))
        chunks.append(T_lin.astype("<f4").tobytes())
        chunks.append(T_vec.astype("<f4").tobytes())
        chunks.append(T_vec2.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(b"".join(chunks))


if __name__ == "__main__":
    main()
```
```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: d bytes (original columns). Output: 1 byte class index.
 * Weights format: magic=0x464D4C31, d, k, C; per class: bias, T_lin[d][256], T_vec[d][256][k], T_vec2[d][256][k] (float32). */
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
    uint8_t *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "weights too small\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D4C31) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d <= 0 || K <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }

    // Expected size: header(16) + C * (1 + d*256*(1 + 2*K)) floats * 4
    long expected = 16 + 4L * C * (1L + (long)d * 256 * (1L + 2L * K));
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    uint8_t *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(C * sizeof(float));
    float *acc_vec = malloc(C * K * sizeof(float));
    float *acc_vec2 = malloc(C * K * sizeof(float));

    const float *p = (const float *)(wb + 16);
    // We'll keep pointers to each class's tables
    const float **bias = malloc(C * sizeof(float*));
    const float ***T_lin = malloc(C * sizeof(float**));
    const float ****T_vec = malloc(C * sizeof(float***));
    const float ****T_vec2 = malloc(C * sizeof(float***));

    for (int c = 0; c < C; c++) {
        bias[c] = p; p += 1;
        T_lin[c] = (const float**)malloc(d * sizeof(float*));
        T_vec[c] = (const float***)malloc(d * sizeof(float**));
        T_vec2[c] = (const float***)malloc(d * sizeof(float**));
        for (int j = 0; j < d; j++) {
            T_lin[c][j] = p; p += 256;
            T_vec[c][j] = (const float**)malloc(256 * sizeof(float*));
            T_vec2[c][j] = (const float**)malloc(256 * sizeof(float*));
            for (int b = 0; b < 256; b++) {
                T_vec[c][j][b] = p; p += K;
                T_vec2[c][j][b] = p; p += K;
            }
        }
    }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            scores[c] = bias[c][0];
            for (int u = 0; u < K; u++) {
                acc_vec[c*K + u] = 0.0f;
                acc_vec2[c*K + u] = 0.0f;
            }
        }

        for (int j = 0; j < d; j++) {
            unsigned char byte = xi[j];
            for (int c = 0; c < C; c++) {
                scores[c] += T_lin[c][j][byte];
                for (int u = 0; u < K; u++) {
                    acc_vec[c*K + u] += T_vec[c][j][byte][u];
                    acc_vec2[c*K + u] += T_vec2[c][j][byte][u];
                }
            }
        }

        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int u = 0; u < K; u++) {
                float v = acc_vec[c*K + u];
                inter += v * v - acc_vec2[c*K + u];
            }
            scores[c] += 0.5f * inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = c;
        put_rec(stdout, &best, 1);
    }

    // Free allocated pointer arrays (not the weight blob)
    for (int c = 0; c < C; c++) {
        for (int j = 0; j < d; j++) {
            free((void*)T_vec[c][j]);
            free((void*)T_vec2[c][j]);
        }
        free((void*)T_lin[c]);
        free((void*)T_vec[c]);
        free((void*)T_vec2[c]);
    }
    free(bias); free(T_lin); free(T_vec); free(T_vec2);
    free(scores); free(acc_vec); free(acc_vec2);
    free(wb); free(x);
    return 0;
}
```