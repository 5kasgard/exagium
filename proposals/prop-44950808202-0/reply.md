```hypothesis
Family: label-fm (Factorization Machine). Adult has mixed ordinal/categorical features; FM captures pairwise interactions with low-rank factors, often boosting tabular accuracy over linear models. With k=8 factors and 14 input bytes, the model stays tiny (≈0.5 MB weights) and inference is O(14·k) per item – similar to linear cost. If dev accuracy ≤ label-linear (0.8524) or Ir/item > 1000, the family does not justify itself.
```
```python
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
    rng = np.random.default_rng(SEED)

    # Build per-column encoding matrices (256 x dim_j)
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)

    # Concatenated design matrix Z (n x d)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, d = Z.shape

    # FM parameters per class: w0[C], W[C, d], V[C, d, K]
    scale = math.sqrt(6.0 / d)
    w0 = np.zeros(C, np.float32)
    W = (rng.uniform(-1, 1, (C, d)) * scale).astype(np.float32)
    V = (rng.uniform(-1, 1, (C, d, K)) * math.sqrt(6.0 / (d * K))).astype(np.float32)

    params = [w0, W, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            m = len(yy)

            # Forward
            linear = w0[None, :] + z @ W.T                       # (m, C)
            zv = z @ V.transpose(0, 2, 1)                        # (m, C, K)
            z2 = z ** 2
            v2 = V ** 2
            zv2 = z2 @ v2.transpose(0, 2, 1)                     # (m, C, K)
            inter = 0.5 * ((zv ** 2 - zv2).sum(axis=2))          # (m, C)
            logits = linear + inter                              # (m, C)

            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(m), yy] -= 1.0
            probs /= m                                           # (m, C)

            # Gradients
            gw0 = probs.sum(axis=0)                              # (C,)
            gW = probs.T @ z + L2 * W                            # (C, d)
            # V gradient: for each class c, factor k: sum_i probs_i,c * (zv_i,c,k * z_i - zv2_i,c,k * z_i)
            # Vectorized:
            #   gV_c,k = z.T @ (probs[:,c] * zv[:,c,k]) - z2.T @ (probs[:,c] * v2_c,k) + L2 * V_c,k
            gV = np.zeros_like(V)
            for c in range(C):
                p = probs[:, c][:, None]                         # (m, 1)
                gV[c] = (z.T @ (p * zv[:, c])) - (z2.T @ (p * v2[c])) + L2 * V[c]

            grads = [gw0, gW, gV]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m1, m2 = state[i]
                m1 *= 0.9; m1 += 0.1 * gr
                m2 *= 0.999; m2 += 0.001 * gr * gr
                p -= LR * (m1 / (1 - 0.9 ** t)) / (np.sqrt(m2 / (1 - 0.999 ** t)) + 1e-8)

    # Build lookup tables for inference: per class, per feature, per byte value
    # T_linear[C, d, 256], T_v[C, d, 256, K], T_v_sq[C, d, 256, K]
    T_linear = np.zeros((C, d, 256), np.float32)
    T_v = np.zeros((C, d, 256, K), np.float32)
    T_v_sq = np.zeros((C, d, 256, K), np.float32)

    off = 0
    for j, E in enumerate(enc):
        dim = E.shape[1]
        for c in range(C):
            w_slice = W[c, off:off + dim]            # (dim,)
            v_slice = V[c, off:off + dim]            # (dim, K)
            v_sq_slice = v_slice ** 2                # (dim, K)
            # For each byte value b, encoded vector e = E[b] (dim,)
            # linear contribution = e @ w_slice
            # v contribution = e @ v_slice
            # v_sq contribution = (e^2) @ v_sq_slice
            T_linear[c, j] = E @ w_slice
            T_v[c, j] = E @ v_slice
            T_v_sq[c, j] = (E ** 2) @ v_sq_slice
        off += dim

    # Pack weights.bin
    # Magic "FM12" = 0x32314D46
    magic = 0x32314D46
    header = struct.pack("<4i", magic, X.shape[1], K, C)
    body = w0.astype("<f4").tobytes()
    body += T_linear.astype("<f4").tobytes()
    body += T_v.astype("<f4").tobytes()
    body += T_v_sq.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```
```c
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
    if (nw < 16 || hd[0] != 0x32314D46) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 64) { fprintf(stderr, "bad params\n"); return 3; }
    size_t expected = 16 + 4L * C + 4L * C * d * 256 + 8L * C * d * 256 * K;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld vs %zu\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *w0 = p; p += C;
    const float *T_lin = p; p += (long)C * d * 256;
    const float *T_v = p; p += (long)C * d * 256 * K;
    const float *T_v_sq = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *lin = malloc(sizeof(float) * C);
    float *vsum = malloc(sizeof(float) * C * K);
    float *vsq = malloc(sizeof(float) * C * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            lin[c] = w0[c];
            for (int f = 0; f < K; f++) {
                vsum[c * K + f] = 0.0f;
                vsq[c * K + f] = 0.0f;
            }
        }

        for (int j = 0; j < d; j++) {
            unsigned char byte = xi[j];
            for (int c = 0; c < C; c++) {
                size_t base = ((long)c * d + j) * 256;
                lin[c] += T_lin[base + byte];
                const float *tv = T_v + (base * K) + byte * K;
                const float *tvs = T_v_sq + (base * K) + byte * K;
                for (int f = 0; f < K; f++) {
                    vsum[c * K + f] += tv[f];
                    vsq[c * K + f] += tvs[f];
                }
            }
        }

        float best_score = -1e30f; unsigned char best = 0;
        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int f = 0; f < K; f++) {
                float v = vsum[c * K + f];
                inter += v * v - vsq[c * K + f];
            }
            float score = lin[c] + 0.5f * inter;
            if (score > best_score) { best_score = score; best = (unsigned char)c; }
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```