```hypothesis
Family: factorization-machine (FM) with per-class latent vectors (k=4). Captures pairwise pixel interactions crucial for digit shapes while staying tiny: ~3.3k params. Expects to beat linear (0.961) and approach mlp16 (0.977) at lower inference cost (no hidden layer, just 64*10*4 FM ops). Fails if dev accuracy < 0.965 or instructions/item > label-linear.
```
```python
"""Factorization Machine for label tasks with fixed-size ordinal bytes.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy. Encodes ordinal bytes via z-score.
Multiclass FM: score_c = b_c + w_c^T x + 0.5*(||V_c^T x||^2 - sum_j ||V_c[j]||^2 x_j^2).
Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)). Latent dim k=4.
Weights: magic 0x464D3130, d, k, C; mu[d], sd[d]; per class: bias, w[d], V[d*k] (all float32 LE)."""
import json, math, struct, sys

import numpy as np

K = 4
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

    # z-score encoding per feature
    mu = X.mean(axis=0).astype(np.float64)
    sd = X.std(axis=0).astype(np.float64)
    sd[sd == 0] = 1.0
    Z = ((X - mu) / sd).astype(np.float32)          # (n, d)
    Z_sq = Z * Z

    # parameters
    bias = np.zeros(C, np.float32)
    W = (rng.uniform(-1, 1, (C, d)) * math.sqrt(6.0 / d)).astype(np.float32)
    V = (rng.uniform(-1, 1, (C, d, K)) * math.sqrt(6.0 / d)).astype(np.float32)
    params = [bias, W, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = Z.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, zsq, yy = Z[idx], Z_sq[idx], y[idx]
            bsz = len(yy)

            # forward
            linear = z @ W.T + bias                          # (bsz, C)
            latent = np.einsum('bd,cdk->bck', z, V)          # (bsz, C, K)
            vnorm = zsq @ (V * V).sum(axis=2).T              # (bsz, C)
            scores = linear + 0.5 * ((latent * latent).sum(axis=2) - vnorm)

            # softmax + cross-entropy grad
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yy] -= 1.0
            dscores = (probs / bsz).astype(np.float32)       # (bsz, C)

            # gradients
            dbias = dscores.sum(axis=0)
            dW = dscores.T @ z + L2 * W
            dV = np.zeros_like(V)
            for c in range(C):
                dc = dscores[:, c]                           # (bsz,)
                lat_c = latent[:, c, :]                      # (bsz, K)
                term1 = (dc[:, None] * lat_c).T @ z          # (K, d)
                S = dc @ zsq                                 # (d,)
                term2 = V[c] * S[:, None]                    # (d, K)
                dV[c] = term1.T - term2 + L2 * V[c]

            grads = [dbias, dW, dV]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights
    magic = 0x464D3130
    out = struct.pack("<4i", magic, d, K, C)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    for c in range(C):
        out += struct.pack("<f", bias[c])
        out += W[c].astype("<f4").tobytes()
        out += V[c].astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d bytes per record (ordinal 0..255). Output: 1 byte class (argmax).
 * Weights: magic 0x464D3130, d, k, C; mu[d], sd[d]; per class: bias, w[d], V[d*k] (float32 LE). */
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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D3130) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d <= 0 || K <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }

    size_t expected = 16 + (size_t)d * 8 + (size_t)C * (4 + (size_t)d * 4 + (size_t)d * K * 4);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size: got %ld expect %zu\n", nw, expected); return 3; }

    float *mu = (float *)(wb + 16);
    float *sd = mu + d;
    float *cls = sd + d;                 // per class: bias, w[d], V[d*K]
    float *vnorm = malloc((size_t)C * d * sizeof(float));
    for (int c = 0; c < C; c++) {
        float *base = cls + c * (1 + d + d * K);
        float *V = base + 1 + d;
        for (int j = 0; j < d; j++) {
            float sum = 0.0f;
            for (int f = 0; f < K; f++) { float v = V[j * K + f]; sum += v * v; }
            vnorm[c * d + j] = sum;
        }
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *latent = malloc((size_t)K * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float best_score = -1e30f; unsigned char best = 0;
        for (int c = 0; c < C; c++) {
            float *base = cls + c * (1 + d + d * K);
            float bias = base[0];
            float *w = base + 1;
            float *V = base + 1 + d;

            float linear = bias;
            for (int f = 0; f < K; f++) latent[f] = 0.0f;
            float vnorm_acc = 0.0f;

            for (int j = 0; j < d; j++) {
                float e = (xi[j] - mu[j]) / sd[j];
                float e_sq = e * e;
                linear += w[j] * e;
                for (int f = 0; f < K; f++) latent[f] += V[j * K + f] * e;
                vnorm_acc += vnorm[c * d + j] * e_sq;
            }
            float latent_sq_sum = 0.0f;
            for (int f = 0; f < K; f++) latent_sq_sum += latent[f] * latent[f];
            float score = linear + 0.5f * (latent_sq_sum - vnorm_acc);
            if (score > best_score) { best_score = score; best = (unsigned char)c; }
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```