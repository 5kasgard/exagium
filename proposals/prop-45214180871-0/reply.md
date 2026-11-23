```hypothesis
Family: label-fm (Factorization Machine with shallow interaction layer).
Why: Adult has mixed ordinal/categorical features where pairwise interactions (e.g., education × occupation, capital-gain × hours) matter. A factorization machine captures these with O(d·k) parameters (k=8) instead of O(d²), adding minimal CPU cost over linear while modeling non-additive effects. The shared k-dim interaction vector projects to logits via a tiny 2nd layer.
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or inference cost > 2× label-linear, the FM overhead isn't justified.
```
```python
"""Factorization Machine for tabular label tasks: ./train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy. Encodes each of d fields as 256-value categorical.
FM: logits = w0 + Σ_j w[j, x_j] + (h @ W2 + b2), where h_f = ½((Σ_j V[j,x_j,f])² - Σ_j V[j,x_j,f]²).
Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)). k=8 factors."""
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
    d, C = X.shape[1], card["classes"]
    rng = np.random.default_rng(SEED)

    # params: w0[C], W[d,256,C], V[d,256,K], W2[K,C], b2[C]
    w0 = np.zeros(C, np.float32)
    W = (rng.uniform(-1, 1, (d, 256, C)) * math.sqrt(6.0 / (256 + C))).astype(np.float32)
    V = (rng.uniform(-1, 1, (d, 256, K)) * math.sqrt(6.0 / (256 + K))).astype(np.float32)
    W2 = (rng.uniform(-1, 1, (K, C)) * math.sqrt(6.0 / (K + C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [w0, W, V, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # linear part: (bsz, C)
            lin = w0 + np.sum(W[np.arange(d)[None, :], xb[:, :, None], :], axis=1)  # (bsz, C)

            # FM part: h[bsz, K]
            h = np.zeros((bsz, K), np.float32)
            for f in range(K):
                v = V[np.arange(d)[None, :], xb[:, :, None], f]  # (bsz, d)
                sum_v = v.sum(axis=1)
                sum_v2 = (v * v).sum(axis=1)
                h[:, f] = 0.5 * (sum_v * sum_v - sum_v2)

            logits = lin + h @ W2 + b2  # (bsz, C)

            # softmax + grad
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yb] -= 1.0
            grad_logits = (probs / bsz).astype(np.float32)  # (bsz, C)

            # gradients
            gw0 = grad_logits.sum(axis=0)  # (C,)
            gW = np.zeros_like(W)
            for j in range(d):
                np.add.at(gW[j], xb[:, j], grad_logits)  # (256, C)
            gV = np.zeros_like(V)
            for f in range(K):
                v = V[np.arange(d)[None, :], xb[:, :, None], f]  # (bsz, d)
                sum_v = v.sum(axis=1, keepdims=True)  # (bsz, 1)
                # dL/dV_{j,v,f} = Σ_c (dL/dh_f) * (sum_v - V_{j,v,f}) * 1_{x_j=v}
                # dL/dh_f = Σ_c grad_logits[:,c] * W2[f,c]
                dh_f = grad_logits @ W2[f]  # (bsz,)
                for j in range(d):
                    diff = (sum_v[:, 0] - v[:, j]) * dh_f  # (bsz,)
                    np.add.at(gV[j, :, f], xb[:, j], diff)
            gW2 = h.T @ grad_logits  # (K, C)
            gb2 = grad_logits.sum(axis=0)  # (C,)

            grads = [gw0, gW, gV, gW2, gb2]
            # L2 regularization
            grads[1] += L2 * W
            grads[2] += L2 * V
            grads[3] += L2 * W2

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # pack weights: magic, d, K, C; then w0, W, V, W2, b2 (all float32 little-endian)
    magic = 0x464D3130  # "FM10"
    out = struct.pack("<4i", magic, d, K, C)
    out += w0.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: 1 byte class (argmax).
 * Weights layout (little-endian): int32 magic(0x464D3130), d, K, C;
 * float32 w0[C], W[d][256][C], V[d][256][K], W2[K][C], b2[C]. */
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1 || K < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * (C + (long)d * 256 * C + (long)d * 256 * K + (long)K * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *w0 = p; p += C;
    const float *W = p; p += (long)d * 256 * C;
    const float *V = p; p += (long)d * 256 * K;
    const float *W2 = p; p += (long)K * C;
    const float *b2 = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = (float *)malloc(sizeof(float) * K);
    float *scores = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // linear part
        for (int c = 0; c < C; c++) scores[c] = w0[c];
        for (int j = 0; j < d; j++) {
            const float *wj = W + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += wj[c];
        }

        // FM interaction vector h
        for (int f = 0; f < K; f++) {
            float sum_v = 0.0f, sum_v2 = 0.0f;
            for (int j = 0; j < d; j++) {
                float v = V[((long)j * 256 + xi[j]) * K + f];
                sum_v += v;
                sum_v2 += v * v;
            }
            h[f] = 0.5f * (sum_v * sum_v - sum_v2);
        }

        // project h to logits
        for (int c = 0; c < C; c++) scores[c] += b2[c];
        for (int f = 0; f < K; f++) {
            const float *w2f = W2 + (long)f * C;
            for (int c = 0; c < C; c++) scores[c] += h[f] * w2f[c];
        }

        // argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(h); free(scores);
    return 0;
}
```