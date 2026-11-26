```hypothesis
Family: label-fm (Factorization Machine). Adult has strong pairwise interactions (education×occupation, marital-status×sex) that linear models miss. FM with k=8 factors captures these in O(d·k) time, adding ~100k params (126KB weights) vs linear's 14k. If dev accuracy ≤0.8524, interactions don't help or k=8 overfits.
```
```python
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.03, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = b"".join(records(sys.argv[2]))
    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_bytes = b"".join(records(sys.argv[3]))
    y = np.frombuffer(y_bytes, dtype=np.uint8).astype(np.int64)
    y = 2 * y - 1  # {-1, 1}

    n, d = X.shape
    C = card["classes"]
    assert C == 2

    rng = np.random.default_rng(SEED)

    w0 = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    V = rng.normal(0, 0.01, size=(d, 256, K)).astype(np.float32)

    m_w0, v_w0 = 0.0, 0.0
    m_w = np.zeros_like(w)
    v_w = np.zeros_like(w)
    m_V = np.zeros_like(V)
    v_V = np.zeros_like(V)

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            bsz = len(yb)

            scores = np.full(bsz, w0, dtype=np.float32)
            sum_f = np.zeros((bsz, K), dtype=np.float32)
            sum_f2 = np.zeros((bsz, K), dtype=np.float32)

            for j in range(d):
                vals = xb[:, j]
                scores += w[j, vals]
                sum_f += V[j, vals]
                sum_f2 += V[j, vals] * V[j, vals]

            interaction = 0.5 * np.sum(sum_f * sum_f - sum_f2, axis=1)
            scores += interaction

            y_scores = yb * scores
            sig = 1.0 / (1.0 + np.exp(-y_scores))
            grad_scores = -yb * (1.0 - sig)

            grad_w0 = np.mean(grad_scores)
            grad_w = np.zeros_like(w)
            grad_V = np.zeros_like(V)

            for j in range(d):
                vals = xb[:, j]
                np.add.at(grad_w[j], vals, grad_scores)
                for f in range(K):
                    diff = sum_f[:, f] - V[j, vals, f]
                    np.add.at(grad_V[j, :, f], vals, grad_scores * diff)

            grad_w /= bsz
            grad_V /= bsz
            grad_w += L2 * w
            grad_V += L2 * V

            t += 1
            m_w0 = beta1 * m_w0 + (1 - beta1) * grad_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * grad_w0 * grad_w0
            w0 -= LR * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)

            m_w = beta1 * m_w + (1 - beta1) * grad_w
            v_w = beta2 * v_w + (1 - beta2) * grad_w * grad_w
            w -= LR * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)

            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * grad_V * grad_V
            V -= LR * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)

    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    out += np.array([w0], dtype="<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K < 1 || K > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }

    long expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *V = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_f = malloc(sizeof(float) * K);
    float *sum_f2 = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = w0;
        for (int f = 0; f < K; f++) { sum_f[f] = 0.0f; sum_f2[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            long idx = (long)j * 256 + v;
            score += w[idx];
            long v_idx = (long)j * 256 * K + v * K;
            for (int f = 0; f < K; f++) {
                float val = V[v_idx + f];
                sum_f[f] += val;
                sum_f2[f] += val * val;
            }
        }

        for (int f = 0; f < K; f++) {
            score += 0.5f * (sum_f[f] * sum_f[f] - sum_f2[f]);
        }

        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```