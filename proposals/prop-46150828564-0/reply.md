```hypothesis
Family: label-fm (degree-2 Factorization Machine). Why: Adult has strong feature interactions (education×occupation, marital-status×relationship, capital-gain×hours-per-week). FM models all pairwise interactions via low-rank factors (k=8), adding only ~32K params/class vs linear's ~3.5K, so it should beat label-linear's 0.8524 accuracy while staying fast (~728 multiply-adds/item). Failure: dev accuracy ≤ 0.8524 or Ir/item ≥ 719 means interactions don't help or rank is too low.
```
```python
"""Factorization Machine (degree 2) for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
FM with k factors per feature value per class. Adam lr 1e-3, L2 1e-4, minibatch 64, epochs min(100, ceil(600000/n)).
Exported: int32 magic 0x464D3231, d, k, C; float32 w0[C], w[C][d][256], v[C][d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-3, 1e-4, 64, 600000, 100


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
    d = X.shape[1]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # params per class: w0[C], w[C,d,256], v[C,d,256,K]
    w0 = np.zeros(C, np.float32)
    w = np.zeros((C, d, 256), np.float32)
    v = (rng.normal(0, 0.01, (C, d, 256, K))).astype(np.float32)

    params = [w0, w, v]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # forward: compute scores for each class
            scores = np.zeros((bsz, C), np.float32)
            for c in range(C):
                sc = scores[:, c]
                sc[:] = w0[c]
                # linear part
                for i in range(d):
                    sc += w[c, i, xb[:, i]]
                # pairwise interactions
                for i in range(d):
                    vi = v[c, i, xb[:, i]]  # (bsz, K)
                    for j in range(i + 1, d):
                        vj = v[c, j, xb[:, j]]
                        sc += (vi * vj).sum(axis=1)

            # softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yb] -= 1.0
            grads = probs / bsz  # (bsz, C)

            # gradients
            gw0 = np.zeros(C, np.float32)
            gw = np.zeros((C, d, 256), np.float32)
            gv = np.zeros((C, d, 256, K), np.float32)

            for c in range(C):
                g = grads[:, c]  # (bsz,)
                gw0[c] = g.sum()
                # linear grads
                for i in range(d):
                    np.add.at(gw[c, i], xb[:, i], g)
                # interaction grads
                for i in range(d):
                    vi = v[c, i, xb[:, i]]  # (bsz, K)
                    for j in range(i + 1, d):
                        vj = v[c, j, xb[:, j]]
                        g_exp = g[:, None]  # (bsz, 1)
                        np.add.at(gv[c, i], xb[:, i], g_exp * vj)
                        np.add.at(gv[c, j], xb[:, j], g_exp * vi)

            # L2 regularization
            gw += L2 * w
            gv += L2 * v

            # Adam update
            all_grads = [gw0, gw, gv]
            t += 1
            for p, gr, (m, v_adam) in zip(params, all_grads, state):
                m *= 0.9; m += 0.1 * gr
                v_adam *= 0.999; v_adam += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v_adam / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # pack weights: magic, d, K, C; then w0[C], w[C,d,256], v[C,d,256,K] all float32 LE
    magic = 0x464D3231  # "12MF"
    header = struct.pack("<4i", magic, d, K, C)
    body = w0.astype("<f4").tobytes() + w.astype("<f4").tobytes() + v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine (degree 2) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). */
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
    if (nw < 16 || hd[0] != 0x464D3231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d < 1 || d > 256 || K < 1 || K > 64 || C < 1 || C > 256) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected_floats = 1L * C * (1 + d * 256 * (1 + K));
    if (nw != 16 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *w0 = p; p += C;
    const float *w = p; p += (long)C * d * 256;
    const float *v = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            float s = w0[c];
            // linear
            for (int j = 0; j < d; j++) s += w[((long)c * d + j) * 256 + xi[j]];
            // pairwise
            for (int j = 0; j < d; j++) {
                const float *vj = v + (((long)c * d + j) * 256 + xi[j]) * K;
                for (int l = j + 1; l < d; l++) {
                    const float *vl = v + (((long)c * d + l) * 256 + xi[l]) * K;
                    for (int f = 0; f < K; f++) s += vj[f] * vl[f];
                }
            }
            scores[c] = s;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```