```hypothesis
Family: label-fm (Factorization Machine with per-feature latent vectors).
Why: Adult has 14 mixed ordinal/categorical features; pairwise interactions matter but full MLP is costly. FM captures all pairwise interactions in O(d·k) time (k=8) using the (sum^2 - sum of squares) trick, adding only ~200 FLOPs/item over linear. It should beat label-linear (0.8524) and approach label-mlp16 (0.8591) at near-linear latency.
Failure signal: dev accuracy ≤ 0.853 or inference cost > 1500 Ir/item.
```

```python
"""label-fm: Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Weight format (little-endian):
  uint32 magic=0x464D4C31 ('FML1'), int32 d, int32 k, int32 C
  for each class c in 0..C-1:
    float32 bias_c
    float32 linear_emb[d][256]
    float32 latent_emb[d][256][k]"""
import json, math, struct, sys

import numpy as np

SEED = 1
K = 8
LR = 1e-2
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100


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
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # params per class: bias (C), linear_emb (C, d, 256), latent_emb (C, d, 256, K)
    bias = np.zeros(C, np.float32)
    lin = (rng.normal(0, 0.01, (C, d, 256))).astype(np.float32)
    lat = (rng.normal(0, 0.01, (C, d, 256, K))).astype(np.float32)
    params = [bias, lin, lat]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(batch_idx):
        """Return scores (batch, C)"""
        xb = X[batch_idx]  # (b, d)
        bsz = xb.shape[0]
        # linear part: bias + sum_j lin[c, j, xb[:,j]]
        scores = bias[None, :]  # (1, C)
        for j in range(d):
            scores += lin[:, j, xb[:, j]].T  # (bsz, C)
        # FM interaction part per class
        for c in range(C):
            # accumulate latent vectors across features
            sum_lat = np.zeros((bsz, K), np.float32)
            sum_sq = np.zeros((bsz, K), np.float32)
            for j in range(d):
                v = lat[c, j, xb[:, j]]  # (bsz, K)
                sum_lat += v
                sum_sq += v * v
            inter = 0.5 * np.sum(sum_lat * sum_lat - sum_sq, axis=1)  # (bsz,)
            scores[:, c] += inter
        return scores

    def softmax_cross_entropy_grad(scores, yy):
        # scores (bsz, C), yy (bsz,)
        max_s = scores.max(axis=1, keepdims=True)
        exp_s = np.exp(scores - max_s)
        probs = exp_s / exp_s.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return probs / len(yy)  # (bsz, C)

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            yy = y[idx]
            scores = forward(idx)
            g = softmax_cross_entropy_grad(scores, yy)  # (bsz, C)

            # gradients
            g_bias = g.sum(axis=0) + L2 * bias
            g_lin = np.zeros_like(lin)
            g_lat = np.zeros_like(lat)
            xb = X[idx]
            bsz = len(idx)
            for c in range(C):
                gc = g[:, c]  # (bsz,)
                # linear grad
                for j in range(d):
                    np.add.at(g_lin[c, j], xb[:, j], gc)
                g_lin[c] += L2 * lin[c]
                # latent grad
                # compute sum_lat per sample
                sum_lat = np.zeros((bsz, K), np.float32)
                for j in range(d):
                    sum_lat += lat[c, j, xb[:, j]]
                for j in range(d):
                    v = lat[c, j, xb[:, j]]  # (bsz, K)
                    # derivative of 0.5*(sum^2 - sum_sq) w.r.t v = sum_lat - v
                    grad_v = (sum_lat - v) * gc[:, None]  # (bsz, K)
                    for k in range(K):
                        np.add.at(g_lat[c, j, :, k], xb[:, j], grad_v[:, k])
                g_lat[c] += L2 * lat[c]

            grads = [g_bias, g_lin, g_lat]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # pack weights
    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    for c in range(C):
        out += struct.pack("<f", bias[c])
        out += lin[c].astype("<f4").tobytes()
        out += lat[c].astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Each input record = d bytes (d=14). Output = 1 byte class index. */
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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wb[0] | wb[1]<<8 | wb[2]<<16 | wb[3]<<24;
    if (magic != 0x464D4C31) { fprintf(stderr, "bad magic\n"); return 3; }
    int32_t d = *(int32_t*)(wb+4);
    int32_t K = *(int32_t*)(wb+8);
    int32_t C = *(int32_t*)(wb+12);
    if (C < 1 || C > 256 || d != 14) { fprintf(stderr, "bad header\n"); return 3; }
    size_t expected = 16 + C * (4 + 4*d*256 + 4*d*256*K);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    const float *p = (const float*)(wb + 16);
    const float *bias = p; p += C;
    const float *lin = p; p += C * d * 256;
    const float *lat = p; p += C * d * 256 * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(C * sizeof(float));
    float *sum_lat = malloc(K * sizeof(float));
    float *sum_sq = malloc(K * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int c = 0; c < C; c++) {
            // linear part
            for (int j = 0; j < d; j++) {
                scores[c] += lin[(c*d + j)*256 + xi[j]];
            }
            // FM interaction
            for (int kk = 0; kk < K; kk++) { sum_lat[kk] = 0.0f; sum_sq[kk] = 0.0f; }
            for (int j = 0; j < d; j++) {
                const float *v = lat + ((c*d + j)*256 + xi[j]) * K;
                for (int kk = 0; kk < K; kk++) {
                    float val = v[kk];
                    sum_lat[kk] += val;
                    sum_sq[kk] += val * val;
                }
            }
            float inter = 0.0f;
            for (int kk = 0; kk < K; kk++) {
                float sl = sum_lat[kk];
                inter += 0.5f * (sl * sl - sum_sq[kk]);
            }
            scores[c] += inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```