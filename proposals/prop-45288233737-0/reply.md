```hypothesis
Family: label-fm (factorization machine with 8-dim embeddings).
Why: Adult has strong feature interactions (e.g., education×occupation, marital-status×relationship) that linear models miss. FM captures all pairwise interactions via low-rank embeddings, adding only ~28k parameters. Inference cost is O(d·k) ≈ 112 multiply-adds per item, far below an MLP.
Failure signal: dev accuracy ≤ 0.855 (does not beat label-linear's 0.8524 by a meaningful margin).
```
```python
"""label-fm: Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Weight format (little-endian):
  int32 magic=0x464D3132, int32 d, int32 k, int32 C=2
  float32 bias
  float32 linear[d*256]
  float32 embed[d*256*k]"""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
SEED = 1
K = 8                # embedding dimension
LR = 0.01            # Adam learning rate
L2 = 1e-4            # L2 regularization
BATCH = 64
EPOCHS = 50
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8

def read_records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2

    X_bytes = b"".join(read_records(sys.argv[2]))
    y_bytes = b"".join(read_records(sys.argv[3]))
    n = len(X_bytes) // d
    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(n, d)
    y = np.frombuffer(y_bytes, dtype=np.uint8).astype(np.float32)  # 0 or 1

    # Split: last 20% for validation (tuning), but we hard-code hyperparams.
    # We'll train on all data for final weights.
    rng = np.random.default_rng(SEED)

    # Parameters
    bias = np.float32(0.0)
    linear = np.zeros((d, 256), dtype=np.float32)
    embed = rng.normal(0, 0.01, size=(d, 256, K)).astype(np.float32)

    # Adam state
    m_bias, v_bias = 0.0, 0.0
    m_lin = np.zeros_like(linear)
    v_lin = np.zeros_like(linear)
    m_emb = np.zeros_like(embed)
    v_emb = np.zeros_like(embed)
    t = 0

    indices = np.arange(n)
    for epoch in range(EPOCHS):
        rng.shuffle(indices)
        for start in range(0, n, BATCH):
            idx = indices[start:start+BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(xb)

            # Forward
            # Linear term: sum_j linear[j, xb[:,j]]
            lin_term = np.zeros(B, dtype=np.float32)
            for j in range(d):
                lin_term += linear[j, xb[:, j]]

            # Embedding sums
            emb_sum = np.zeros((B, K), dtype=np.float32)
            emb_sq_sum = np.zeros((B, K), dtype=np.float32)
            for j in range(d):
                e = embed[j, xb[:, j], :]      # (B, K)
                emb_sum += e
                emb_sq_sum += e * e

            inter = 0.5 * np.sum(emb_sum * emb_sum - emb_sq_sum, axis=1)
            logit = bias + lin_term + inter
            p = 1.0 / (1.0 + np.exp(-logit))
            grad = p - yb                      # (B,)

            # Backward
            g_bias = np.mean(grad)
            # Linear gradients
            g_lin = np.zeros_like(linear)
            for j in range(d):
                np.add.at(g_lin[j], xb[:, j], grad)
            g_lin /= B
            g_lin += L2 * linear

            # Embedding gradients
            g_emb = np.zeros_like(embed)
            for j in range(d):
                e = embed[j, xb[:, j], :]          # (B, K)
                # grad[:,None] * (emb_sum - e)
                delta = grad[:, None] * (emb_sum - e) / B
                np.add.at(g_emb[j], xb[:, j], delta)
            g_emb += L2 * embed

            # Adam update
            t += 1
            # bias
            m_bias = BETA1 * m_bias + (1 - BETA1) * g_bias
            v_bias = BETA2 * v_bias + (1 - BETA2) * g_bias * g_bias
            m_hat = m_bias / (1 - BETA1 ** t)
            v_hat = v_bias / (1 - BETA2 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + EPS)

            # linear
            m_lin = BETA1 * m_lin + (1 - BETA1) * g_lin
            v_lin = BETA2 * v_lin + (1 - BETA2) * g_lin * g_lin
            m_hat = m_lin / (1 - BETA1 ** t)
            v_hat = v_lin / (1 - BETA2 ** t)
            linear -= LR * m_hat / (np.sqrt(v_hat) + EPS)

            # embed
            m_emb = BETA1 * m_emb + (1 - BETA1) * g_emb
            v_emb = BETA2 * v_emb + (1 - BETA2) * g_emb * g_emb
            m_hat = m_emb / (1 - BETA1 ** t)
            v_hat = v_emb / (1 - BETA2 ** t)
            embed -= LR * m_hat / (np.sqrt(v_hat) + EPS)

    # Write weights
    magic = 0x464D3132
    out = struct.pack("<4i", magic, d, K, C)
    out += struct.pack("<f", bias)
    out += linear.astype("<f4").tobytes()
    out += embed.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class (0 or 1).
 * Weight format (little-endian):
 *   int32 magic=0x464D3132, int32 d, int32 k, int32 C=2
 *   float32 bias
 *   float32 linear[d*256]
 *   float32 embed[d*256*k] */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static unsigned char *slurp(const char *path, long *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(3); }
    fseek(f, 0, SEEK_END);
    *len = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(*len ? *len : 1);
    if (fread(buf, 1, *len, f) != (size_t)*len) {
        fprintf(stderr, "short read %s\n", path);
        exit(3);
    }
    fclose(f);
    return buf;
}

static const unsigned char *next_rec(const unsigned char *buf, long n, long *pos, uint32_t *len) {
    if (*pos + 4 > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    uint32_t k = (uint32_t)buf[*pos] |
                 (uint32_t)buf[*pos+1] << 8 |
                 (uint32_t)buf[*pos+2] << 16 |
                 (uint32_t)buf[*pos+3] << 24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *rec = buf + *pos + 4;
    *pos += 4 + k;
    return rec;
}

static void put_rec(FILE *f, const unsigned char *rec, uint32_t k) {
    unsigned char hdr[4] = {k & 255, (k>>8)&255, (k>>16)&255, (k>>24)&255};
    fwrite(hdr, 1, 4, f);
    fwrite(rec, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: infer weights.bin in.rec N\n");
        return 2;
    }
    long nw, nx = 0;
    int32_t *hdr = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hdr[0] != 0x464D3132) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hdr[1], K = hdr[2], C = hdr[3];
    if (C != 2 || d <= 0 || K <= 0) { fprintf(stderr, "bad header dims\n"); return 3; }

    long expected = 16 + 4L + 4L*d*256 + 4L*d*256*K;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hdr + 4);
    const float *linear = bias + 1;
    const float *embed = linear + (long)d * 256;

    unsigned char *x = NULL;
    long N = atol(argv[3]);
    if (N > 0) x = slurp(argv[2], &nx);
    long pos = 0;

    float *emb_sum = malloc(K * sizeof(float));
    float *emb_sq = malloc(K * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k;
        const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld has %u bytes, expected %d\n", i, k, d); return 3; }

        float logit = bias[0];
        // linear term
        for (int j = 0; j < d; j++) {
            logit += linear[(long)j * 256 + xi[j]];
        }
        // embedding sums
        memset(emb_sum, 0, K * sizeof(float));
        memset(emb_sq, 0, K * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *e = embed + ((long)j * 256 + xi[j]) * K;
            for (int f = 0; f < K; f++) {
                float v = e[f];
                emb_sum[f] += v;
                emb_sq[f] += v * v;
            }
        }
        for (int f = 0; f < K; f++) {
            logit += 0.5f * (emb_sum[f] * emb_sum[f] - emb_sq[f]);
        }

        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }

    free(emb_sum);
    free(emb_sq);
    return 0;
}
```