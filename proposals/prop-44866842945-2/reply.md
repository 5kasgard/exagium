```hypothesis
Family: label-hashed-bigram. Adds 4 hashed bigram features (last 4 adjacent byte pairs, 2048 buckets each) to the linear unigram baseline. Text prediction relies heavily on local n-gram statistics; hashing captures pair interactions without blowing up parameters (adds ~2.1 MB). Same Adam/L2/epoch schedule as label-linear. If dev accuracy ≤ 0.3375, the hashed bigrams either collide too destructively or the task needs longer-range context.
```
```python
"""Hashed bigram + unigram linear model for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Unigrams: per-position one-hot over training values (like label-linear).
Bigrams: last 4 adjacent pairs (positions 11-12,12-13,13-14,14-15), hashed to 2048 buckets.
Linear head (C=65) trained with Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100,ceil(600000/n)).
Exports: magic 0x48424731, d_uni, d_bi, nbuckets, C; float32 T_uni[d_uni][256][C], T_bi[d_bi][nbuckets][C], bias[C]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
D_UNI = 16
D_BI = 4
NBUCKETS = 2048
BI_POS = [(11,12), (12,13), (13,14), (14,15)]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    assert card["input"]["bytes"] == D_UNI
    C = card["classes"]
    assert C == 65

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, D_UNI)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Unigram encoders (per column unique values)
    enc_uni = []
    uni_sizes = []
    for j in range(D_UNI):
        vals = np.unique(X[:, j])
        V = len(vals)
        E = np.zeros((256, V), np.float32)
        E[vals, np.arange(V)] = 1.0
        enc_uni.append(E)
        uni_sizes.append(V)
    K_uni = sum(uni_sizes)

    # Weights
    W_uni = np.zeros((K_uni, C), np.float32)
    W_bi = np.zeros((D_BI * NBUCKETS, C), np.float32)
    bias = np.zeros(C, np.float32)

    # Adam state
    m_uni, v_uni = np.zeros_like(W_uni), np.zeros_like(W_uni)
    m_bi, v_bi = np.zeros_like(W_bi), np.zeros_like(W_bi)
    m_b, v_b = np.zeros_like(bias), np.zeros_like(bias)

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb = X[idx]
            yb = y[idx]
            B = len(yb)

            logits = np.tile(bias, (B, 1))

            # Unigram
            off = 0
            for j in range(D_UNI):
                V = uni_sizes[j]
                feat = enc_uni[j][Xb[:, j]]
                logits += feat @ W_uni[off:off+V]
                off += V

            # Bigram
            for bi, (p1, p2) in enumerate(BI_POS):
                h = (Xb[:, p1].astype(np.uint32) * 257 + Xb[:, p2].astype(np.uint32)) % NBUCKETS
                logits += W_bi[bi * NBUCKETS + h]

            # Softmax gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad = probs / B

            # Gradients
            g_bias = grad.sum(axis=0) + L2 * bias

            g_uni = np.zeros_like(W_uni)
            off = 0
            for j in range(D_UNI):
                V = uni_sizes[j]
                feat = enc_uni[j][Xb[:, j]]
                g_uni[off:off+V] = feat.T @ grad + L2 * W_uni[off:off+V]
                off += V

            g_bi = np.zeros_like(W_bi)
            for bi, (p1, p2) in enumerate(BI_POS):
                h = (Xb[:, p1].astype(np.uint32) * 257 + Xb[:, p2].astype(np.uint32)) % NBUCKETS
                np.add.at(g_bi, bi * NBUCKETS + h, grad)
            g_bi += L2 * W_bi

            # Adam update
            t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8

            m_uni = beta1 * m_uni + (1 - beta1) * g_uni
            v_uni = beta2 * v_uni + (1 - beta2) * g_uni * g_uni
            W_uni -= LR * (m_uni / (1 - beta1 ** t)) / (np.sqrt(v_uni / (1 - beta2 ** t)) + eps)

            m_bi = beta1 * m_bi + (1 - beta1) * g_bi
            v_bi = beta2 * v_bi + (1 - beta2) * g_bi * g_bi
            W_bi -= LR * (m_bi / (1 - beta1 ** t)) / (np.sqrt(v_bi / (1 - beta2 ** t)) + eps)

            m_b = beta1 * m_b + (1 - beta1) * g_bias
            v_b = beta2 * v_b + (1 - beta2) * g_bias * g_bias
            bias -= LR * (m_b / (1 - beta1 ** t)) / (np.sqrt(v_b / (1 - beta2 ** t)) + eps)

    # Export baked tables
    T_uni = []
    off = 0
    for j in range(D_UNI):
        V = uni_sizes[j]
        T_uni.append(enc_uni[j] @ W_uni[off:off+V])
        off += V
    T_uni = np.stack(T_uni).astype(np.float32)
    T_bi = W_bi.reshape(D_BI, NBUCKETS, C).astype(np.float32)
    bias = bias.astype(np.float32)

    out = struct.pack("<5i", 0x48424731, D_UNI, D_BI, NBUCKETS, C)
    out += T_uni.tobytes()
    out += T_bi.tobytes()
    out += bias.tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Hashed bigram + unigram linear model: ./infer weights.bin in.rec N
 * Input: 16 bytes. Output: class index (1 byte).
 * Weights: magic 0x48424731, d_uni, d_bi, nbuckets, C;
 * float32 T_uni[d_uni][256][C], T_bi[d_bi][nbuckets][C], bias[C]. */
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
    if (nw < 20 || hd[0] != 0x48424731) { fprintf(stderr, "bad weights\n"); return 3; }
    int d_uni = hd[1], d_bi = hd[2], nbuckets = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d_uni != 16 || d_bi != 4 || nbuckets != 2048) {
        fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected_floats = (long)d_uni * 256 * C + (long)d_bi * nbuckets * C + C;
    if (nw != 20 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T_uni = (const float *)(hd + 5);
    const float *T_bi = T_uni + (long)d_uni * 256 * C;
    const float *bias = T_bi + (long)d_bi * nbuckets * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sc = malloc(sizeof(float) * C);
    const int bi_pos[4][2] = {{11,12}, {12,13}, {13,14}, {14,15}};

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }
        memcpy(sc, bias, sizeof(float) * C);
        for (int j = 0; j < d_uni; j++) {
            const float *t = T_uni + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) sc[c] += t[c];
        }
        for (int bi = 0; bi < d_bi; bi++) {
            unsigned char b1 = xi[bi_pos[bi][0]];
            unsigned char b2 = xi[bi_pos[bi][1]];
            unsigned h = (b1 * 257 + b2) % nbuckets;
            const float *t = T_bi + ((long)bi * nbuckets + h) * C;
            for (int c = 0; c < C; c++) sc[c] += t[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```