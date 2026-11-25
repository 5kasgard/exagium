```hypothesis
Family: label-embedlin (Embedding-based Factorized Linear). Shares a 256×D character embedding matrix across all 16 positions, with position-specific D×C projection matrices. This reduces parameters from 266K to ~10K (D=8), acting as strong regularization for the 24K training samples while capturing character similarities (vowels, punctuation clusters). At inference the factorized weights are expanded to a full 16×256×C lookup table, matching label-linear's memory access pattern and speed. Failure if dev accuracy ≤ 0.3375 (label-linear) or inference instructions/item exceed label-linear by >20%.
```
```python
import json, math, struct, sys

import numpy as np

D = 8
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
    d = card["input"]["bytes"]
    C = card["classes"]
    assert d == 16
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Factorized parameters: E[256, D], P[d, D, C], b[C]
    E = (rng.normal(0, 1 / math.sqrt(D), (256, D))).astype(np.float32)
    P = (rng.normal(0, 1 / math.sqrt(D), (d, D, C))).astype(np.float32)
    b = np.zeros(C, np.float32)

    # Adam state
    mE, vE = np.zeros_like(E), np.zeros_like(E)
    mP, vP = np.zeros_like(P), np.zeros_like(P)
    mb, vb = np.zeros_like(b), np.zeros_like(b)

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            B = len(yb)

            # Forward: logits = sum_j E[xb[:,j]] @ P[j] + b
            logits = np.zeros((B, C), np.float32)
            for j in range(d):
                logits += E[xb[:, j]] @ P[j]
            logits += b

            # Softmax gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = (probs / B).astype(np.float32)

            # Gradients
            gE = np.zeros_like(E)
            gP = np.zeros_like(P)
            gb = grad_logits.sum(axis=0)
            for j in range(d):
                Ej = E[xb[:, j]]                 # (B, D)
                gP[j] = Ej.T @ grad_logits       # (D, C)
                contrib = grad_logits @ P[j].T   # (B, D)
                np.add.at(gE, xb[:, j], contrib)

            # Add L2 regularization to gradients
            gE += L2 * E
            gP += L2 * P
            gb += L2 * b

            # Adam update
            t += 1
            # E
            mE *= 0.9; mE += 0.1 * gE
            vE *= 0.999; vE += 0.001 * gE * gE
            mE_hat = mE / (1 - 0.9 ** t)
            vE_hat = vE / (1 - 0.999 ** t)
            E -= LR * mE_hat / (np.sqrt(vE_hat) + 1e-8)
            # P
            mP *= 0.9; mP += 0.1 * gP
            vP *= 0.999; vP += 0.001 * gP * gP
            mP_hat = mP / (1 - 0.9 ** t)
            vP_hat = vP / (1 - 0.999 ** t)
            P -= LR * mP_hat / (np.sqrt(vP_hat) + 1e-8)
            # b
            mb *= 0.9; mb += 0.1 * gb
            vb *= 0.999; vb += 0.001 * gb * gb
            mb_hat = mb / (1 - 0.9 ** t)
            vb_hat = vb / (1 - 0.999 ** t)
            b -= LR * mb_hat / (np.sqrt(vb_hat) + 1e-8)

    # Export: magic=0x464C4531 ('FLE1'), d, D, C, E, P, b
    magic = 0x464C4531
    out = struct.pack("<4i", magic, d, D, C)
    out += E.astype("<f4").tobytes()
    out += P.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x464C4531) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16 || D < 1 || D > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (256L * D + (long)d * D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *E = (const float *)(hd + 4);
    const float *P = E + 256L * D;
    const float *b = P + (long)d * D * C;

    /* Expand factorized weights to full lookup table T[d][256][C] (matches label-linear layout). */
    float *T = malloc(sizeof(float) * (long)d * 256 * C);
    for (int j = 0; j < d; j++) {
        for (int c = 0; c < 256; c++) {
            const float *e = E + (long)c * D;
            const float *pj = P + (long)j * D * C;
            float *tjc = T + ((long)j * 256 + c) * C;
            for (int k = 0; k < C; k++) tjc[k] = 0.0f;
            for (int m = 0; m < D; m++) {
                float em = e[m];
                const float *pj_m = pj + (long)m * C;
                for (int k = 0; k < C; k++) tjc[k] += em * pj_m[k];
            }
        }
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) acc[c] = b[c];
        for (int j = 0; j < d; j++) {
            const float *tj = T + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) acc[c] += tj[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```