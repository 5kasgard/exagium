```hypothesis
Factorization Machine (degree 2) with small embeddings per feature-value.
Treats all 14 columns as 256-category fields, learns a linear weight and a K-dim vector per (field,value).
Pairwise dot products capture interactions; K=4 keeps params ~18k and inference ~400 FLOPs/item.
Should beat label-linear (0.8524) by modeling interactions, at lower cost than label-mlp16 (2325 Ir/item).
Fails if dev accuracy ≤ 0.8524 or Ir/item > 1500.
```

```python
"""Factorization Machine (degree 2) for tabular label tasks with byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encodes each of the 14 bytes as a categorical field with 256 values (0-255).
Learns: w0, linear[d][256], emb[d][256][K]. Binary cross-entropy with Adam.
Tunes K in {2,4,8} on last 20% of train.in (file order), then retrains on all train."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, EPOCHS_TUNE, EPOCHS_FULL = 1, 0.01, 1e-4, 64, 8, 30
K_CHOICES = [2, 4, 8]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(logits, y):
    return np.mean((logits > 0) == y)


def train_fm(X, y, K, epochs, rng):
    n, d = X.shape
    # params
    w0 = np.float32(0.0)
    lin = np.zeros((d, 256), np.float32)
    emb = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)
    # Adam state
    m_w0, v_w0 = np.float32(0.0), np.float32(0.0)
    m_lin, v_lin = np.zeros_like(lin), np.zeros_like(lin)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)
            # forward
            logits = np.full(bsz, w0, np.float32)
            for j in range(d):
                v = xb[:, j]
                logits += lin[j, v]
            # pairwise interactions
            for i in range(d):
                ei = emb[i, xb[:, i]]  # (bsz, K)
                for j in range(i + 1, d):
                    ej = emb[j, xb[:, j]]
                    logits += np.sum(ei * ej, axis=1)
            # loss gradient (binary cross-entropy with logits)
            p = 1.0 / (1.0 + np.exp(-logits))
            g = (p - yb).astype(np.float32)  # dL/dlogit
            # gradients
            gw0 = np.mean(g)
            glin = np.zeros_like(lin)
            gemb = np.zeros_like(emb)
            for j in range(d):
                v = xb[:, j]
                np.add.at(glin[j], v, g)
            for i in range(d):
                ei = emb[i, xb[:, i]]
                for j in range(i + 1, d):
                    ej = emb[j, xb[:, j]]
                    # d/demb_i = g * ej, d/demb_j = g * ei
                    g_exp = g[:, None]
                    np.add.at(gemb[i], xb[:, i], g_exp * ej)
                    np.add.at(gemb[j], xb[:, j], g_exp * ei)
            # L2 on embeddings only (linear already has one-hot structure)
            gemb += L2 * emb
            # Adam update
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * gw0 * gw0
            w0 -= LR * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)
            # lin
            m_lin = beta1 * m_lin + (1 - beta1) * glin
            v_lin = beta2 * v_lin + (1 - beta2) * glin * glin
            lin -= LR * (m_lin / (1 - beta1 ** t)) / (np.sqrt(v_lin / (1 - beta2 ** t)) + eps)
            # emb
            m_emb = beta1 * m_emb + (1 - beta1) * gemb
            v_emb = beta2 * v_emb + (1 - beta2) * gemb * gemb
            emb -= LR * (m_emb / (1 - beta1 ** t)) / (np.sqrt(v_emb / (1 - beta2 ** t)) + eps)
    return w0, lin, emb


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X.shape
    assert d == 14
    # split last 20% for tuning
    n_tune = n // 5
    X_tr, y_tr = X[:-n_tune], y[:-n_tune]
    X_te, y_te = X[-n_tune:], y[-n_tune:]
    rng = np.random.default_rng(SEED)
    best_K, best_acc = 2, -1.0
    for K in K_CHOICES:
        r = np.random.default_rng(SEED + K)  # deterministic per K
        w0, lin, emb = train_fm(X_tr, y_tr, K, EPOCHS_TUNE, r)
        # eval on tune
        logits = np.full(n_tune, w0, np.float32)
        for j in range(d):
            v = X_te[:, j]
            logits += lin[j, v]
        for i in range(d):
            ei = emb[i, X_te[:, i]]
            for j in range(i + 1, d):
                ej = emb[j, X_te[:, j]]
                logits += np.sum(ei * ej, axis=1)
        acc = accuracy(logits, y_te)
        if acc > best_acc:
            best_acc, best_K = acc, K
    # retrain on full data
    r = np.random.default_rng(SEED + best_K)
    w0, lin, emb = train_fm(X, y, best_K, EPOCHS_FULL, r)
    # write weights: magic, d, K, C=2; then w0, lin[d][256], emb[d][256][K]
    magic = 0x464D3231  # 'FM21'
    out = struct.pack("<4i", magic, d, best_K, 2)
    out += np.array([w0], dtype="<f4").tobytes()
    out += lin.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine (degree 2) inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes (one per feature). Output: one byte class (0/1). */
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
    if (nw < 16 || hd[0] != 0x464D3231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K < 1 || K > 8) { fprintf(stderr, "bad weights config\n"); return 3; }
    long expected = 16 + 4L + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *lin = p + 1;
    const float *emb = lin + d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *ei = malloc(K * sizeof(float));
    float *ej = malloc(K * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float logit = w0;
        for (int j = 0; j < d; j++) {
            logit += lin[j * 256 + xi[j]];
        }
        for (int i1 = 0; i1 < d; i1++) {
            const float *e1 = emb + (i1 * 256 + xi[i1]) * K;
            for (int j1 = i1 + 1; j1 < d; j1++) {
                const float *e2 = emb + (j1 * 256 + xi[j1]) * K;
                float sum = 0.0f;
                for (int kk = 0; kk < K; kk++) sum += e1[kk] * e2[kk];
                logit += sum;
            }
        }
        unsigned char cls = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    free(ei); free(ej);
    return 0;
}
```