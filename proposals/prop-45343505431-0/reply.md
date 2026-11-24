```hypothesis
Family: label-fm (Factorization Machine). The Adult dataset has high-cardinality categorical features (e.g., native-country, occupation) where pairwise interactions are known to be predictive. A low-rank FM captures these interactions with O(k·FM_k) parameters and inference cost, often beating linear models on tabular data while staying fast. If dev accuracy ≤ 0.8524 (label-linear) or inference cost > 2× label-linear, the approach fails.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encoding: ordinal -> z-scored single feature; categorical -> one-hot over training values + unseen (255).
FM: w0 + w·x + sum_f 0.5[(V_f·x)^2 - sum_i V_fi^2 x_i]. Binary logistic loss, Adam.
Exported: magic 0x464D4C31, d, FM_k, k, C; per-column encoding; w0, w[k], V[FM_k][k]."""
import json, math, struct, sys

import numpy as np

FM_K = 8
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-4, 64, 30


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
    types = card["input"]["types"]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # Build encoding
    enc_info = []
    feat_idx = 0
    for j in range(d):
        col = X[:, j]
        if types[j] == "ordinal":
            mean = float(col.mean())
            std = float(col.std())
            if std == 0.0:
                std = 1.0
            enc_info.append(("ordinal", mean, std, feat_idx))
            feat_idx += 1
        else:
            uniq = np.unique(col)
            uniq = uniq[uniq != 255]          # 255 is unseen token
            n_cats = len(uniq)
            unseen_idx = feat_idx + n_cats
            map_arr = np.full(256, unseen_idx, dtype=np.uint16)
            for idx, val in enumerate(uniq):
                map_arr[val] = feat_idx + idx
            enc_info.append(("categorical", map_arr, unseen_idx, n_cats))
            feat_idx += n_cats + 1
    k = feat_idx

    # Build dense feature matrix
    n = X.shape[0]
    X_dense = np.zeros((n, k), dtype=np.float32)
    for j in range(d):
        col = X[:, j]
        info = enc_info[j]
        if info[0] == "ordinal":
            _, mean, std, fidx = info
            X_dense[:, fidx] = (col.astype(np.float32) - mean) / std
        else:
            _, map_arr, unseen_idx, n_cats = info
            idxs = map_arr[col]
            X_dense[np.arange(n), idxs] = 1.0

    # FM parameters
    w0 = np.float32(0.0)
    w = np.zeros(k, dtype=np.float32)
    V = rng.normal(0, 0.01, size=(FM_K, k)).astype(np.float32)

    # Adam state
    m_w0, v_w0 = np.float32(0.0), np.float32(0.0)
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)

    t = 0
    y2 = 2 * y - 1  # -1, 1
    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X_dense[idx]
            yb = y2[idx]
            B = len(idx)

            # Forward
            linear = w0 + xb @ w
            inter = np.zeros(B, dtype=np.float32)
            for f in range(FM_K):
                Vf = V[f]
                dot = xb @ Vf
                dot2 = xb @ (Vf * Vf)
                inter += 0.5 * (dot * dot - dot2)
            score = linear + inter

            # Logistic loss gradient
            z = -yb * score
            sig = 1.0 / (1.0 + np.exp(-z))
            grad_score = (-yb * sig).astype(np.float32) / B

            # Gradients
            gw0 = grad_score.sum()
            gw = xb.T @ grad_score + L2 * w
            gV = np.zeros_like(V)
            for f in range(FM_K):
                Vf = V[f]
                dot = xb @ Vf
                A = xb.T @ (grad_score * dot)
                Bv = xb.T @ grad_score
                gV[f] = A - Vf * Bv + L2 * Vf

            # Adam updates
            t += 1
            beta1, beta2 = 0.9, 0.999
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * gw0 * gw0
            w0 -= LR * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + 1e-8)

            m_w = beta1 * m_w + (1 - beta1) * gw
            v_w = beta2 * v_w + (1 - beta2) * gw * gw
            w -= LR * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + 1e-8)

            m_V = beta1 * m_V + (1 - beta1) * gV
            v_V = beta2 * v_V + (1 - beta2) * gV * gV
            V -= LR * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + 1e-8)

    # Write weights.bin
    magic = 0x464D4C31  # 'FML1'
    header = struct.pack("<5i", magic, d, FM_K, k, C)
    col_data = bytearray()
    for j in range(d):
        info = enc_info[j]
        if info[0] == "ordinal":
            _, mean, std, fidx = info
            col_data += struct.pack("<BffI", 0, mean, std, fidx)
        else:
            _, map_arr, unseen_idx, n_cats = info
            col_data += struct.pack("<B", 1)
            col_data += map_arr.astype("<u2").tobytes()
    col_data += struct.pack("<f", w0)
    col_data += w.astype("<f4").tobytes()
    col_data += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + col_data)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class index (0 or 1). */
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
    long nw, nx = 0, N = atol(argv[3]);
    unsigned char *wbuf = slurp(argv[1], &nw);
    unsigned char *xbuf = N ? slurp(argv[2], &nx) : NULL;

    if (nw < 20) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    int32_t *hdr = (int32_t *)wbuf;
    if (hdr[0] != 0x464D4C31) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hdr[1], FM_k = hdr[2], k = hdr[3], C = hdr[4];
    if (d != 14 || C != 2) { fprintf(stderr, "bad header values\n"); return 3; }

    unsigned char *p = wbuf + 20;
    uint8_t col_type[14];
    float ord_mean[14], ord_std[14];
    int32_t ord_feat_idx[14];
    uint16_t *cat_map[14];

    for (int j = 0; j < d; j++) {
        uint8_t type = *p++;
        col_type[j] = type;
        if (type == 0) {
            float mean, std; int32_t feat_idx;
            memcpy(&mean, p, 4); p += 4;
            memcpy(&std, p, 4); p += 4;
            memcpy(&feat_idx, p, 4); p += 4;
            ord_mean[j] = mean; ord_std[j] = std; ord_feat_idx[j] = feat_idx;
        } else if (type == 1) {
            cat_map[j] = (uint16_t *)p;
            p += 256 * 2;
        } else {
            fprintf(stderr, "unknown column type\n"); return 3;
        }
    }

    float w0; memcpy(&w0, p, 4); p += 4;
    float *w = (float *)p; p += (long)k * 4;
    float *V = (float *)p; p += (long)FM_k * k * 4;

    if (p != wbuf + nw) { fprintf(stderr, "weights size mismatch\n"); return 3; }

    float *x = (float *)malloc(k * sizeof(float));
    if (!x) { fprintf(stderr, "malloc failed\n"); return 3; }

    long pos = 0;
    for (long i = 0; i < N; i++) {
        uint32_t len;
        const unsigned char *rec = next_rec(xbuf, nx, &pos, &len);
        if (len != 14) { fprintf(stderr, "record %ld has length %u, expected 14\n", i, len); return 3; }

        memset(x, 0, k * sizeof(float));
        for (int j = 0; j < d; j++) {
            uint8_t byte = rec[j];
            if (col_type[j] == 0) {
                int idx = ord_feat_idx[j];
                x[idx] = (byte - ord_mean[j]) / ord_std[j];
            } else {
                uint16_t idx = cat_map[j][byte];
                x[idx] = 1.0f;
            }
        }

        float score = w0;
        for (int i = 0; i < k; i++) score += w[i] * x[i];
        for (int f = 0; f < FM_k; f++) {
            float dot = 0.0f, dot2 = 0.0f;
            const float *Vf = V + (long)f * k;
            for (int i = 0; i < k; i++) {
                float xi = x[i];
                if (xi != 0.0f) {
                    float v = Vf[i];
                    dot += v * xi;
                    dot2 += v * v * xi;
                }
            }
            score += 0.5f * (dot * dot - dot2);
        }

        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }

    free(x); free(wbuf); if (xbuf) free(xbuf);
    return 0;
}
```