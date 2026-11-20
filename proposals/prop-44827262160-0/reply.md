```hypothesis
Family: label-fm (factorization machine with pairwise interactions).
Why: Adult has strong feature interactions (e.g., education × occupation, marital-status × relationship). A degree-2 FM captures all pairwise interactions via low-rank embeddings (k=8) while staying linear in the number of non-zero features (14 per row). It adds only ~d*k parameters over linear, so inference stays fast. If dev accuracy ≤ label-linear (0.8524) or Ir/item > 1000, the hypothesis fails.
```

```python
"""label-fm: Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Encoding: ordinal -> z-score; categorical -> one-hot (training cats + unseen=255).
FM: w0 + sum(w_i x_i) + 0.5*( (sum V_i x_i)^2 - sum (V_i x_i)^2 ).
Adam lr=1e-2, L2=1e-4, batch=256, epochs=min(20, ceil(600000/n)).
Weights: magic 0x464D4C31, d, k, nfeat=14; per-feature metadata; w0, w[d], V[d*k] (all float32)."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 256, 600000, 20


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, nfeat = X_bytes.shape
    assert nfeat == 14
    types = card["input"]["types"]

    # --- preprocessing ---
    ordinal_idx = [j for j, t in enumerate(types) if t == "ordinal"]
    cat_idx = [j for j, t in enumerate(types) if t == "categorical"]

    # ordinal: compute mean/std per feature
    ord_mean = np.zeros(len(ordinal_idx), dtype=np.float32)
    ord_std = np.ones(len(ordinal_idx), dtype=np.float32)
    for ii, j in enumerate(ordinal_idx):
        v = X_bytes[:, j].astype(np.float32)
        ord_mean[ii] = v.mean()
        sd = v.std()
        if sd > 0:
            ord_std[ii] = sd

    # categorical: collect training categories + unseen (255)
    cat_maps = []          # list of dict: byte_value -> global feature index
    cat_unseen_idx = []    # global index for unseen (255)
    global_idx = len(ordinal_idx)
    for j in cat_idx:
        vals = np.unique(X_bytes[:, j])
        # ensure 255 is included as unseen
        if 255 not in vals:
            vals = np.append(vals, 255)
        mapping = {int(v): global_idx + i for i, v in enumerate(vals)}
        cat_maps.append(mapping)
        cat_unseen_idx.append(mapping[255])
        global_idx += len(vals)
    d = global_idx

    # build sparse representation: for each sample, 14 (idx, value) pairs
    idx_arr = np.zeros((n, 14), dtype=np.int32)
    val_arr = np.zeros((n, 14), dtype=np.float32)
    for i in range(n):
        pos = 0
        # ordinal
        for ii, j in enumerate(ordinal_idx):
            idx_arr[i, pos] = ii
            val_arr[i, pos] = (X_bytes[i, j] - ord_mean[ii]) / ord_std[ii]
            pos += 1
        # categorical
        for ci, j in enumerate(cat_idx):
            b = int(X_bytes[i, j])
            gidx = cat_maps[ci].get(b, cat_unseen_idx[ci])
            idx_arr[i, pos] = gidx
            val_arr[i, pos] = 1.0
            pos += 1

    # --- FM parameters ---
    rng = np.random.default_rng(SEED)
    w0 = np.float32(0.0)
    w = np.zeros(d, dtype=np.float32)
    V = (rng.normal(0, 0.01, (d, K))).astype(np.float32)

    # Adam state
    m_w0, v_w0 = np.float32(0.0), np.float32(0.0)
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            batch_idx = order[s:s + BATCH]
            bsz = len(batch_idx)
            idx_b = idx_arr[batch_idx]          # (bsz, 14)
            val_b = val_arr[batch_idx]          # (bsz, 14)
            y_b = y[batch_idx]

            # gather weights and embeddings
            w_b = w[idx_b]                      # (bsz, 14)
            V_b = V[idx_b]                      # (bsz, 14, K)

            # linear term
            lin = w0 + np.sum(w_b * val_b, axis=1)                # (bsz,)

            # interaction term
            Vx = V_b * val_b[:, :, None]                          # (bsz, 14, K)
            sum_Vx = np.sum(Vx, axis=1)                           # (bsz, K)
            sum_Vx2 = np.sum(Vx * Vx, axis=1)                     # (bsz, K)
            inter = 0.5 * np.sum(sum_Vx * sum_Vx - sum_Vx2, axis=1)  # (bsz,)

            score = lin + inter
            p = sigmoid(score)
            # gradient of log loss w.r.t score
            grad_score = (p - y_b).astype(np.float32)             # (bsz,)

            # gradients
            g_w0 = np.mean(grad_score)
            g_w = np.zeros_like(w)
            g_V = np.zeros_like(V)

            # linear weights gradient
            for f in range(14):
                np.add.at(g_w, idx_b[:, f], grad_score * val_b[:, f])
            g_w /= bsz
            g_w += L2 * w

            # V gradient
            # d(inter)/d(Vx) = sum_Vx - Vx  (broadcast over features)
            d_inter_d_Vx = sum_Vx[:, None, :] - Vx                # (bsz, 14, K)
            d_score_d_Vx = grad_score[:, None, None] * d_inter_d_Vx  # (bsz, 14, K)
            for f in range(14):
                np.add.at(g_V, idx_b[:, f], d_score_d_Vx[:, f, :])
            g_V /= bsz
            g_V += L2 * V

            # Adam update
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            w0 -= LR * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)
            # w
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            w -= LR * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)
            # V
            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * g_V * g_V
            V -= LR * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)

    # --- write weights.bin ---
    # magic, d, K, nfeat
    out = struct.pack("<4i", 0x464D4C31, d, K, nfeat)
    # per feature metadata
    for j in range(nfeat):
        if types[j] == "ordinal":
            # find which ordinal index
            oi = ordinal_idx.index(j)
            out += struct.pack("<bff", 0, ord_mean[oi], ord_std[oi])  # type, mean, std
            out += struct.pack("<i", oi)                               # global index
        else:
            out += struct.pack("<b", 1)  # type categorical
            # map[256] as int16
            mapping = cat_maps[cat_idx.index(j)]
            arr = np.full(256, -1, dtype=np.int16)
            for v, g in mapping.items():
                arr[v] = g
            out += arr.tobytes()
    # parameters
    out += struct.pack("<f", w0)
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes. Output: 1 byte class (0/1). */
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
    int32_t *hdr = (int32_t *)wb;
    if (hdr[0] != 0x464D4C31) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hdr[1], K = hdr[2], nfeat = hdr[3];
    if (nfeat != 14 || d < 14 || K < 1 || K > 64) { fprintf(stderr, "bad header\n"); return 3; }

    unsigned char *p = wb + 16;
    float ord_mean[14], ord_std[14];
    int32_t ord_gidx[14];
    int16_t cat_map[14][256];
    int feature_type[14]; // 0 ordinal, 1 categorical
    for (int j = 0; j < 14; j++) {
        int8_t type = *(int8_t *)p; p += 1;
        feature_type[j] = type;
        if (type == 0) {
            ord_mean[j] = *(float *)p; p += 4;
            ord_std[j] = *(float *)p; p += 4;
            ord_gidx[j] = *(int32_t *)p; p += 4;
        } else {
            memcpy(cat_map[j], p, 256 * sizeof(int16_t));
            p += 256 * sizeof(int16_t);
        }
    }
    float w0 = *(float *)p; p += 4;
    float *w = (float *)p; p += d * 4;
    float *V = (float *)p; p += (long)d * K * 4;
    if (p != wb + nw) { fprintf(stderr, "weights size mismatch\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_V[64], sum_V2[64];
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 14) { fprintf(stderr, "item %ld is %u bytes, not 14\n", i, k); return 3; }

        float linear = w0;
        for (int u = 0; u < K; u++) { sum_V[u] = 0.0f; sum_V2[u] = 0.0f; }

        for (int j = 0; j < 14; j++) {
            if (feature_type[j] == 0) { // ordinal
                float val = (xi[j] - ord_mean[j]) / ord_std[j];
                int idx = ord_gidx[j];
                linear += w[idx] * val;
                for (int u = 0; u < K; u++) {
                    float v = V[(long)idx * K + u] * val;
                    sum_V[u] += v;
                    sum_V2[u] += v * v;
                }
            } else { // categorical
                int idx = cat_map[j][xi[j]];
                if (idx >= 0) {
                    linear += w[idx];
                    for (int u = 0; u < K; u++) {
                        float v = V[(long)idx * K + u];
                        sum_V[u] += v;
                        sum_V2[u] += v * v;
                    }
                }
            }
        }
        float inter = 0.0f;
        for (int u = 0; u < K; u++) inter += sum_V[u] * sum_V[u] - sum_V2[u];
        inter *= 0.5f;
        float score = linear + inter;
        float prob = 1.0f / (1.0f + expf(-score));
        unsigned char cls = (prob > 0.5f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```