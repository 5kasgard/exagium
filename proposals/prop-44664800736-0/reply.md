```hypothesis
Family: label-fm (Factorization Machine with rank-8 latent interactions).
Why: Adult has strong categorical interactions (e.g., education × occupation) that linear models miss. A rank-8 FM adds only ~8×D parameters but captures pairwise effects, aiming to beat label-linear (0.8524) while staying cheaper than label-mlp16 (2325 Ir/item).
Failure: If dev accuracy ≤ 0.8524 or Ir/item ≥ 2325, the hypothesis is invalid.
```

```python
"""Factorization Machine family for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes ordinal as z-scored, categorical as one-hot over training values (unseen=zero).
Trains binary FM with Adam (lr=1e-2, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n))).
Exports: magic 0x464D4C31, d, K, C, D; per-feature encoding; w0, w[D], V[D*K] (float32 LE)."""
import json, math, struct, sys
import numpy as np

SEED, K, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 8, 1e-2, 1e-4, 64, 600000, 100


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
    n, d = X.shape
    rng = np.random.default_rng(SEED)

    enc_info = []
    Z_parts = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j]
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc_info.append(("ordinal", mu, sd))
            Z_parts.append(((v - mu) / sd)[:, None].astype(np.float32))
        else:
            vals = np.unique(v)
            val_to_idx = {val: i for i, val in enumerate(vals)}
            enc_info.append(("categorical", vals, val_to_idx))
            z = np.zeros((n, len(vals)), dtype=np.float32)
            for i, val in enumerate(v):
                idx = val_to_idx.get(val)
                if idx is not None:
                    z[i, idx] = 1.0
            Z_parts.append(z)
    Z = np.concatenate(Z_parts, axis=1)
    D = Z.shape[1]

    w0 = 0.0
    w = np.zeros(D, np.float32)
    V = rng.normal(0, 0.01, (D, K)).astype(np.float32)
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            zV = z @ V
            z2 = z * z
            V2 = V * V
            linear = w0 + z @ w
            inter = 0.5 * np.sum(zV * zV - z2 @ V2, axis=1)
            logit = linear + inter
            p = 1.0 / (1.0 + np.exp(-logit))
            grad = (p - yy).astype(np.float32)
            g_w0 = grad.mean()
            g_w = (z.T @ grad) / len(yy) + L2 * w
            g_V = (z.T @ (grad[:, None] * zV) - z2.T @ (grad[:, None] * V)) / len(yy) + L2 * V
            t += 1
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            w0 -= LR * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            w -= LR * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)
            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * g_V * g_V
            V -= LR * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)

    buf = bytearray()
    buf += struct.pack("<5i", 0x464D4C31, d, K, 2, D)
    for info in enc_info:
        if info[0] == "ordinal":
            buf += struct.pack("<iff", 0, info[1], info[2])
        else:
            vals = info[1]
            buf += struct.pack("<ii", 1, len(vals))
            buf += vals.astype(np.uint8).tobytes()
    buf += struct.pack("<f", w0)
    buf += w.astype("<f4").tobytes()
    buf += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(buf)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class (0/1). */
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

typedef struct {
    int type;          // 0 ordinal, 1 categorical
    float mean, std;   // ordinal
    int ncat;          // categorical
    int offset;        // start index in w/V
    unsigned char map[256]; // categorical: byte -> index (255=unseen)
} Feature;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    uint32_t *hd = (uint32_t *)wb;
    if (nw < 20 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3], D = hd[4];
    if (d != 14 || C != 2 || K <= 0 || D <= 0) { fprintf(stderr, "bad header\n"); return 3; }

    Feature feats[14];
    const unsigned char *p = wb + 20;
    long pmax = nw;
    for (int j = 0; j < d; j++) {
        if (p + 4 > wb + nw) { fprintf(stderr, "truncated weights\n"); return 3; }
        int type = *(int32_t *)p; p += 4;
        feats[j].type = type;
        if (type == 0) { // ordinal
            if (p + 8 > wb + nw) { fprintf(stderr, "truncated weights\n"); return 3; }
            feats[j].mean = *(float *)p; p += 4;
            feats[j].std = *(float *)p; p += 4;
            feats[j].ncat = 0;
        } else { // categorical
            if (p + 4 > wb + nw) { fprintf(stderr, "truncated weights\n"); return 3; }
            feats[j].ncat = *(int32_t *)p; p += 4;
            int nc = feats[j].ncat;
            if (p + nc > wb + nw) { fprintf(stderr, "truncated weights\n"); return 3; }
            memset(feats[j].map, 255, 256);
            for (int i = 0; i < nc; i++) feats[j].map[p[i]] = i;
            p += nc;
        }
    }
    if (p + 4 > wb + nw) { fprintf(stderr, "truncated weights\n"); return 3; }
    float w0 = *(float *)p; p += 4;
    if (p + (long)D * 4 > wb + nw) { fprintf(stderr, "truncated weights\n"); return 3; }
    const float *w = (const float *)p; p += D * 4;
    if (p + (long)D * K * 4 > wb + nw) { fprintf(stderr, "truncated weights\n"); return 3; }
    const float *V = (const float *)p; p += D * K * 4;
    if (p != wb + nw) { fprintf(stderr, "weights size mismatch\n"); return 3; }

    // assign offsets
    int off = 0;
    for (int j = 0; j < d; j++) {
        feats[j].offset = off;
        off += (feats[j].type == 0) ? 1 : feats[j].ncat;
    }
    if (off != D) { fprintf(stderr, "offset mismatch\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_vx = calloc(K, sizeof(float));
    float *sum_v2x2 = calloc(K, sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float linear = w0;
        memset(sum_vx, 0, K * sizeof(float));
        memset(sum_v2x2, 0, K * sizeof(float));
        for (int j = 0; j < d; j++) {
            Feature *f = &feats[j];
            int idx = f->offset;
            if (f->type == 0) {
                float val = (xi[j] - f->mean) / f->std;
                linear += w[idx] * val;
                for (int u = 0; u < K; u++) {
                    float v = V[(long)idx * K + u];
                    sum_vx[u] += v * val;
                    sum_v2x2[u] += v * v * val * val;
                }
            } else {
                unsigned char mi = f->map[xi[j]];
                if (mi != 255) {
                    int idxx = idx + mi;
                    linear += w[idxx];
                    for (int u = 0; u < K; u++) {
                        float v = V[(long)idxx * K + u];
                        sum_vx[u] += v;
                        sum_v2x2[u] += v * v;
                    }
                }
            }
        }
        float inter = 0.0f;
        for (int u = 0; u < K; u++) inter += 0.5f * (sum_vx[u] * sum_vx[u] - sum_v2x2[u]);
        float logit = linear + inter;
        unsigned char cls = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```