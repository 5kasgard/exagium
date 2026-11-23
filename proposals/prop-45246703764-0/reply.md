```hypothesis
Factorization Machine (FM) with low-rank pairwise interactions (rank 4-8) should capture important feature crosses in Adult (e.g., education×occupation, marital-status×relationship) that linear models miss. Training uses Adam on logistic loss with per-byte lookup tables for fast inference. If dev accuracy does not exceed label-linear's 0.8524, it suggests interactions are not beneficial or rank is insufficient.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 256
MAX_EPOCHS = 50
TUNE_EPOCHS = 30
RANK_CANDIDATES = [2, 4, 8, 16]

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
    n, d = X_bytes.shape
    assert d == 14
    types = card["input"]["types"]

    ordinal_stats = {}
    cat_vocabs = {}
    feature_to_expanded = []
    expanded_idx = 0
    for j, t in enumerate(types):
        col = X_bytes[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = col.mean(), col.std() or 1.0
            ordinal_stats[j] = (mu, sd)
            feature_to_expanded.append({})
            for byte in range(256):
                val = (byte - mu) / sd
                feature_to_expanded[j][byte] = (expanded_idx, val)
            expanded_idx += 1
        else:
            vals = np.unique(col)
            cat_vocabs[j] = vals
            feature_to_expanded.append({})
            for v in vals:
                feature_to_expanded[j][int(v)] = (expanded_idx, 1.0)
            for byte in range(256):
                if byte not in feature_to_expanded[j]:
                    feature_to_expanded[j][byte] = (-1, 0.0)
            expanded_idx += len(vals)
    k = expanded_idx

    Z = np.zeros((n, k), dtype=np.float32)
    for j in range(d):
        col = X_bytes[:, j]
        if types[j] == "ordinal":
            mu, sd = ordinal_stats[j]
            idx = feature_to_expanded[j][0][0]
            Z[:, idx] = (col - mu) / sd
        else:
            for v in cat_vocabs[j]:
                idx = feature_to_expanded[j][int(v)][0]
                Z[:, idx] = (col == v).astype(np.float32)

    tune_size = n // 5
    train_size = n - tune_size
    Z_train, Z_tune = Z[:train_size], Z[train_size:]
    y_train, y_tune = y[:train_size], y[train_size:]

    best_r = None
    best_acc = -1
    for r in RANK_CANDIDATES:
        rng = np.random.default_rng(SEED)
        bias = 0.0
        w = (rng.uniform(-1, 1, k) * math.sqrt(6.0 / k)).astype(np.float32)
        V = (rng.uniform(-1, 1, (k, r)) * math.sqrt(6.0 / k)).astype(np.float32)
        m_bias, v_bias = 0.0, 0.0
        m_w, v_w = np.zeros_like(w), np.zeros_like(w)
        m_V, v_V = np.zeros_like(V), np.zeros_like(V)
        t = 0
        for epoch in range(TUNE_EPOCHS):
            order = rng.permutation(train_size)
            for s in range(0, train_size, BATCH):
                idx = order[s:s+BATCH]
                z = Z_train[idx]
                yy = y_train[idx]
                b = len(yy)
                linear = z @ w
                zV = z @ V
                zV2 = z @ (V * V)
                fm = 0.5 * np.sum(zV * zV - zV2, axis=1)
                logits = bias + linear + fm
                p = 1.0 / (1.0 + np.exp(-logits))
                grad_logit = (p - yy).astype(np.float32)
                grad_bias = np.mean(grad_logit)
                grad_w = (z.T @ grad_logit) / b + L2 * w
                grad_V = (z.T @ (grad_logit[:, None] * zV)) / b
                grad_V -= (z.T @ grad_logit)[:, None] * V / b
                grad_V += L2 * V
                t += 1
                m_bias = 0.9 * m_bias + 0.1 * grad_bias
                v_bias = 0.999 * v_bias + 0.001 * grad_bias * grad_bias
                bias -= LR * (m_bias / (1 - 0.9**t)) / (math.sqrt(v_bias / (1 - 0.999**t)) + 1e-8)
                m_w = 0.9 * m_w + 0.1 * grad_w
                v_w = 0.999 * v_w + 0.001 * grad_w * grad_w
                w -= LR * (m_w / (1 - 0.9**t)) / (np.sqrt(v_w / (1 - 0.999**t)) + 1e-8)
                m_V = 0.9 * m_V + 0.1 * grad_V
                v_V = 0.999 * v_V + 0.001 * grad_V * grad_V
                V -= LR * (m_V / (1 - 0.9**t)) / (np.sqrt(v_V / (1 - 0.999**t)) + 1e-8)
        linear_tune = Z_tune @ w
        zV_tune = Z_tune @ V
        zV2_tune = Z_tune @ (V * V)
        fm_tune = 0.5 * np.sum(zV_tune * zV_tune - zV2_tune, axis=1)
        logits_tune = bias + linear_tune + fm_tune
        preds = (logits_tune > 0).astype(np.int64)
        acc = np.mean(preds == y_tune)
        if acc > best_acc:
            best_acc = acc
            best_r = r

    r = best_r
    rng = np.random.default_rng(SEED)
    bias = 0.0
    w = (rng.uniform(-1, 1, k) * math.sqrt(6.0 / k)).astype(np.float32)
    V = (rng.uniform(-1, 1, (k, r)) * math.sqrt(6.0 / k)).astype(np.float32)
    m_bias, v_bias = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0
    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            z = Z[idx]
            yy = y[idx]
            b = len(yy)
            linear = z @ w
            zV = z @ V
            zV2 = z @ (V * V)
            fm = 0.5 * np.sum(zV * zV - zV2, axis=1)
            logits = bias + linear + fm
            p = 1.0 / (1.0 + np.exp(-logits))
            grad_logit = (p - yy).astype(np.float32)
            grad_bias = np.mean(grad_logit)
            grad_w = (z.T @ grad_logit) / b + L2 * w
            grad_V = (z.T @ (grad_logit[:, None] * zV)) / b
            grad_V -= (z.T @ grad_logit)[:, None] * V / b
            grad_V += L2 * V
            t += 1
            m_bias = 0.9 * m_bias + 0.1 * grad_bias
            v_bias = 0.999 * v_bias + 0.001 * grad_bias * grad_bias
            bias -= LR * (m_bias / (1 - 0.9**t)) / (math.sqrt(v_bias / (1 - 0.999**t)) + 1e-8)
            m_w = 0.9 * m_w + 0.1 * grad_w
            v_w = 0.999 * v_w + 0.001 * grad_w * grad_w
            w -= LR * (m_w / (1 - 0.9**t)) / (np.sqrt(v_w / (1 - 0.999**t)) + 1e-8)
            m_V = 0.9 * m_V + 0.1 * grad_V
            v_V = 0.999 * v_V + 0.001 * grad_V * grad_V
            V -= LR * (m_V / (1 - 0.9**t)) / (np.sqrt(v_V / (1 - 0.999**t)) + 1e-8)

    linear_table = np.zeros((d, 256), dtype=np.float32)
    V_table = np.zeros((d, 256, r), dtype=np.float32)
    for j in range(d):
        for byte in range(256):
            exp_idx, val = feature_to_expanded[j][byte]
            if exp_idx >= 0:
                linear_table[j, byte] = val * w[exp_idx]
                V_table[j, byte, :] = val * V[exp_idx, :]

    out = struct.pack("<5i", 0x464D4C31, d, k, r, 2)
    out += struct.pack("<f", bias)
    for j in range(d):
        out += linear_table[j].astype("<f4").tobytes()
        out += V_table[j].astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
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
    if (nw < 20 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], r = hd[3], C = hd[4];
    if (d != 14 || C != 2 || r < 1 || r > 32) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected_floats = 1 + (long)d * 256 * (1 + r);
    long expected_bytes = 20 + 4 * expected_floats;
    if (nw != expected_bytes) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *bias = (const float *)(hd + 5);
    const float *tables = bias + 1;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = calloc(r, sizeof(float));
    float *sum_v2 = calloc(r, sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if (len != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        float logit = *bias;
        for (int j = 0; j < d; j++) {
            unsigned char byte = xi[j];
            const float *feat_table = tables + (long)j * 256 * (1 + r);
            const float *linear_table = feat_table;
            const float *V_table = feat_table + 256;
            logit += linear_table[byte];
            for (int f = 0; f < r; f++) {
                float v = V_table[(long)byte * r + f];
                sum_v[f] += v;
                sum_v2[f] += v * v;
            }
        }
        for (int f = 0; f < r; f++) {
            logit += 0.5f * (sum_v[f] * sum_v[f] - sum_v2[f]);
            sum_v[f] = 0.0f;
            sum_v2[f] = 0.0f;
        }
        unsigned char cls = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```