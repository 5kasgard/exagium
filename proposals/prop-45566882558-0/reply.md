```hypothesis
Factorization Machines (FM) model pairwise feature interactions via low-rank factorization, which is known to excel on tabular data like Adult. By adding a rank-8 factor matrix to a linear model, we capture interactions at inference cost O(d*k) (d=14 columns, k=8), comparable to label-linear. We tune k on the last 20% of train.in, then train on all data. Expected outcome: dev accuracy >0.855 (beating label-linear's 0.8524) with Ir/item <1500 (below label-mlp16's 2325). If accuracy does not improve by ≥0.003 or cost exceeds label-mlp16, the approach fails.
```
```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes ordinal as standardized, categorical as one-hot (unseen=zero).
Trains FM with Adam (lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n))).
Tunes factor rank k on last 20% of train.in, then retrains on all data.
Exports per-byte lookup tables: magic 0x464D3130, d, k, C, bias, then d tables of 256*(1+2k) float32."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
K_CHOICES = [4, 8, 16]  # factor ranks to try on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_fm(Z, y, idx, k, rng, epochs):
    """Train FM on subset idx, return w, V, bias."""
    n, D = Z.shape
    w = np.zeros(D, np.float32)
    V = (rng.uniform(-1, 1, (D, k)) * math.sqrt(6.0 / D)).astype(np.float32)
    bias = np.array([0.0], np.float32)
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    m_b, v_b = np.zeros_like(bias), np.zeros_like(bias)
    t = 0
    for _ in range(epochs):
        order = rng.permutation(len(idx))
        for s in range(0, len(idx), BATCH):
            batch_idx = idx[order[s:s + BATCH]]
            z = Z[batch_idx]
            yy = y[batch_idx]
            # forward
            linear = z @ w + bias[0]
            inter = np.zeros_like(linear)
            for f in range(k):
                Vf = V[:, f]
                sum_f = z @ Vf
                sq_sum_f = (z * z) @ (Vf * Vf)
                inter += 0.5 * (sum_f * sum_f - sq_sum_f)
            scores = linear + inter
            p = 1.0 / (1.0 + np.exp(-scores))
            g = p - yy  # (B,)
            # gradients
            grad_w = (z.T @ g) / len(batch_idx) + L2 * w
            grad_b = np.array([g.mean()], np.float32)
            grad_V = np.zeros_like(V)
            for f in range(k):
                Vf = V[:, f]
                sum_f = z @ Vf
                term = sum_f[:, None] - z * Vf[None, :]
                grad_V[:, f] = (z * term).T @ g / len(batch_idx) + L2 * Vf
            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            m_w = beta1 * m_w + (1 - beta1) * grad_w
            v_w = beta2 * v_w + (1 - beta2) * grad_w * grad_w
            w -= LR * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + 1e-8)
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * grad_b * grad_b
            bias -= LR * (m_b / (1 - beta1 ** t)) / (np.sqrt(v_b / (1 - beta2 ** t)) + 1e-8)
            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * grad_V * grad_V
            V -= LR * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + 1e-8)
    return w, V, bias


def evaluate_fm(Z, y, idx, w, V, bias, k):
    z = Z[idx]
    yy = y[idx]
    linear = z @ w + bias[0]
    inter = np.zeros_like(linear)
    for f in range(k):
        Vf = V[:, f]
        sum_f = z @ Vf
        sq_sum_f = (z * z) @ (Vf * Vf)
        inter += 0.5 * (sum_f * sum_f - sq_sum_f)
    scores = linear + inter
    pred = (scores > 0).astype(np.int64)
    return (pred == yy).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X_bytes.shape
    rng = np.random.default_rng(SEED)

    # --- encoding ---
    col_types = []
    ord_params = []   # (mu, sd, feat_idx)
    cat_params = []   # (mapping, offset, num_cats)
    D = 0
    for j, t in enumerate(card["input"]["types"]):
        col = X_bytes[:, j]
        if t == "ordinal":
            mu, sd = col.mean(), col.std()
            if sd == 0:
                sd = 1.0
            col_types.append("ordinal")
            ord_params.append((mu, sd, D))
            D += 1
        else:
            vals = np.unique(col)
            mapping = {v: i for i, v in enumerate(vals)}
            col_types.append("categorical")
            cat_params.append((mapping, D, len(vals)))
            D += len(vals)

    Z = np.zeros((n, D), np.float32)
    for j, t in enumerate(col_types):
        if t == "ordinal":
            mu, sd, fi = ord_params[j]
            Z[:, fi] = (X_bytes[:, j] - mu) / sd
        else:
            mapping, off, _ = cat_params[j]
            for v, ci in mapping.items():
                Z[X_bytes[:, j] == v, off + ci] = 1.0

    # --- tuning split (last 20% in file order) ---
    split = int(0.8 * n)
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)

    best_k, best_acc = K_CHOICES[0], -1.0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(train_idx)))
    for k in K_CHOICES:
        w, V, bias = train_fm(Z, y, train_idx, k, rng, epochs)
        acc = evaluate_fm(Z, y, val_idx, w, V, bias, k)
        if acc > best_acc:
            best_acc, best_k = acc, k

    # --- final training on all data ---
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    w, V, bias = train_fm(Z, y, np.arange(n), best_k, rng, epochs)

    # --- build lookup tables ---
    tables = []
    for j, t in enumerate(col_types):
        tbl = np.zeros((256, 1 + 2 * best_k), np.float32)
        if t == "ordinal":
            mu, sd, fi = ord_params[j]
            for b in range(256):
                x = (b - mu) / sd
                tbl[b, 0] = w[fi] * x
                for f in range(best_k):
                    v = V[fi, f]
                    tbl[b, 1 + f] = v * x
                    tbl[b, 1 + best_k + f] = (v * x) ** 2
        else:
            mapping, off, _ = cat_params[j]
            for b in range(256):
                if b in mapping:
                    fi = off + mapping[b]
                    tbl[b, 0] = w[fi]
                    for f in range(best_k):
                        v = V[fi, f]
                        tbl[b, 1 + f] = v
                        tbl[b, 1 + best_k + f] = v * v
        tables.append(tbl)

    # --- write weights.bin ---
    magic = 0x464D3130
    C = card["classes"]
    out = struct.pack("<4i", magic, d, best_k, C)
    out += np.array([bias[0]], dtype="<f4").tobytes()
    for tbl in tables:
        out += tbl.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: one byte class (0/1).
 * Weights format: magic 0x464D3130, int32 d, k, C; float32 bias; then d tables of 256*(1+2k) float32.
 * Each table entry: linear, k factor contributions, k squared contributions. */
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
    long nw, nx = 0, N = atol(argv[3]);
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d != 14 || k <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4 + (long)d * 256 * (1 + 2 * k) * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    float bias = ((float *)(hd + 4))[0];
    const float *T = (const float *)(hd + 4) + 1;  // skip bias
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    long pos = 0;
    float *factor_acc = calloc(k, sizeof(float));
    float *sq_sum = calloc(k, sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        float linear_acc = bias;
        for (int f = 0; f < k; f++) { factor_acc[f] = 0; sq_sum[f] = 0; }
        for (int j = 0; j < d; j++) {
            unsigned char byte = xi[j];
            const float *row = T + ((long)j * 256 + byte) * (1 + 2 * k);
            linear_acc += row[0];
            for (int f = 0; f < k; f++) {
                factor_acc[f] += row[1 + f];
                sq_sum[f] += row[1 + k + f];
            }
        }
        float interaction = 0;
        for (int f = 0; f < k; f++) {
            interaction += factor_acc[f] * factor_acc[f] - sq_sum[f];
        }
        float score = linear_acc + 0.5f * interaction;
        unsigned char cls = (score > 0) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```