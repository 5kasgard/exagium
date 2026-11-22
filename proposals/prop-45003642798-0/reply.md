```hypothesis
Factorization Machine (FM) with latent dim k=4, trained via SGD on log loss. FM captures pairwise feature interactions crucial for tabular data (e.g., education×occupation) while adding only O(k·M) params. Inference costs O(k·d) per item (d=14), faster than GBDT/MLP. If dev accuracy does not exceed label-linear by ≥0.5% (i.e., <0.857), the hypothesis fails.
```

```python
"""Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes ordinal as z-scored continuous, categorical as one-hot.
Tunes hyperparameters on last 20% of train.in (file order), then retrains on all data.
Exports weights as: magic(0x464D3100), d, k, M, C; per-column encoding; w0, w[M], V[M,k]."""
import json, math, struct, sys

import numpy as np

# Fixed seed for reproducibility
SEED = 1
# Hyperparameter grid for tuning on the tuning split
GRID = {
    'k': [2, 4, 8],
    'lr': [0.01, 0.05],
    'l2': [1e-4, 1e-3],
    'epochs': [5, 10],
    'batch': [64]
}

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = records(sys.argv[2])
    y_bytes = records(sys.argv[3])
    n = len(X_bytes)
    d = card["input"]["bytes"]
    types = card["input"]["types"]

    X = np.frombuffer(b"".join(X_bytes), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in y_bytes], dtype=np.int64)

    # Split: last 20% for tuning (file order)
    split_idx = int(n * 0.8)
    X_train, X_tune = X[:split_idx], X[split_idx:]
    y_train, y_tune = y[:split_idx], y[split_idx:]

    # Build encoding from training data only
    ord_mean = np.zeros(d, dtype=np.float32)
    ord_std  = np.zeros(d, dtype=np.float32)
    ord_feat = np.zeros(d, dtype=np.int32)
    cat_map  = [None] * d
    feat_idx_counter = 0

    for j, t in enumerate(types):
        col = X_train[:, j]
        if t == "ordinal":
            mu = col.mean()
            sd = col.std() or 1.0
            ord_mean[j] = mu
            ord_std[j]  = sd
            ord_feat[j] = feat_idx_counter
            feat_idx_counter += 1
        else:
            uniq = np.unique(col)
            mapping = -np.ones(256, dtype=np.int32)
            for v in uniq:
                mapping[v] = feat_idx_counter
                feat_idx_counter += 1
            cat_map[j] = mapping

    M = feat_idx_counter
    print(f"Encoded features M={M}", file=sys.stderr)

    # Precompute active features for a dataset
    def precompute(X_part):
        active = []
        for i in range(X_part.shape[0]):
            acts = []
            for j, t in enumerate(types):
                if t == "ordinal":
                    idx = ord_feat[j]
                    val = (X_part[i, j] - ord_mean[j]) / ord_std[j]
                    acts.append((idx, val))
                else:
                    idx = cat_map[j][X_part[i, j]]
                    if idx >= 0:
                        acts.append((idx, 1.0))
            active.append(acts)
        return active

    train_active = precompute(X_train)
    tune_active  = precompute(X_tune)
    full_active  = precompute(X)

    # FM training with given hyperparameters
    def train_fm(active, y, k, lr, l2, epochs, batch, rng):
        w0 = 0.0
        w  = np.zeros(M, dtype=np.float32)
        V  = rng.normal(0, 0.01, (M, k)).astype(np.float32)
        n_samples = len(active)
        for _ in range(epochs):
            order = rng.permutation(n_samples)
            for s in range(0, n_samples, batch):
                idx = order[s:s+batch]
                bsz = len(idx)
                gw0 = 0.0
                gw  = np.zeros(M, dtype=np.float32)
                gV  = np.zeros((M, k), dtype=np.float32)
                for i in idx:
                    acts = active[i]
                    linear = w0
                    sum_v = np.zeros(k, dtype=np.float32)
                    sum_sq_v = np.zeros(k, dtype=np.float32)
                    for fi, val in acts:
                        linear += w[fi] * val
                        vrow = V[fi]
                        for f in range(k):
                            vf = vrow[f] * val
                            sum_v[f] += vf
                            sum_sq_v[f] += vf * vf
                    inter = 0.5 * np.sum(sum_v * sum_v - sum_sq_v)
                    logit = linear + inter
                    p = 1.0 / (1.0 + math.exp(-logit))
                    grad = p - y[i]
                    gw0 += grad
                    for fi, val in acts:
                        gw[fi] += grad * val
                        vrow = V[fi]
                        for f in range(k):
                            gV[fi, f] += grad * val * (sum_v[f] - val * vrow[f])
                # L2 regularization
                gw0 += l2 * w0
                gw  += l2 * w
                gV  += l2 * V
                # Update
                w0 -= lr * gw0 / bsz
                w  -= lr * gw  / bsz
                V  -= lr * gV  / bsz
        return w0, w, V

    def evaluate(active, y, w0, w, V):
        correct = 0
        k = V.shape[1]
        for i, acts in enumerate(active):
            linear = w0
            sum_v = np.zeros(k, dtype=np.float32)
            sum_sq_v = np.zeros(k, dtype=np.float32)
            for fi, val in acts:
                linear += w[fi] * val
                vrow = V[fi]
                for f in range(k):
                    vf = vrow[f] * val
                    sum_v[f] += vf
                    sum_sq_v[f] += vf * vf
            inter = 0.5 * np.sum(sum_v * sum_v - sum_sq_v)
            logit = linear + inter
            pred = 1 if logit > 0 else 0
            if pred == y[i]:
                correct += 1
        return correct / len(active)

    # Grid search on tuning set
    param_combos = []
    for k in GRID['k']:
        for lr in GRID['lr']:
            for l2 in GRID['l2']:
                for ep in GRID['epochs']:
                    for bs in GRID['batch']:
                        param_combos.append((k, lr, l2, ep, bs))

    best_acc = -1.0
    best_params = None
    best_model = None
    for idx, (k, lr, l2, ep, bs) in enumerate(param_combos):
        trial_rng = np.random.default_rng(SEED + idx)
        w0, w, V = train_fm(train_active, y_train, k, lr, l2, ep, bs, trial_rng)
        acc = evaluate(tune_active, y_tune, w0, w, V)
        print(f"k={k} lr={lr} l2={l2} ep={ep} bs={bs} acc={acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc = acc
            best_params = (k, lr, l2, ep, bs)
            best_model = (w0, w, V)

    print(f"Best: {best_params} acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best hyperparameters
    k, lr, l2, ep, bs = best_params
    final_rng = np.random.default_rng(SEED)
    w0, w, V = train_fm(full_active, y, k, lr, l2, ep, bs, final_rng)

    # Write weights.bin
    with open(sys.argv[4], "wb") as f:
        f.write(struct.pack("<I", 0x464D3100))          # magic
        f.write(struct.pack("<i", d))                   # d_raw
        f.write(struct.pack("<i", k))                   # latent dim
        f.write(struct.pack("<i", M))                   # num encoded features
        f.write(struct.pack("<i", 2))                   # classes
        for j, t in enumerate(types):
            if t == "ordinal":
                f.write(struct.pack("<i", 0))           # type ordinal
                f.write(struct.pack("<f", ord_mean[j]))
                f.write(struct.pack("<f", ord_std[j]))
                f.write(struct.pack("<i", ord_feat[j]))
            else:
                f.write(struct.pack("<i", 1))           # type categorical
                f.write(cat_map[j].astype(np.int32).tobytes())
        f.write(struct.pack("<f", w0))
        f.write(w.astype(np.float32).tobytes())
        f.write(V.astype(np.float32).tobytes())

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for tabular label tasks.
 * ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: one byte (class 0/1). */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *weights = slurp(argv[1], &nw);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = weights[0] | weights[1]<<8 | weights[2]<<16 | weights[3]<<24;
    if (magic != 0x464D3100) { fprintf(stderr, "bad magic\n"); return 3; }

    int32_t *hd = (int32_t*)(weights + 4);
    int d = hd[0], k = hd[1], M = hd[2], C = hd[3];
    if (d != 14 || C != 2 || k <= 0 || M <= 0) { fprintf(stderr, "bad header\n"); return 3; }

    unsigned char *ptr = weights + 20;
    long rem = nw - 20;

    int *col_type = malloc(d * sizeof(int));
    float *ord_mean = malloc(d * sizeof(float));
    float *ord_std  = malloc(d * sizeof(float));
    int *ord_feat = malloc(d * sizeof(int));
    int32_t *cat_map = malloc(d * 256 * sizeof(int32_t));

    for (int j = 0; j < d; j++) {
        if (rem < 4) { fprintf(stderr, "truncated col type\n"); return 3; }
        int32_t type = *(int32_t*)ptr; ptr += 4; rem -= 4;
        col_type[j] = type;
        if (type == 0) { /* ordinal */
            if (rem < 12) { fprintf(stderr, "truncated ordinal\n"); return 3; }
            ord_mean[j] = *(float*)ptr; ptr += 4; rem -= 4;
            ord_std[j]  = *(float*)ptr; ptr += 4; rem -= 4;
            ord_feat[j] = *(int32_t*)ptr; ptr += 4; rem -= 4;
        } else if (type == 1) { /* categorical */
            if (rem < 256*4) { fprintf(stderr, "truncated categorical\n"); return 3; }
            memcpy(&cat_map[j*256], ptr, 256*4);
            ptr += 256*4; rem -= 256*4;
        } else { fprintf(stderr, "unknown col type\n"); return 3; }
    }

    if (rem < 4) { fprintf(stderr, "truncated w0\n"); return 3; }
    float w0 = *(float*)ptr; ptr += 4; rem -= 4;

    if (rem < (long)M * 4) { fprintf(stderr, "truncated w\n"); return 3; }
    float *w = (float*)ptr; ptr += M * 4; rem -= M * 4;

    if (rem < (long)M * k * 4) { fprintf(stderr, "truncated V\n"); return 3; }
    float *V = (float*)ptr; ptr += M * k * 4; rem -= M * k * 4;

    float *sum_v = malloc(k * sizeof(float));
    float *sum_sq_v = malloc(k * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }

        float linear = w0;
        for (int f = 0; f < k; f++) { sum_v[f] = 0.0f; sum_sq_v[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char byte = xi[j];
            if (col_type[j] == 0) { /* ordinal */
                int fi = ord_feat[j];
                float val = (byte - ord_mean[j]) / ord_std[j];
                linear += w[fi] * val;
                float *vrow = &V[(long)fi * k];
                for (int f = 0; f < k; f++) {
                    float vf = vrow[f] * val;
                    sum_v[f] += vf;
                    sum_sq_v[f] += vf * vf;
                }
            } else { /* categorical */
                int fi = cat_map[j*256 + byte];
                if (fi >= 0) {
                    linear += w[fi];
                    float *vrow = &V[(long)fi * k];
                    for (int f = 0; f < k; f++) {
                        sum_v[f] += vrow[f];
                        sum_sq_v[f] += vrow[f] * vrow[f];
                    }
                }
            }
        }

        float inter = 0.0f;
        for (int f = 0; f < k; f++) inter += sum_v[f]*sum_v[f] - sum_sq_v[f];
        inter *= 0.5f;
        float logit = linear + inter;

        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }

    free(col_type); free(ord_mean); free(ord_std); free(ord_feat); free(cat_map);
    free(sum_v); free(sum_sq_v);
    free(weights);
    if (x) free(x);
    return 0;
}
```