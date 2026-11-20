```hypothesis
Family: label-linear-ordinal-interactions (LIP2). Adds pairwise products of z-scored ordinal features to a linear model. Ordinal features (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week) likely have non-linear effects; interactions capture this with only 15 extra weights per class. Inference cost rises marginally (6 z-scores + 15 products + 15*2 MACs). If dev accuracy does not exceed label-linear (0.8524) or cost exceeds label-mlp16 (2325 Ir/item), the approach fails.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
LR_GRID = [1e-3, 3e-3]
L2_GRID = [1e-4, 1e-3]
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def encode_data(X, types, enc_params=None):
    n, d = X.shape
    if enc_params is None:
        means, stds, cat_vals = [], [], []
        ordinal_indices = []
        ordinal_feature_indices = []
        enc_mats = []
        off = 0
        for j, t in enumerate(types):
            if t == "ordinal":
                v = X[:, j].astype(np.float64)
                mu, sd = v.mean(), v.std() or 1.0
                means.append(mu)
                stds.append(sd)
                ordinal_indices.append(j)
                ordinal_feature_indices.append(off)
                E = ((np.arange(256) - mu) / sd)[:, None].astype(np.float32)
                enc_mats.append(E)
                off += 1
                cat_vals.append(None)
            else:
                vals = np.unique(X[:, j])
                cat_vals.append(vals)
                E = np.zeros((256, len(vals)), np.float32)
                E[vals, np.arange(len(vals))] = 1.0
                enc_mats.append(E)
                off += len(vals)
                ordinal_indices.append(-1)
        enc_params = (means, stds, cat_vals, ordinal_indices, ordinal_feature_indices)
    else:
        means, stds, cat_vals, ordinal_indices, ordinal_feature_indices = enc_params
        enc_mats = []
        for j, t in enumerate(types):
            if t == "ordinal":
                idx = ordinal_indices.index(j)
                mu, sd = means[idx], stds[idx]
                E = ((np.arange(256) - mu) / sd)[:, None].astype(np.float32)
                enc_mats.append(E)
            else:
                vals = cat_vals[j]
                E = np.zeros((256, len(vals)), np.float32)
                E[vals, np.arange(len(vals))] = 1.0
                enc_mats.append(E)
    Z_main = np.concatenate([enc_mats[j][X[:, j]] for j in range(d)], axis=1).astype(np.float32)
    num_ordinal = len(ordinal_feature_indices)
    num_interactions = num_ordinal * (num_ordinal - 1) // 2
    Z_inter = np.zeros((n, num_interactions), np.float32)
    idx = 0
    for i in range(num_ordinal):
        for j in range(i+1, num_ordinal):
            col_i = ordinal_feature_indices[i]
            col_j = ordinal_feature_indices[j]
            Z_inter[:, idx] = Z_main[:, col_i] * Z_main[:, col_j]
            idx += 1
    return Z_main, Z_inter, enc_params

def train_linear(Z, y, C, lr, l2, batch, samples, max_epochs, rng):
    n, k = Z.shape
    W = np.zeros((k, C), np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(max_epochs, math.ceil(samples / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            z, yy = Z[idx], y[idx]
            scores = z @ W + b
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad = probs / len(yy)
            gW = z.T @ grad + l2 * W
            gb = grad.sum(axis=0)
            grads = [gW, gb]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return W, b

def evaluate(Z, y, W, b):
    scores = Z @ W + b
    preds = scores.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    types = card["input"]["types"]
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]
    
    best_acc = -1
    best_hyper = None
    best_enc_params = None
    best_W = None
    best_b = None
    
    for lr in LR_GRID:
        for l2 in L2_GRID:
            rng = np.random.default_rng(SEED)
            Z_main_tr, Z_inter_tr, enc_params = encode_data(X_train, types)
            Z_tr = np.concatenate([Z_main_tr, Z_inter_tr], axis=1)
            W, b = train_linear(Z_tr, y_train, C, lr, l2, BATCH, SAMPLES, MAX_EPOCHS, rng)
            Z_main_tune, Z_inter_tune, _ = encode_data(X_tune, types, enc_params)
            Z_tune = np.concatenate([Z_main_tune, Z_inter_tune], axis=1)
            acc = evaluate(Z_tune, y_tune, W, b)
            if acc > best_acc:
                best_acc = acc
                best_hyper = (lr, l2)
                best_enc_params = enc_params
                best_W = W.copy()
                best_b = b.copy()
    
    lr, l2 = best_hyper
    rng = np.random.default_rng(SEED)
    Z_main_all, Z_inter_all, enc_params = encode_data(X, types)
    Z_all = np.concatenate([Z_main_all, Z_inter_all], axis=1)
    W, b = train_linear(Z_all, y, C, lr, l2, BATCH, SAMPLES, MAX_EPOCHS, rng)
    
    W_main = W[:Z_main_all.shape[1]]
    W_inter = W[Z_main_all.shape[1]:]
    
    means, stds, cat_vals, ordinal_indices, ordinal_feature_indices = enc_params
    enc_mats = []
    off = 0
    for j, t in enumerate(types):
        if t == "ordinal":
            idx = ordinal_indices.index(j)
            mu, sd = means[idx], stds[idx]
            E = ((np.arange(256) - mu) / sd)[:, None].astype(np.float32)
            enc_mats.append(E)
            off += 1
        else:
            vals = cat_vals[j]
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc_mats.append(E)
            off += len(vals)
    T = []
    off = 0
    for j in range(d):
        E = enc_mats[j]
        num_feat = E.shape[1]
        Tj = E @ W_main[off:off+num_feat]
        T.append(Tj)
        off += num_feat
    T = np.stack(T, axis=0).astype(np.float32)
    
    ordinal_indices_sorted = [idx for idx in ordinal_indices if idx != -1]
    num_ordinal = len(ordinal_indices_sorted)
    num_interactions = num_ordinal * (num_ordinal - 1) // 2
    
    magic = 0x4C495032
    header = struct.pack("<6i", magic, d, 0, C, W_main.shape[1], num_ordinal)
    header += struct.pack(f"<{num_ordinal}i", *ordinal_indices_sorted)
    header += struct.pack(f"<{num_ordinal}f", *means)
    header += struct.pack(f"<{num_ordinal}f", *stds)
    header += T.tobytes()
    header += b.astype(np.float32).tobytes()
    header += W_inter.astype(np.float32).tobytes()
    
    open(sys.argv[4], "wb").write(header)

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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 24 || hd[0] != 0x4C495032) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W_main = hd[4], num_ordinal = hd[5];
    if (H != 0 || C < 1 || C > 256 || num_ordinal < 0 || num_ordinal > d) { fprintf(stderr, "bad weights header\n"); return 3; }
    int32_t *ordinal_indices = (int32_t *)(hd + 6);
    float *means = (float *)(ordinal_indices + num_ordinal);
    float *stds = means + num_ordinal;
    float *T = (float *)(stds + num_ordinal);
    float *b_main = T + (long)d * 256 * C;
    float *W_inter = b_main + C;
    int num_interactions = num_ordinal * (num_ordinal - 1) / 2;
    long expected_size = 24 + 4*num_ordinal + 4*num_ordinal + 4*num_ordinal + 4L*d*256*C + 4*C + 4L*num_interactions*C;
    if (nw != expected_size) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected_size); return 3; }
    
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *z_ordinal = malloc(sizeof(float) * num_ordinal);
    
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = b_main[c];
        for (int j = 0; j < d; j++) {
            const float *tj = T + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += tj[c];
        }
        for (int o = 0; o < num_ordinal; o++) {
            int col = ordinal_indices[o];
            unsigned char byte = xi[col];
            z_ordinal[o] = (byte - means[o]) / stds[o];
        }
        int inter_idx = 0;
        for (int o1 = 0; o1 < num_ordinal; o1++) {
            for (int o2 = o1 + 1; o2 < num_ordinal; o2++) {
                float inter_val = z_ordinal[o1] * z_ordinal[o2];
                const float *w_inter = W_inter + (long)inter_idx * C;
                for (int c = 0; c < C; c++) scores[c] += inter_val * w_inter[c];
                inter_idx++;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```