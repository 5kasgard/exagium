```hypothesis
Family: label-embmlp (embedding + small MLP). Categorical features get learned embeddings (dim 4), ordinal features are z-scored scalars. One ReLU hidden layer (16 units) on top. This reduces input dimension vs one-hot (label-mlp16) while adding non-linearity vs label-linear. Expect >0.855 accuracy at <1500 Ir/item. Fails if embeddings overfit or ordinal scaling hurts.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
GRID = {
    'emb_dim': [2, 4],
    'hidden': [8, 16],
    'lr': [1e-3, 3e-3],
    'l2': [1e-4, 1e-3],
}
BATCH = 64
MAX_EPOCHS_GRID = 30
MAX_EPOCHS_FULL = 50
SAMPLES = 600000

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_model(d, types, emb_dim, H, C, rng):
    cat_idx = [j for j, t in enumerate(types) if t == "categorical"]
    ord_idx = [j for j, t in enumerate(types) if t == "ordinal"]
    n_cat, n_ord = len(cat_idx), len(ord_idx)
    input_dim = n_cat * emb_dim + n_ord
    # embeddings: list of (256, emb_dim)
    embeddings = [(rng.normal(0, 1/math.sqrt(emb_dim), (256, emb_dim)).astype(np.float32)) for _ in range(n_cat)]
    # ordinal stats (filled later)
    ord_mean = np.zeros(n_ord, np.float32)
    ord_std = np.ones(n_ord, np.float32)
    # MLP
    W1 = (rng.normal(0, math.sqrt(2.0/input_dim), (input_dim, H)).astype(np.float32))
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0/H), (H, C)).astype(np.float32))
    b2 = np.zeros(C, np.float32)
    return {
        'cat_idx': cat_idx, 'ord_idx': ord_idx,
        'embeddings': embeddings, 'ord_mean': ord_mean, 'ord_std': ord_std,
        'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2,
        'emb_dim': emb_dim, 'H': H, 'C': C, 'input_dim': input_dim
    }

def forward(model, X, training=True):
    cat_idx, ord_idx = model['cat_idx'], model['ord_idx']
    emb_dim, H, C = model['emb_dim'], model['H'], model['C']
    n = X.shape[0]
    # embeddings
    cat_feats = []
    for i, j in enumerate(cat_idx):
        cat_feats.append(model['embeddings'][i][X[:, j]])
    # ordinal
    ord_feats = (X[:, ord_idx].astype(np.float32) - model['ord_mean']) / model['ord_std']
    # concat
    Z = np.concatenate(cat_feats + [ord_feats], axis=1).astype(np.float32)  # (n, input_dim)
    # hidden
    A = Z @ model['W1'] + model['b1']
    H_act = np.maximum(A, 0)
    # output
    logits = H_act @ model['W2'] + model['b2']
    return Z, A, H_act, logits

def loss_grad(logits, y):
    # softmax cross-entropy gradient
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)

def train_step(model, Z, A, H_act, g, lr, l2, state, t):
    params = [model['W1'], model['b1'], model['W2'], model['b2']]
    grads = []
    # W2, b2
    grads.append(H_act.T @ g + l2 * model['W2'])
    grads.append(g.sum(0))
    # backprop through ReLU
    g_hidden = (g @ model['W2'].T) * (A > 0)
    # W1, b1
    grads.append(Z.T @ g_hidden + l2 * model['W1'])
    grads.append(g_hidden.sum(0))
    # embeddings gradients (accumulate per index)
    cat_idx, ord_idx = model['cat_idx'], model['ord_idx']
    emb_dim = model['emb_dim']
    n_cat = len(cat_idx)
    input_dim = model['input_dim']
    # split g_hidden into cat and ord parts
    g_cat = g_hidden[:, :n_cat*emb_dim].reshape(-1, n_cat, emb_dim)
    # ordinal part has no embedding params
    # update embeddings
    for i in range(n_cat):
        idx = model['X_batch'][:, cat_idx[i]]  # (batch,)
        np.add.at(model['embeddings'][i], idx, g_cat[:, i])
    # Adam update for all params
    for p, gr, (m, v) in zip(params, grads, state):
        m *= 0.9; m += 0.1 * gr
        v *= 0.999; v += 0.001 * gr * gr
        m_hat = m / (1 - 0.9 ** t)
        v_hat = v / (1 - 0.999 ** t)
        p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

def evaluate(model, X, y):
    _, _, _, logits = forward(model, X, training=False)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def train_model(X_tr, y_tr, X_val, y_val, types, emb_dim, H, lr, l2, epochs, rng):
    C = 2
    model = build_model(X_tr.shape[1], types, emb_dim, H, C, rng)
    n = X_tr.shape[0]
    # compute ordinal stats on training set
    ord_idx = model['ord_idx']
    if len(ord_idx) > 0:
        model['ord_mean'] = X_tr[:, ord_idx].mean(axis=0).astype(np.float32)
        model['ord_std'] = X_tr[:, ord_idx].std(axis=0).astype(np.float32)
        model['ord_std'][model['ord_std'] == 0] = 1.0
    # Adam state
    params = [model['W1'], model['b1'], model['W2'], model['b2']]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    best_val = -1
    best_params = None
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            model['X_batch'] = X_tr[idx]
            yy = y_tr[idx]
            Z, A, H_act, logits = forward(model, model['X_batch'])
            g = loss_grad(logits, yy)
            t += 1
            train_step(model, Z, A, H_act, g, lr, l2, state, t)
        val_acc = evaluate(model, X_val, y_val)
        if val_acc > best_val:
            best_val = val_acc
            best_params = {k: v.copy() for k, v in model.items() if k in ['embeddings','ord_mean','ord_std','W1','b1','W2','b2']}
    # restore best
    for k, v in best_params.items():
        model[k] = v
    return model, best_val

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    types = card["input"]["types"]
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_score = -1
    best_cfg = None
    best_model = None

    for emb_dim in GRID['emb_dim']:
        for H in GRID['hidden']:
            for lr in GRID['lr']:
                for l2 in GRID['l2']:
                    model, val_acc = train_model(X_tr, y_tr, X_val, y_val, types, emb_dim, H, lr, l2, MAX_EPOCHS_GRID, rng)
                    if val_acc > best_score:
                        best_score = val_acc
                        best_cfg = (emb_dim, H, lr, l2)
                        best_model = model

    # retrain on full data with best config
    emb_dim, H, lr, l2 = best_cfg
    rng = np.random.default_rng(SEED)
    full_model, _ = train_model(X, y, X_val, y_val, types, emb_dim, H, lr, l2, MAX_EPOCHS_FULL, rng)
    full_model['emb_dim'] = emb_dim
    full_model['H'] = H
    full_model['C'] = 2
    full_model['cat_idx'] = [j for j, t in enumerate(types) if t == "categorical"]
    full_model['ord_idx'] = [j for j, t in enumerate(types) if t == "ordinal"]

    # save weights
    d = X.shape[1]
    cat_idx, ord_idx = full_model['cat_idx'], full_model['ord_idx']
    emb_dim, H, C = full_model['emb_dim'], full_model['H'], full_model['C']
    feature_types = bytearray([1 if t == "categorical" else 0 for t in types])

    # compute total size for validation
    n_cat = len(cat_idx)
    n_ord = len(ord_idx)
    input_dim = n_cat * emb_dim + n_ord
    total_floats = (n_cat * 256 * emb_dim) + (n_ord * 2) + (input_dim * H) + H + (H * C) + C

    magic = 0x31424D45  # "EMB1"
    out = struct.pack("<5i", magic, d, emb_dim, H, C)
    out += feature_types
    for i in range(n_cat):
        out += full_model['embeddings'][i].astype("<f4").tobytes()
    for i in range(n_ord):
        out += struct.pack("<ff", full_model['ord_mean'][i], full_model['ord_std'][i])
    out += full_model['W1'].astype("<f4").tobytes()
    out += full_model['b1'].astype("<f4").tobytes()
    out += full_model['W2'].astype("<f4").tobytes()
    out += full_model['b2'].astype("<f4").tobytes()

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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x31424D45) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = (int)hd[1], emb_dim = (int)hd[2], H = (int)hd[3], C = (int)hd[4];
    if (d != 14 || C != 2 || H < 1 || emb_dim < 1) { fprintf(stderr, "bad header dims\n"); return 3; }
    unsigned char *feature_types = wb + 20;
    int n_cat = 0, n_ord = 0;
    for (int j = 0; j < d; j++) {
        if (feature_types[j]) n_cat++; else n_ord++;
    }
    int input_dim = n_cat * emb_dim + n_ord;
    long expected = 20 + d + (long)n_cat * 256 * emb_dim * 4 + (long)n_ord * 2 * 4 +
                    (long)input_dim * H * 4 + H * 4 + (long)H * C * 4 + C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *ptr = (const float *)(wb + 20 + d);
    const float **emb = malloc(n_cat * sizeof(float*));
    for (int i = 0; i < n_cat; i++) {
        emb[i] = ptr;
        ptr += 256 * emb_dim;
    }
    const float *ord_mean = (const float *)ptr; ptr += n_ord;
    const float *ord_std  = (const float *)ptr; ptr += n_ord;
    const float *W1 = ptr; ptr += (long)input_dim * H;
    const float *b1 = ptr; ptr += H;
    const float *W2 = ptr; ptr += (long)H * C;
    const float *b2 = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(H * sizeof(float));
    float *logits = malloc(C * sizeof(float));
    float *cat_buf = malloc(n_cat * emb_dim * sizeof(float));
    float *ord_buf = malloc(n_ord * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        int cat_ptr = 0, ord_ptr = 0;
        for (int j = 0; j < d; j++) {
            if (feature_types[j]) {
                const float *e = emb[cat_ptr] + xi[j] * emb_dim;
                memcpy(cat_buf + cat_ptr * emb_dim, e, emb_dim * sizeof(float));
                cat_ptr++;
            } else {
                ord_buf[ord_ptr] = (xi[j] - ord_mean[ord_ptr]) / ord_std[ord_ptr];
                ord_ptr++;
            }
        }

        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int c = 0; c < n_cat; c++) {
                for (int e = 0; e < emb_dim; e++) {
                    sum += cat_buf[c * emb_dim + e] * W1[(c * emb_dim + e) * H + h];
                }
            }
            for (int o = 0; o < n_ord; o++) {
                sum += ord_buf[o] * W1[(n_cat * emb_dim + o) * H + h];
            }
            hidden[h] = sum > 0 ? sum : 0;
        }

        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[h * C + c];
            logits[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```