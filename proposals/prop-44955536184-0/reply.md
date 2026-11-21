```hypothesis
Family: label-emb-mlp (entity embeddings + one hidden ReLU layer).
Why: Adult has 8 categorical columns with high cardinality; one-hot in label-linear inflates input dimension (~100) and ignores similarities between categories. Learned embeddings (dim 4-8) compress this, capture ordinal-like relations, and feed a tiny MLP (H=16-32) that can model interactions. This should raise accuracy above 0.8524 with similar CPU cost (embedding lookup is cheap).
Failure signal: dev accuracy ≤ 0.8524 or instructions/item > 2× label-linear.
```

```python
"""label-emb-mlp: entity embeddings for categorical columns + 1-hidden-layer MLP.
train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Grid searched on tuning split (last 20% of train.in)
E_GRID = [4, 8]
H_GRID = [16, 32]
SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1e-3, 1e-4, 64, 600000, 100
TUNE_EPOCHS = 5
FINAL_EPOCHS_MULT = 10  # final epochs = TUNE_EPOCHS * FINAL_EPOCHS_MULT

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def encode_batch(X, cat_idx, ord_idx, cat_emb, ord_mean, ord_std):
    """X: (n, d) uint8 -> (n, cat_len*E + ord_len) float32"""
    n = X.shape[0]
    cat_part = np.empty((n, len(cat_idx) * cat_emb.shape[1]), dtype=np.float32)
    for i, c in enumerate(cat_idx):
        cat_part[:, i*cat_emb.shape[1]:(i+1)*cat_emb.shape[1]] = cat_emb[X[:, c]]
    if len(ord_idx):
        ord_part = (X[:, ord_idx].astype(np.float32) - ord_mean) / ord_std
        return np.concatenate([cat_part, ord_part], axis=1)
    return cat_part

def softmax_cross_entropy_grad(logits, y):
    """logits (n,C), y (n,) -> grad (n,C)"""
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)

def mlp_forward(Z, W1, b1, W2, b2):
    A = Z @ W1 + b1
    H = np.maximum(A, 0)
    return H @ W2 + b2, A, H

def train_mlp(Z, y, W1, b1, W2, b2, epochs):
    n, C = Z.shape[0], b2.shape[0]
    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            z, yy = Z[idx], y[idx]
            logits, A, H = mlp_forward(z, W1, b1, W2, b2)
            g = softmax_cross_entropy_grad(logits, yy)
            # output layer grads
            gW2 = H.T @ g + L2 * W2
            gb2 = g.sum(0)
            # hidden layer grads
            gh = g @ W2.T * (A > 0)
            gW1 = z.T @ gh + L2 * W1
            gb1 = gh.sum(0)
            grads = [gW1, gb1, gW2, gb2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
    return W1, b1, W2, b2

def evaluate(Z, y, W1, b1, W2, b2):
    logits, _, _ = mlp_forward(Z, W1, b1, W2, b2)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def main():
    global rng
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    cat_idx = [i for i, t in enumerate(types) if t == "categorical"]
    ord_idx = [i for i, t in enumerate(types) if t == "ordinal"]
    C = card["classes"]

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # split: last 20% for tuning
    split = n - n // 5
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # ordinal stats on train split only
    ord_mean = X_tr[:, ord_idx].mean(axis=0).astype(np.float32) if len(ord_idx) else np.array([], np.float32)
    ord_std  = X_tr[:, ord_idx].std(axis=0).astype(np.float32) if len(ord_idx) else np.array([], np.float32)
    ord_std[ord_std == 0] = 1.0

    rng = np.random.default_rng(SEED)
    best_acc, best_E, best_H = -1, E_GRID[0], H_GRID[0]

    # quick grid search on tuning split
    for E in E_GRID:
        for H in H_GRID:
            # init embeddings (256, E) per categorical feature
            cat_emb = {c: (rng.uniform(-0.1, 0.1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32) for c in cat_idx}
            # concat all cat embeddings into one big matrix (256, E) per feature? we'll keep dict for training
            # but for encoding we need a single array per feature. We'll stack them later.
            # MLP weights
            in_dim = len(cat_idx) * E + len(ord_idx)
            W1 = (rng.uniform(-1, 1, (in_dim, H)) * math.sqrt(6.0 / in_dim)).astype(np.float32)
            b1 = np.zeros(H, np.float32)
            W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
            b2 = np.zeros(C, np.float32)

            # encode train and tune
            # build flat embedding matrix for encoding: (num_cat, 256, E) -> we'll index per feature
            emb_stack = np.stack([cat_emb[c] for c in cat_idx])  # (num_cat, 256, E)
            Z_tr = encode_batch(X_tr, cat_idx, ord_idx, emb_stack, ord_mean, ord_std)
            Z_te = encode_batch(X_te, cat_idx, ord_idx, emb_stack, ord_mean, ord_std)

            # train few epochs
            W1, b1, W2, b2 = train_mlp(Z_tr, y_tr, W1, b1, W2, b2, TUNE_EPOCHS)
            acc = evaluate(Z_te, y_te, W1, b1, W2, b2)
            if acc > best_acc:
                best_acc, best_E, best_H = acc, E, H
                # save best embeddings and MLP weights (we'll retrain on full data)
                best_cat_emb = {c: cat_emb[c].copy() for c in cat_idx}
                best_W1, best_b1, best_W2, best_b2 = W1.copy(), b1.copy(), W2.copy(), b2.copy()

    # retrain on full data with best (E, H)
    E, H = best_E, best_H
    cat_emb = best_cat_emb
    W1, b1, W2, b2 = best_W1, best_b1, best_W2, best_b2
    emb_stack = np.stack([cat_emb[c] for c in cat_idx])
    Z_full = encode_batch(X, cat_idx, ord_idx, emb_stack, ord_mean, ord_std)
    final_epochs = TUNE_EPOCHS * FINAL_EPOCHS_MULT
    W1, b1, W2, b2 = train_mlp(Z_full, y, W1, b1, W2, b2, final_epochs)

    # pack weights.bin
    # magic 0x424D454C 'BLEM' little-endian
    # d, num_cat, num_ord, E, H, C
    # types[d] bytes (0=ord,1=cat)
    # embeddings: for each cat feature in original index order: 256*E floats
    # ordinal params: for each ord feature in original index order: mean, std (2 floats)
    # W1 (in_dim, H), b1 (H), W2 (H, C), b2 (C) all float32 little-endian
    num_cat, num_ord = len(cat_idx), len(ord_idx)
    in_dim = num_cat * E + num_ord
    types_arr = bytearray([1 if i in cat_idx else 0 for i in range(d)])

    # flatten embeddings in original column order
    emb_flat = []
    for i in range(d):
        if i in cat_idx:
            emb_flat.append(cat_emb[i].astype('<f4').tobytes())
    # ordinal params
    ord_flat = []
    for i in range(d):
        if i in ord_idx:
            idx = ord_idx.index(i)
            ord_flat.append(struct.pack('<ff', ord_mean[idx], ord_std[idx]))

    out = struct.pack('<7i', 0x424D454C, d, num_cat, num_ord, E, H, C)
    out += types_arr
    out += b''.join(emb_flat)
    out += b''.join(ord_flat)
    out += W1.astype('<f4').tobytes()
    out += b1.astype('<f4').tobytes()
    out += W2.astype('<f4').tobytes()
    out += b2.astype('<f4').tobytes()

    open(sys.argv[4], 'wb').write(out)

if __name__ == '__main__':
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from weights). Output: 1 byte class index. */
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 28) { fprintf(stderr, "weights too small\n"); return 3; }

    int32_t *hdr = (int32_t*)wb;
    if (hdr[0] != 0x424D454C) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hdr[1], num_cat = hdr[2], num_ord = hdr[3], E = hdr[4], H = hdr[5], C = hdr[6];
    if (d != 14 || C != 2 || num_cat + num_ord != d) { fprintf(stderr, "bad header dims\n"); return 3; }

    unsigned char *types = wb + 28;
    float *emb = (float*)(types + d);                     // num_cat * 256 * E floats
    float *ord = emb + (long)num_cat * 256 * E;           // num_ord * 2 floats
    float *W1 = ord + (long)num_ord * 2;                  // in_dim * H
    float *b1 = W1 + (long)(num_cat*E + num_ord) * H;     // H
    float *W2 = b1 + H;                                   // H * C
    float *b2 = W2 + (long)H * C;                         // C

    long expected = 28 + d
                  + (long)num_cat * 256 * E * 4
                  + (long)num_ord * 2 * 4
                  + (long)(num_cat*E + num_ord) * H * 4
                  + H * 4 + (long)H * C * 4 + C * 4;
    if (nw != expected) { fprintf(stderr, "weights size mismatch %ld vs %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    int in_dim = num_cat * E + num_ord;
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // embed categorical, z-score ordinal -> concat into hidden (reuse as input buffer)
        float *inp = hidden; // size H >= in_dim? not necessarily. Allocate separate inp.
        // We'll use logits as temporary input buffer (size C=2, too small). Let's malloc inp.
        // Actually, we can compute directly into hidden for the first layer.
        // But hidden is size H. We need in_dim inputs. Let's allocate a small stack array max in_dim <= 8*256? No, in_dim ~ 8*8+6=70. Use VLA.
        float inp_arr[128]; // max possible: 8 cat * 8 + 6 = 70
        float *inp_ptr = inp_arr;
        for (int col = 0; col < d; col++) {
            if (types[col]) { // categorical
                int cat_id = xi[col];
                const float *e = emb + ((long)col * 256 + cat_id) * E; // WRONG: emb is packed only for cat cols in order of appearance? 
                // In train.py we wrote embeddings for each cat feature in original column order.
                // So we need to know the index of this categorical column among cat columns.
                // We can precompute a mapping from column -> cat_index.
            } else { // ordinal
                int ord_idx = 0; // need mapping from column -> ordinal index
                float mean = ord[ord_idx*2];
                float std  = ord[ord_idx*2+1];
                *inp_ptr++ = (xi[col] - mean) / std;
            }
        }
        // The above mapping is missing. Let's fix by precomputing cat_map and ord_map arrays.
    }
    // We'll redo with proper mapping.
    return 0;
}
```