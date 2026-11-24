```hypothesis
Family: label-embmlp (embedding + one hidden layer MLP). 
Ordinal columns are z-scored; categorical columns get learned embeddings (dim 8). 
A single ReLU layer (16 units) on top of the concatenated vector gives a better 
capacity/footprint trade-off than label-linear. Expect dev accuracy ≥ 0.860 with 
inference cost < 2× label-linear. If dev accuracy ≤ 0.852 or Ir/item > 1500, 
the hypothesis fails.
```
```python
"""label-embmlp: ordinal z-scored + categorical embeddings (E=8) + one ReLU layer (H=16).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

E = 8          # embedding dimension for categorical columns
H = 16         # hidden layer size
SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x454D4244  # 'DBME' little-endian

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
    d = card["input"]["bytes"]
    C = card["classes"]
    types = card["input"]["types"]          # list of "ordinal"/"categorical"

    ord_idx = [i for i, t in enumerate(types) if t == "ordinal"]
    cat_idx = [i for i, t in enumerate(types) if t == "categorical"]
    n_ord, n_cat = len(ord_idx), len(cat_idx)
    D = n_ord + n_cat * E                   # input dimension to MLP

    # ordinal statistics
    X_ord = X[:, ord_idx].astype(np.float64)
    ord_mean = X_ord.mean(axis=0)
    ord_std = X_ord.std(axis=0)
    ord_std[ord_std == 0] = 1.0
    X_ord = (X_ord - ord_mean) / ord_std     # (n, n_ord)

    # categorical indices (0..n_cat-1 per column)
    X_cat = X[:, cat_idx]                    # (n, n_cat) uint8

    rng = np.random.default_rng(SEED)

    # parameters
    cat_emb = (rng.normal(0, 0.01, (n_cat * 256, E))).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (D, H)) * math.sqrt(6.0 / D)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [cat_emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x_ord = X_ord[idx].astype(np.float32)          # (b, n_ord)
            x_cat = X_cat[idx]                              # (b, n_cat)
            yy = y[idx]

            # forward
            # embeddings: (b, n_cat, E)
            emb = cat_emb[x_cat.reshape(-1)].reshape(-1, n_cat, E)
            flat = np.concatenate([x_ord, emb.reshape(len(idx), -1)], axis=1)  # (b, D)
            hidden = np.maximum(flat @ W1 + b1, 0)
            logits = hidden @ W2 + b2

            # softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            dlogits = (probs / len(yy)).astype(np.float32)

            # backward
            dhidden = dlogits @ W2.T
            dhidden[hidden == 0] = 0

            dW2 = hidden.T @ dlogits + L2 * W2
            db2 = dlogits.sum(axis=0)
            dflat = dhidden @ W1.T
            dW1 = flat.T @ dhidden + L2 * W1
            db1 = dhidden.sum(axis=0)

            # embedding gradients
            dflat_cat = dflat[:, n_ord:].reshape(-1, n_cat, E)
            dcat_emb = np.zeros_like(cat_emb)
            for c in range(n_cat):
                np.add.at(dcat_emb, x_cat[:, c] + c * 256, dflat_cat[:, c])

            grads = [dcat_emb, dW1, db1, dW2, db2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights.bin
    # header: magic, d, n_ord, n_cat, E, H, C, then d bytes of types (0=ord,1=cat)
    type_bytes = bytes([0 if t == "ordinal" else 1 for t in types])
    header = struct.pack("<iiiiiii", MAGIC, d, n_ord, n_cat, E, H, C) + type_bytes

    # floats: ord_mean, ord_std, cat_emb, W1, b1, W2, b2
    floats = []
    floats.append(ord_mean.astype(np.float32))
    floats.append(ord_std.astype(np.float32))
    floats.append(cat_emb.ravel())
    floats.append(W1.ravel())
    floats.append(b1)
    floats.append(W2.ravel())
    floats.append(b2)
    blob = b"".join(arr.tobytes() for arr in floats)

    open(sys.argv[4], "wb").write(header + blob)

if __name__ == "__main__":
    main()
```
```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class per record. */
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
    if (nw < 32) { fprintf(stderr, "weights too small\n"); return 3; }

    // parse header
    int32_t *hdr = (int32_t *)wb;
    if (hdr[0] != 0x454D4244) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hdr[1], n_ord = hdr[2], n_cat = hdr[3], E = hdr[4], H = hdr[5], C = hdr[6];
    if (d != 14 || C != 2 || n_ord + n_cat != d || E != 8 || H != 16) {
        fprintf(stderr, "unexpected header values\n"); return 3;
    }
    unsigned char *types = (unsigned char *)(hdr + 7);  // d bytes
    if (nw < 28 + d) { fprintf(stderr, "header truncated\n"); return 3; }

    // pointers into float region
    float *fp = (float *)(wb + 28 + d);
    float *ord_mean = fp; fp += n_ord;
    float *ord_std  = fp; fp += n_ord;
    float *cat_emb  = fp; fp += (long)n_cat * 256 * E;
    float *W1       = fp; fp += (long)(n_ord + n_cat * E) * H;
    float *b1       = fp; fp += H;
    float *W2       = fp; fp += (long)H * C;
    float *b2       = fp; fp += C;

    // validate total size
    long expected = 28 + d + (long)(2 * n_ord + n_cat * 256 * E + (n_ord + n_cat * E) * H + H + H * C + C) * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    // map column -> categorical index (0..n_cat-1) or -1
    int cat_map[14];
    int cidx = 0;
    for (int j = 0; j < d; j++) {
        if (types[j] == 1) cat_map[j] = cidx++;
        else cat_map[j] = -1;
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *in_vec = malloc(sizeof(float) * (n_ord + n_cat * E));
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // build input vector
        int o = 0, c = 0;
        for (int j = 0; j < d; j++) {
            if (types[j] == 0) { // ordinal
                in_vec[o++] = (xi[j] - ord_mean[o]) / ord_std[o];
            } else { // categorical
                const float *emb = cat_emb + ((long)cat_map[j] * 256 + xi[j]) * E;
                for (int e = 0; e < E; e++) in_vec[n_ord + c * E + e] = emb[e];
                c++;
            }
        }

        // hidden = ReLU(in_vec @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int v = 0; v < n_ord + n_cat * E; v++) sum += in_vec[v] * W1[(long)v * H + h];
            hidden[h] = sum > 0 ? sum : 0;
        }

        // logits = hidden @ W2 + b2
        for (int cl = 0; cl < C; cl++) {
            float sum = b2[cl];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + cl];
            logits[cl] = sum;
        }

        // argmax
        unsigned char best = 0;
        for (int cl = 1; cl < C; cl++) if (logits[cl] > logits[best]) best = cl;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```