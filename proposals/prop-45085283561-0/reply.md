```hypothesis
Family: label-emblin (embedding + linear). Categorical features get learned embeddings (size 4), ordinal features are z-scored; a single linear layer on the concatenated vector predicts the class. This reduces input dimension from ~108 (one-hot) to 38, capturing category similarities and lowering inference cost. Expect higher accuracy than label-linear (0.8524) with lower ns/item. Failure: dev accuracy ≤ 0.8524 or ns/item ≥ label-linear.
```
```python
"""label-emblin: embedding + linear for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format: magic 0x454D424C, d, E, C, feature_types[d],
  per ordinal: mean, std (float32),
  per categorical: 256*E float32 embeddings (in feature order),
  linear weight [D, C] float32 (row-major), linear bias [C] float32.
D = num_ordinal + num_categorical * E."""
import json, math, struct, sys

import numpy as np

E = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


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
    C = card["classes"]
    types = card["input"]["types"]
    d = len(types)

    rng = np.random.default_rng(SEED)

    # Determine ordinal stats and categorical counts
    ordinal_mean = []
    ordinal_std = []
    cat_features = []
    ord_features = []
    for j, t in enumerate(types):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            ordinal_mean.append(mu)
            ordinal_std.append(sd)
            ord_features.append(j)
        else:
            cat_features.append(j)

    n_cat = len(cat_features)
    n_ord = len(ord_features)
    D = n_ord + n_cat * E

    # Initialize parameters
    embeddings = [rng.normal(0, 0.01, (256, E)).astype(np.float32) for _ in range(n_cat)]
    W = np.zeros((D, C), dtype=np.float32)
    b = np.zeros(C, dtype=np.float32)

    # Adam state
    params = embeddings + [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # Precompute normalized ordinal features
    X_ord = np.zeros((X.shape[0], n_ord), dtype=np.float32)
    for idx, j in enumerate(ord_features):
        X_ord[:, idx] = (X[:, j].astype(np.float32) - ordinal_mean[idx]) / ordinal_std[idx]

    # Training loop
    t = 0
    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            batch_size = len(idx)
            # Build concatenated feature matrix Z [batch, D]
            Z = np.zeros((batch_size, D), dtype=np.float32)
            Z[:, :n_ord] = X_ord[idx]
            # Embeddings
            off = n_ord
            for ci, j in enumerate(cat_features):
                Z[:, off:off+E] = embeddings[ci][X[idx, j]]
                off += E

            # Forward
            logits = Z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(batch_size), y[idx]] -= 1.0
            grad_logits = probs / batch_size

            # Gradients
            grad_W = Z.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
            grad_Z = grad_logits @ W.T

            # Embedding gradients
            grad_emb = [np.zeros_like(e) for e in embeddings]
            off = n_ord
            for ci, j in enumerate(cat_features):
                g = grad_Z[:, off:off+E]
                np.add.at(grad_emb[ci], X[idx, j], g)
                off += E

            # Adam update
            all_grads = grad_emb + [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, all_grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    magic = 0x454D424C
    feature_types = bytearray(1 if t == "categorical" else 0 for t in types)
    with open(sys.argv[4], "wb") as f:
        f.write(struct.pack("<I3i", magic, d, E, C))
        f.write(feature_types)
        for j in range(d):
            if types[j] == "ordinal":
                f.write(struct.pack("<2f", ordinal_mean[ord_features.index(j)], ordinal_std[ord_features.index(j)]))
        for e in embeddings:
            f.write(e.astype("<f4").tobytes())
        f.write(W.astype("<f4").tobytes())
        f.write(b.astype("<f4").tobytes())


if __name__ == "__main__":
    main()
```
```c
/* label-emblin infer: ./infer weights.bin in.rec N */
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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hdr = (uint32_t *)wb;
    if (hdr[0] != 0x454D424C) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = (int)hdr[1], E = (int)hdr[2], C = (int)hdr[3];
    if (d != 14 || C != 2 || E != 4) { fprintf(stderr, "unexpected dims\n"); return 3; }
    unsigned char *ft = wb + 16;
    float *ptr = (float *)(ft + d);
    float ord_mean[14], ord_std[14];
    float *cat_emb[14] = {0};
    int n_ord = 0, n_cat = 0;
    for (int j = 0; j < d; j++) {
        if (ft[j] == 0) {
            ord_mean[j] = ptr[0]; ord_std[j] = ptr[1]; ptr += 2;
            n_ord++;
        } else {
            cat_emb[j] = ptr; ptr += 256 * E;
            n_cat++;
        }
    }
    int D = n_ord + n_cat * E;
    if (nw != 16 + d + 4L*(2*n_ord + 256*E*n_cat + D*C + C)) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    float *W = ptr; ptr += D * C;
    float *b = ptr;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(D * sizeof(float));
    float *scores = malloc(C * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int off = 0;
        for (int j = 0; j < d; j++) {
            if (ft[j] == 0) {
                z[off++] = (xi[j] - ord_mean[j]) / ord_std[j];
            } else {
                memcpy(z + off, cat_emb[j] + xi[j] * E, E * sizeof(float));
                off += E;
            }
        }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < D; j++) {
            float zj = z[j];
            for (int c = 0; c < C; c++) scores[c] += zj * W[j*C + c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```