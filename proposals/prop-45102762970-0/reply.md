```hypothesis
Family: label-emb-linear (embedding + linear). Categorical columns get 4-d embeddings (256×4 each), ordinal columns are z-scored scalars. A single linear layer on the concatenated vector (9×4 + 5 = 41 inputs) predicts 2 classes. This captures category similarities with far fewer parameters than one-hot, should beat label-linear (0.8524) on accuracy while staying near its 719 Ir/item cost. Failure: dev accuracy ≤ 0.8524 or Ir/item > 1000.
```
```python
"""label-emb-linear: embedding + linear for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format (little-endian):
  int32 magic=0x454D424C, d, E, C, n_cat, n_ord
  uint8 type[d] (0=ordinal,1=categorical)
  for each column: if cat: 256*E float32; else: 2 float32 (mean,std)
  float32 W[D][C], b[C]  (D = n_cat*E + n_ord)"""
import json, math, struct, sys

import numpy as np

E = 4                # embedding dim
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-3, 1e-4, 64, 600000, 100

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
    types = card["input"]["types"]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    cat_cols = [j for j, t in enumerate(types) if t == "categorical"]
    ord_cols = [j for j, t in enumerate(types) if t == "ordinal"]
    n_cat, n_ord = len(cat_cols), len(ord_cols)
    D = n_cat * E + n_ord

    # ordinal stats (fixed)
    ord_mean = X[:, ord_cols].mean(axis=0).astype(np.float32)
    ord_std  = X[:, ord_cols].std(axis=0).astype(np.float32)
    ord_std[ord_std == 0] = 1.0
    X_ord = ((X[:, ord_cols].astype(np.float32) - ord_mean) / ord_std).astype(np.float32)  # (n, n_ord)

    # categorical indices
    X_cat = X[:, cat_cols].astype(np.int32)  # (n, n_cat)

    # parameters
    emb = [rng.uniform(-0.1, 0.1, (256, E)).astype(np.float32) for _ in range(n_cat)]
    W = rng.uniform(-0.1, 0.1, (D, C)).astype(np.float32)
    b = np.zeros(C, np.float32)

    # Adam state
    params = emb + [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            # build features
            feats = np.zeros((len(idx), D), np.float32)
            feats[:, n_cat*E:] = X_ord[idx]
            for cj, col in enumerate(cat_cols):
                feats[:, cj*E:(cj+1)*E] = emb[cj][X_cat[idx, cj]]
            # forward
            scores = feats @ W + b
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(len(idx)), y[idx]] -= 1.0
            grad_scores = probs / len(idx)
            # gradients
            grad_W = feats.T @ grad_scores + L2 * W
            grad_b = grad_scores.sum(axis=0)
            grad_feats = grad_scores @ W.T
            # split grad_feats
            grad_emb = [np.zeros_like(e) for e in emb]
            for cj in range(n_cat):
                g = grad_feats[:, cj*E:(cj+1)*E]
                idx_cat = X_cat[idx, cj]
                np.add.at(grad_emb[cj], idx_cat, g)
                grad_emb[cj] += L2 * emb[cj]
            # update
            grads = grad_emb + [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights
    magic = 0x454D424C  # "LBME"
    type_arr = bytearray(1 if t == "categorical" else 0 for t in types)
    out = struct.pack("<6i", magic, d, E, C, n_cat, n_ord)
    out += type_arr
    for cj in range(n_cat):
        out += emb[cj].astype("<f4").tobytes()
    for oj in range(n_ord):
        out += struct.pack("<2f", ord_mean[oj], ord_std[oj])
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-emb-linear inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from header). Output: 1 byte class per record. */
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
    if (nw < 24) { fprintf(stderr, "weights too small\n"); return 3; }
    int32_t *hdr = (int32_t *)wb;
    if (hdr[0] != 0x454D424C) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hdr[1], E = hdr[2], C = hdr[3], n_cat = hdr[4], n_ord = hdr[5];
    if (d <= 0 || C <= 0 || C > 256 || E <= 0) { fprintf(stderr, "bad header dims\n"); return 3; }
    unsigned char *type = wb + 24;
    if (nw < 24 + d) { fprintf(stderr, "truncated type array\n"); return 3; }
    const float *p = (const float *)(wb + 24 + d);
    long expected = 24 + d;
    for (int j = 0; j < d; j++) {
        if (type[j]) expected += 256L * E * 4;
        else expected += 2 * 4;
    }
    int D = n_cat * E + n_ord;
    expected += (long)D * C * 4 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    /* pointers */
    const float **cat_emb = malloc(n_cat * sizeof(float *));
    int cat_idx = 0;
    const float *ord_mean = NULL, *ord_std = NULL;
    for (int j = 0; j < d; j++) {
        if (type[j]) {
            cat_emb[cat_idx++] = p;
            p += 256 * E;
        } else {
            if (!ord_mean) { ord_mean = p; p += n_ord; ord_std = p; p += n_ord; }
            /* we only need the first n_ord means/stds; they are stored per ordinal column in order */
        }
    }
    const float *W = p; p += (long)D * C;
    const float *b = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = malloc(D * sizeof(float));
    float *scores = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int f = 0;
        int cat_ptr = 0, ord_ptr = 0;
        for (int j = 0; j < d; j++) {
            if (type[j]) {
                const float *emb = cat_emb[cat_ptr++] + (long)xi[j] * E;
                for (int e = 0; e < E; e++) feat[f++] = emb[e];
            } else {
                float v = (float)xi[j];
                feat[f++] = (v - ord_mean[ord_ptr]) / ord_std[ord_ptr];
                ord_ptr++;
            }
        }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < D; j++) {
            float fv = feat[j];
            for (int c = 0; c < C; c++) scores[c] += fv * W[(long)j * C + c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```