```hypothesis
Family: label-emblin (embedding + linear). Categorical features get learned embeddings (dim=8), ordinal features are z-scored, then a single linear layer predicts the class. This adds minimal non-linearity via embeddings and should beat linear on this mixed-type tabular data while staying fast and small. If dev accuracy <= 0.8524 or latency >= label-linear, the hypothesis fails.
```

```python
"""label-emblin: embeddings for categorical features + linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy+stdlib only.
Weights format (little-endian):
  int32 magic=0x454D424C, d, E, C
  uint8 types[d] (1=categorical, 0=ordinal)
  For each feature j:
    if categorical: float32 emb[256][E]
    if ordinal:     float32 mean, std
  float32 W[total_dim][C], b[C]   (total_dim = num_cat*E + num_ord)
"""
import json, math, struct, sys

import numpy as np

E = 8                 # embedding dimension
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    types = [1 if t == "categorical" else 0 for t in card["input"]["types"]]
    cat_idx = [j for j, t in enumerate(types) if t == 1]
    ord_idx = [j for j, t in enumerate(types) if t == 0]
    n_cat = len(cat_idx)
    n_ord = len(ord_idx)
    total_dim = n_cat * E + n_ord

    rng = np.random.default_rng(SEED)

    # Initialize embeddings for categorical features
    emb = [rng.normal(0, 0.01, (256, E)).astype(np.float32) for _ in range(n_cat)]
    # Initialize ordinal scalers
    ord_mean = np.zeros(n_ord, np.float32)
    ord_std = np.ones(n_ord, np.float32)
    for oi, j in enumerate(ord_idx):
        v = X[:, j].astype(np.float64)
        ord_mean[oi] = v.mean()
        ord_std[oi] = v.std() or 1.0

    # Linear weights
    W = rng.normal(0, math.sqrt(2.0 / total_dim), (total_dim, C)).astype(np.float32)
    b = np.zeros(C, np.float32)

    # Adam state
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    mEmb = [np.zeros_like(e) for e in emb]
    vEmb = [np.zeros_like(e) for e in emb]

    def forward(batch_idx):
        # Build input matrix for batch
        Z = np.zeros((len(batch_idx), total_dim), np.float32)
        col = 0
        for ci, j in enumerate(cat_idx):
            Z[:, col:col+E] = emb[ci][X[batch_idx, j]]
            col += E
        for oi, j in enumerate(ord_idx):
            Z[:, col] = (X[batch_idx, j] - ord_mean[oi]) / ord_std[oi]
            col += 1
        logits = Z @ W + b
        return Z, logits

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return probs.astype(np.float32) / len(yy)

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / X.shape[0]))):
        order = rng.permutation(X.shape[0])
        for s in range(0, X.shape[0], BATCH):
            idx = order[s:s+BATCH]
            Z, logits = forward(idx)
            g = softmax_grad(logits, y[idx])

            # Gradients
            grad_W = Z.T @ g + L2 * W
            grad_b = g.sum(axis=0)
            grad_emb = [np.zeros_like(e) for e in emb]
            col = 0
            for ci, j in enumerate(cat_idx):
                # g_emb = Z_grad for embedding part: for each example, gradient flows to the embedding row
                # We accumulate gradient for each embedding row
                g_emb = g[:, col:col+E]  # (batch, E)
                rows = X[idx, j]         # (batch,)
                np.add.at(grad_emb[ci], rows, g_emb)
                grad_emb[ci] += L2 * emb[ci]
                col += E

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            # W, b
            mW = beta1 * mW + (1-beta1) * grad_W
            vW = beta2 * vW + (1-beta2) * grad_W * grad_W
            W -= LR * (mW / (1 - beta1**t)) / (np.sqrt(vW / (1 - beta2**t)) + eps)
            mb = beta1 * mb + (1-beta1) * grad_b
            vb = beta2 * vb + (1-beta2) * grad_b * grad_b
            b -= LR * (mb / (1 - beta1**t)) / (np.sqrt(vb / (1 - beta2**t)) + eps)
            # embeddings
            for ci in range(n_cat):
                mEmb[ci] = beta1 * mEmb[ci] + (1-beta1) * grad_emb[ci]
                vEmb[ci] = beta2 * vEmb[ci] + (1-beta2) * grad_emb[ci] * grad_emb[ci]
                emb[ci] -= LR * (mEmb[ci] / (1 - beta1**t)) / (np.sqrt(vEmb[ci] / (1 - beta2**t)) + eps)

    # Write weights
    magic = 0x454D424C  # 'EMBL'
    out = struct.pack("<4i", magic, d, E, C)
    out += bytes(types)
    for ci in range(n_cat):
        out += emb[ci].astype("<f4").tobytes()
    for oi in range(n_ord):
        out += struct.pack("<2f", ord_mean[oi], ord_std[oi])
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-emblin inference: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: one byte class index (argmax). */
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
    int32_t *hd = (int32_t*)wb;
    if (hd[0] != 0x454D424C) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 14 || C != 2 || E <= 0 || E > 32) { fprintf(stderr, "bad header\n"); return 3; }
    unsigned char *types = (unsigned char*)(hd + 4);
    int n_cat = 0, n_ord = 0;
    for (int j=0;j<d;j++) { if (types[j]) n_cat++; else n_ord++; }
    int total_dim = n_cat * E + n_ord;
    long expected = 16L + d + 4L*(256L * E * n_cat + 2L * n_ord + (long)total_dim * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld vs %ld\n", nw, expected); return 3; }

    const float *ptr = (const float*)(types + d);
    const float **emb = malloc(n_cat * sizeof(float*));
    for (int ci=0; ci<n_cat; ci++) {
        emb[ci] = ptr;
        ptr += 256 * E;
    }
    const float *ord_mean = (const float*)ptr; ptr += n_ord;
    const float *ord_std  = (const float*)ptr; ptr += n_ord;
    const float *W = ptr; ptr += (long)total_dim * C;
    const float *b = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(total_dim * sizeof(float));
    float *scores = malloc(C * sizeof(float));

    for (long i=0;i<N;i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        int col = 0;
        int ci = 0, oi = 0;
        for (int j=0;j<d;j++) {
            if (types[j]) {
                const float *e = emb[ci] + (long)xi[j] * E;
                for (int u=0;u<E;u++) acc[col++] = e[u];
                ci++;
            } else {
                acc[col++] = (xi[j] - ord_mean[oi]) / ord_std[oi];
                oi++;
            }
        }
        for (int c=0;c<C;c++) scores[c] = b[c];
        for (int c=0;c<C;c++) {
            float sum = 0.0f;
            for (int v=0;v<total_dim;v++) sum += acc[v] * W[(long)v * C + c];
            scores[c] += sum;
        }
        unsigned char best = 0;
        for (int c=1;c<C;c++) if (scores[c] > scores[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```