```hypothesis
Family: label-emb-linear (entity embeddings + linear). Categorical columns get learned embeddings (dim 8), ordinal columns are z-scored; all concatenated into a linear classifier. This reduces dimensionality vs one-hot, captures category similarity, and stays extremely fast at inference (only table lookups and a dot product). Should beat label-linear (0.8524) on accuracy while keeping Ir/item near 1k. Failure: if dev accuracy ≤ 0.8524 or Ir/item > 2000.
```
```python
"""label-emb-linear: entity embeddings for categorical, z-score for ordinal, then linear.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weight format (little-endian):
  int32 magic=0x454D424C, d, H=0, C, E=8
  per column: int32 type (0=ordinal,1=categorical), then if ordinal: float32 mean, std; if cat: int32 ncat
  float32 T[d][256][C]   (precomputed logit contributions)
  float32 bias[C]
"""
import json, math, struct, sys

import numpy as np

E = 8                     # embedding dimension
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
    d = card["input"]["bytes"]
    types = card["input"]["types"]

    rng = np.random.default_rng(SEED)

    # --- analyze columns ---
    col_info = []          # list of dicts per column
    feat_dim = 0
    for j, t in enumerate(types):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            col_info.append({"type": 0, "mean": mu, "std": sd, "feat_idx": feat_dim})
            feat_dim += 1
        else:
            vals = np.unique(X[:, j])
            ncat = len(vals) + 1            # +1 for unseen (255)
            mapping = {v: i for i, v in enumerate(vals)}
            col_info.append({"type": 1, "ncat": ncat, "mapping": mapping, "feat_idx": feat_dim})
            feat_dim += E

    n = X.shape[0]

    # --- initialize parameters ---
    # embeddings for categorical columns
    emb = []
    for info in col_info:
        if info["type"] == 1:
            # (ncat, E)
            w = (rng.normal(0, 0.01, (info["ncat"], E))).astype(np.float32)
            emb.append(w)
        else:
            emb.append(None)

    # linear weights: (feat_dim, C) and bias (C)
    W = (rng.uniform(-1, 1, (feat_dim, C)) * math.sqrt(6.0 / feat_dim)).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = [W, b] + [e for e in emb if e is not None]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # --- training loop ---
    def build_batch(idx):
        Z = np.zeros((len(idx), feat_dim), np.float32)
        for j, info in enumerate(col_info):
            xj = X[idx, j]
            if info["type"] == 0:
                Z[:, info["feat_idx"]] = (xj - info["mean"]) / info["std"]
            else:
                # map to indices, unseen -> last row
                mapped = np.array([info["mapping"].get(v, info["ncat"] - 1) for v in xj], dtype=np.int32)
                Z[:, info["feat_idx"]:info["feat_idx"] + E] = emb[j][mapped]
        return Z, y[idx]

    def softmax_grad(logits, yy):
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        prob = exp / exp.sum(axis=1, keepdims=True)
        prob[np.arange(len(yy)), yy] -= 1.0
        return (prob / len(yy)).astype(np.float32)

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Z, yy = build_batch(idx)
            logits = Z @ W + b
            g = softmax_grad(logits, yy)
            grad_W = Z.T @ g + L2 * W
            grad_b = g.sum(0)
            grads = [grad_W, grad_b]
            # embedding gradients
            for j, info in enumerate(col_info):
                if info["type"] == 1:
                    mapped = np.array([info["mapping"].get(v, info["ncat"] - 1) for v in X[idx, j]], dtype=np.int32)
                    grad_emb = np.zeros_like(emb[j])
                    np.add.at(grad_emb, mapped, g @ W[info["feat_idx"]:info["feat_idx"] + E].T)
                    grad_emb += L2 * emb[j]
                    grads.append(grad_emb)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # --- precompute T[d][256][C] ---
    T = np.zeros((d, 256, C), np.float32)
    for j, info in enumerate(col_info):
        if info["type"] == 0:
            w_row = W[info["feat_idx"]]          # (C,)
            for v in range(256):
                norm = (v - info["mean"]) / info["std"]
                T[j, v] = norm * w_row
        else:
            w_slice = W[info["feat_idx"]:info["feat_idx"] + E]   # (E, C)
            for v in range(256):
                idx = info["mapping"].get(v, info["ncat"] - 1)
                T[j, v] = emb[j][idx] @ w_slice

    # --- write weights ---
    magic = 0x454D424C
    header = struct.pack("<5i", magic, d, 0, C, E)
    col_bytes = b""
    for info in col_info:
        if info["type"] == 0:
            col_bytes += struct.pack("<i2f", 0, info["mean"], info["std"])
        else:
            col_bytes += struct.pack("<2i", 1, info["ncat"])
    out = header + col_bytes + T.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-emb-linear inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index. */
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
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], E = hd[4];
    if (H != 0 || C < 1 || C > 256 || d != 14) { fprintf(stderr, "bad header\n"); return 3; }

    /* parse column info */
    int col_type[14];
    float col_mean[14], col_std[14];
    int col_ncat[14];
    unsigned char *p = wb + 20;
    for (int j = 0; j < d; j++) {
        int32_t type = *(int32_t *)p; p += 4;
        col_type[j] = type;
        if (type == 0) {
            col_mean[j] = *(float *)p; p += 4;
            col_std[j] = *(float *)p; p += 4;
        } else {
            col_ncat[j] = *(int32_t *)p; p += 4;
        }
    }

    /* T[d][256][C] */
    float *T = (float *)p;
    size_t T_bytes = (size_t)d * 256 * C * sizeof(float);
    p += T_bytes;
    float *bias = (float *)p;

    /* validate total size */
    size_t expected = 20;
    for (int j = 0; j < d; j++)
        expected += (col_type[j] == 0) ? 12 : 8;
    expected += T_bytes + C * sizeof(float);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float scores[2];  // C <= 2 for this task
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        scores[0] = bias[0]; scores[1] = bias[1];
        for (int j = 0; j < d; j++) {
            const float *tj = T + ((size_t)j * 256 + xi[j]) * C;
            scores[0] += tj[0];
            scores[1] += tj[1];
        }
        unsigned char best = (scores[1] > scores[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```