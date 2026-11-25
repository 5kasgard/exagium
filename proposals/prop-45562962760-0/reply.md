```hypothesis
Family: label-emb (embedding-based linear model). Ordinal features get a scalar weight per bin; categorical features get a 4-d embedding per category. A single linear layer on the concatenated embeddings predicts the class. This should beat label-linear because embeddings capture similarity among categorical values (e.g., workclass, occupation) while adding only ~10k parameters. Inference is ~14*4=56 multiply-adds per item, far cheaper than label-mlp16. If it fails to beat 0.8524, categorical similarities are not useful here or ordinal bins need richer representation.
```

```python
"""label-emb: embedding + linear for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 1e-2
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
CAT_EMB_DIM = 4  # embedding dimension for categorical features


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

    # embedding dim per feature: 1 for ordinal, CAT_EMB_DIM for categorical
    dims = np.array([1 if t == "ordinal" else CAT_EMB_DIM for t in types], dtype=np.int32)
    total_dim = int(dims.sum())
    offsets = np.zeros(d + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(dims * 256)

    rng = np.random.default_rng(SEED)
    # embeddings: flat array of size sum(dims*256)
    emb_size = int((dims * 256).sum())
    emb = (rng.normal(0, 0.01, emb_size)).astype(np.float32)
    # linear weights: (total_dim, C)
    W = (rng.normal(0, math.sqrt(6.0 / total_dim), (total_dim, C))).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    # Adam state
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    t = 0

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # forward: gather embeddings
            # shape (B, total_dim)
            z = np.zeros((B, total_dim), dtype=np.float32)
            for j in range(d):
                dim_j = dims[j]
                off = offsets[j]
                # emb[off + xb[:,j]*dim_j : off + (xb[:,j]+1)*dim_j] for each sample
                # vectorized via advanced indexing
                idx_emb = off + xb[:, j].astype(np.int32) * dim_j
                for k in range(dim_j):
                    z[:, offsets[j] // 256 + k] = emb[idx_emb + k]  # offsets[j]//256 is feature start in z
            # linear
            logits = z @ W + b  # (B, C)
            # softmax + cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = (probs / B).astype(np.float32)  # (B, C)

            # gradient wrt W, b
            grad_W = z.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
            # gradient wrt z
            grad_z = grad_logits @ W.T  # (B, total_dim)
            # gradient wrt embeddings (scatter add)
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                dim_j = dims[j]
                off = offsets[j]
                feat_start = off // 256
                for k in range(dim_j):
                    np.add.at(grad_emb, off + xb[:, j].astype(np.int32) * dim_j + k, grad_z[:, feat_start + k])

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            # embeddings
            m_emb = beta1 * m_emb + (1 - beta1) * grad_emb
            v_emb = beta2 * v_emb + (1 - beta2) * grad_emb * grad_emb
            m_hat = m_emb / (1 - beta1 ** t)
            v_hat = v_emb / (1 - beta2 ** t)
            emb -= LR * m_hat / (np.sqrt(v_hat) + eps)
            # W
            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * grad_W * grad_W
            m_hat = m_W / (1 - beta1 ** t)
            v_hat = v_W / (1 - beta2 ** t)
            W -= LR * m_hat / (np.sqrt(v_hat) + eps)
            # b
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * grad_b * grad_b
            m_hat = m_b / (1 - beta1 ** t)
            v_hat = v_b / (1 - beta2 ** t)
            b -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # pack weights
    magic = 0x454D424C  # 'EMBL' little-endian
    header = struct.pack("<4I", magic, d, CAT_EMB_DIM, C)
    dims_bytes = dims.astype(np.uint8).tobytes()
    emb_bytes = emb.astype("<f4").tobytes()
    W_bytes = W.astype("<f4").tobytes()
    b_bytes = b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + dims_bytes + emb_bytes + W_bytes + b_bytes)


if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x454D424C) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], cat_emb_dim = hd[2], C = hd[3];
    if (d != 14 || C != 2 || cat_emb_dim != 4) { fprintf(stderr, "unexpected header values\n"); return 3; }
    const uint8_t *dims = (const uint8_t *)(hd + 4);
    int total_dim = 0;
    for (int i = 0; i < d; i++) total_dim += dims[i];
    /* pointers into weight blob */
    const float *emb = (const float *)(dims + d);
    const float *W = emb;
    for (int i = 0; i < d; i++) W += 256 * dims[i];
    const float *b = W + (long)total_dim * C;
    /* validate total size */
    long expected = 16 + d + 4L * (256L * total_dim + (long)total_dim * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * total_dim);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        /* gather embeddings into z */
        int z_idx = 0;
        for (int j = 0; j < d; j++) {
            int dim_j = dims[j];
            const float *e = emb + (long)xi[j] * dim_j;
            for (int k = 0; k < dim_j; k++) z[z_idx++] = e[k];
        }
        /* linear layer */
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int idx = 0; idx < total_dim; idx++) {
            float zv = z[idx];
            const float *wrow = W + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += zv * wrow[c];
        }
        /* argmax */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```