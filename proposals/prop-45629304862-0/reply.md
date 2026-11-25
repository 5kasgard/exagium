```hypothesis
Family: label-emb-mlp. Learns 4-d embeddings for each of 8 categorical features, z-scores 6 ordinal features, then a 16-unit ReLU MLP. Embeddings reduce high-cardinality one-hot dimensionality, capturing feature interactions cheaply. Should beat linear baseline on accuracy with similar inference cost (small dense matmul). Failure: if dev accuracy ≤ 0.8524 or latency > label-linear.
```

```python
import json
import math
import struct
import sys

import numpy as np

SEED = 1
EMBEDDING_DIM = 4
HIDDEN = 16
LR = 0.01
L2 = 1e-4
BATCH = 64
EPOCHS = 50

def read_records(path):
    data = open(path, "rb").read()
    out = []
    i = 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    types = card["input"]["types"]
    d = card["input"]["bytes"]
    C = card["classes"]
    ordinal_idx = [i for i, t in enumerate(types) if t == "ordinal"]
    cat_idx = [i for i, t in enumerate(types) if t == "categorical"]
    num_ordinal = len(ordinal_idx)
    num_categorical = len(cat_idx)
    input_dim = num_ordinal + num_categorical * EMBEDDING_DIM

    X_bytes = np.frombuffer(b"".join(read_records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in read_records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]

    ord_means = np.zeros(num_ordinal, dtype=np.float32)
    ord_stds = np.ones(num_ordinal, dtype=np.float32)
    for i, col in enumerate(ordinal_idx):
        vals = X_bytes[:, col].astype(np.float64)
        mu = vals.mean()
        sd = vals.std()
        if sd == 0:
            sd = 1.0
        ord_means[i] = mu
        ord_stds[i] = sd

    rng = np.random.default_rng(SEED)
    embeddings = rng.normal(0, 0.01, size=(num_categorical, 256, EMBEDDING_DIM)).astype(np.float32)
    W1 = rng.normal(0, math.sqrt(2.0 / input_dim), size=(input_dim, HIDDEN)).astype(np.float32)
    b1 = np.zeros(HIDDEN, dtype=np.float32)
    W2 = rng.normal(0, math.sqrt(2.0 / HIDDEN), size=(HIDDEN, C)).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    params = [embeddings, W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    ord_part = np.zeros((n, num_ordinal), dtype=np.float32)
    for i, col in enumerate(ordinal_idx):
        ord_part[:, i] = (X_bytes[:, col].astype(np.float32) - ord_means[i]) / ord_stds[i]
    cat_indices = X_bytes[:, cat_idx]

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            batch_size = len(idx)
            x_batch = np.zeros((batch_size, input_dim), dtype=np.float32)
            x_batch[:, :num_ordinal] = ord_part[idx]
            for c in range(num_categorical):
                emb = embeddings[c]
                cat_vals = cat_indices[idx, c]
                x_batch[:, num_ordinal + c*EMBEDDING_DIM : num_ordinal + (c+1)*EMBEDDING_DIM] = emb[cat_vals]
            yy = y[idx]

            hidden_pre = x_batch @ W1 + b1
            hidden = np.maximum(hidden_pre, 0)
            logits = hidden @ W2 + b2

            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(batch_size), yy] -= 1.0
            grad_logits = probs / batch_size

            grad_W2 = hidden.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_hidden = grad_logits @ W2.T
            grad_hidden[hidden_pre <= 0] = 0.0
            grad_W1 = x_batch.T @ grad_hidden + L2 * W1
            grad_b1 = grad_hidden.sum(axis=0)

            grad_embeddings = np.zeros_like(embeddings)
            for c in range(num_categorical):
                start = num_ordinal + c*EMBEDDING_DIM
                end = start + EMBEDDING_DIM
                grad_x_emb = grad_hidden[:, start:end]
                cat_vals = cat_indices[idx, c]
                np.add.at(grad_embeddings[c], cat_vals, grad_x_emb)
            grad_embeddings += L2 * embeddings

            t += 1
            grads = [grad_embeddings, grad_W1, grad_b1, grad_W2, grad_b2]
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    magic = 0x454D424D
    type_map = np.zeros(d, dtype=np.uint8)
    type_map[ordinal_idx] = 1
    out = struct.pack("<7i", magic, d, num_categorical, num_ordinal, EMBEDDING_DIM, HIDDEN, C)
    out += type_map.tobytes()
    out += ord_means.astype("<f4").tobytes()
    out += ord_stds.astype("<f4").tobytes()
    out += embeddings.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
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
    unsigned char *wbuf = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wbuf;
    if (nw < 32 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1];
    int num_categorical = hd[2];
    int num_ordinal = hd[3];
    int embedding_dim = hd[4];
    int hidden = hd[5];
    int C = hd[6];
    if (C != 2 || num_ordinal + num_categorical != d) {
        fprintf(stderr, "bad weights config\n"); return 3;
    }
    int input_dim = num_ordinal + num_categorical * embedding_dim;
    unsigned char *type_map = wbuf + 28;
    float *ptr = (float *)(wbuf + 28 + d);
    float *ord_means = ptr; ptr += num_ordinal;
    float *ord_stds = ptr; ptr += num_ordinal;
    float *embeddings = ptr; ptr += (long)num_categorical * 256 * embedding_dim;
    float *W1 = ptr; ptr += (long)input_dim * hidden;
    float *b1 = ptr; ptr += hidden;
    float *W2 = ptr; ptr += (long)hidden * C;
    float *b2 = ptr;

    long expected = 28 + d
        + (long)num_ordinal * 2 * 4
        + (long)num_categorical * 256 * embedding_dim * 4
        + (long)input_dim * hidden * 4
        + (long)hidden * 4
        + (long)hidden * C * 4
        + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *input_vec = malloc(sizeof(float) * input_dim);
    float *hidden_vec = malloc(sizeof(float) * hidden);
    float *output_vec = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        int ord_idx = 0;
        int cat_idx = 0;
        for (int j = 0; j < d; j++) {
            if (type_map[j]) {
                float val = (float)xi[j];
                input_vec[ord_idx] = (val - ord_means[ord_idx]) / ord_stds[ord_idx];
                ord_idx++;
            } else {
                const float *emb = embeddings + (long)cat_idx * 256 * embedding_dim + (long)xi[j] * embedding_dim;
                memcpy(input_vec + num_ordinal + (long)cat_idx * embedding_dim, emb, embedding_dim * sizeof(float));
                cat_idx++;
            }
        }

        for (int h = 0; h < hidden; h++) {
            float sum = b1[h];
            for (int id = 0; id < input_dim; id++) {
                sum += input_vec[id] * W1[(long)id * hidden + h];
            }
            hidden_vec[h] = sum > 0.0f ? sum : 0.0f;
        }

        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < hidden; h++) {
                sum += hidden_vec[h] * W2[(long)h * C + c];
            }
            output_vec[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (output_vec[c] > output_vec[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(input_vec);
    free(hidden_vec);
    free(output_vec);
    if (x) free(x);
    free(wbuf);
    return 0;
}
```