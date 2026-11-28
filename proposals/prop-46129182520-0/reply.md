```hypothesis
Factorization Machines (FM) capture pairwise feature interactions via factorized embeddings, which are crucial for tabular data with mixed categorical/ordinal features like Adult. The FM's O(d·k) inference (d=14, k=8) is only slightly slower than linear models but should significantly improve accuracy by modeling interactions. If dev accuracy does not exceed label-linear (0.8524), it indicates that either the interaction strength is negligible or the chosen embedding dimension k is too small for this dataset.
```

```python
"""Factorization Machine family for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes ordinal features as normalized scalars, categorical as one-hot over training values.
Learns linear weights w and embedding vectors V (dim k) per encoded dimension.
Exports per-byte lookup tables: int32 magic 0x464D4C31, d, k, C=1; float32 bias, linear[d][256], embed[d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 256, 600000, 100


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
    rng = np.random.default_rng(SEED)

    # Build encoding matrices E_j (256 x d_j)
    enc = []
    d_js = []
    for j, t in enumerate(card["input"]["types"]):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None]  # (256,1)
            enc.append(E)
            d_js.append(1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
            d_js.append(len(vals))
    D = sum(d_js)

    # Concatenated encoding for each example: Z (n x D)
    # We'll compute on the fly per batch to save memory.
    # Precompute row pointers for each feature.
    ptr = [0]
    for dj in d_js:
        ptr.append(ptr[-1] + dj)

    # Parameters: w (D), V (D x K), bias (scalar)
    w = rng.normal(0, 0.01, D).astype(np.float32)
    V = rng.normal(0, 0.01, (D, K)).astype(np.float32)
    bias = np.float32(0.0)

    # Adam state
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    m_b, v_b = np.float32(0.0), np.float32(0.0)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    n = X.shape[0]
    num_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(num_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            batch_size = len(idx)
            # Build Z for this batch
            Z = np.zeros((batch_size, D), dtype=np.float32)
            for j in range(d):
                dj = d_js[j]
                if dj == 1:
                    Z[:, ptr[j]] = enc[j][X[idx, j], 0]
                else:
                    rows = enc[j][X[idx, j]]  # (batch_size, dj)
                    Z[:, ptr[j]:ptr[j+1]] = rows
            z = Z
            zz = z * z  # elementwise square

            # Forward
            s_vec = z @ V                    # (batch_size, K)
            q_vec = zz @ (V * V)             # (batch_size, K)
            interaction = 0.5 * np.sum(s_vec * s_vec - q_vec, axis=1)
            linear = z @ w
            logits = linear + interaction + bias

            # Loss gradient (binary cross-entropy with logits)
            # y is 0/1, logits are for class 1.
            # dL/dlogit = sigmoid(logit) - y
            sig = 1.0 / (1.0 + np.exp(-logits))
            grad_logit = (sig - y[idx]).astype(np.float32)  # (batch_size,)

            # Gradients
            grad_w = z.T @ grad_logit + L2 * w
            # dL/dV = z^T @ (grad_logit * s_vec) - zz^T @ (grad_logit * V)
            # grad_logit * s_vec: (batch_size, K)  (broadcast grad_logit over K)
            grad_V = z.T @ (grad_logit[:, None] * s_vec) - zz.T @ (grad_logit[:, None] * V) + L2 * V
            grad_b = np.float32(grad_logit.sum())

            t += 1
            # Adam updates
            m_w = beta1 * m_w + (1 - beta1) * grad_w
            v_w = beta2 * v_w + (1 - beta2) * grad_w * grad_w
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            w -= LR * m_w_hat / (np.sqrt(v_w_hat) + eps)

            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * grad_V * grad_V
            m_V_hat = m_V / (1 - beta1 ** t)
            v_V_hat = v_V / (1 - beta2 ** t)
            V -= LR * m_V_hat / (np.sqrt(v_V_hat) + eps)

            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * grad_b * grad_b
            m_b_hat = m_b / (1 - beta1 ** t)
            v_b_hat = v_b / (1 - beta2 ** t)
            bias -= LR * m_b_hat / (np.sqrt(v_b_hat) + eps)

    # Bake per-byte lookup tables
    linear_table = np.zeros((d, 256), dtype=np.float32)
    embed_table = np.zeros((d, 256, K), dtype=np.float32)
    for j in range(d):
        E = enc[j]  # (256, dj)
        w_j = w[ptr[j]:ptr[j+1]]
        V_j = V[ptr[j]:ptr[j+1]]
        linear_table[j] = E @ w_j
        embed_table[j] = E @ V_j  # (256, K)

    # Write weights.bin
    # Header: magic, d, K, C=1
    out = struct.pack("<4i", 0x464D4C31, d, K, 1)
    out += struct.pack("<f", bias)
    out += linear_table.astype("<f4").tobytes()
    out += embed_table.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class index (0 or 1).
 * Weights format: int32 magic(0x464D4C31), d, k, C(=1); float32 bias; float32 linear[d][256]; float32 embed[d][256][k]. */
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 1) { fprintf(stderr, "expected C=1\n"); return 3; }
    long expected = 16 + 4L + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + 1;
    const float *embed = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = (float *)malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float sum_w = *bias;
        memset(sum_v, 0, sizeof(float) * K);

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            sum_w += linear[(long)j * 256 + v];
            const float *ev = embed + ((long)j * 256 + v) * K;
            for (int f = 0; f < K; f++) sum_v[f] += ev[f];
        }

        float interaction = 0.0f;
        for (int f = 0; f < K; f++) interaction += sum_v[f] * sum_v[f];
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *ev = embed + ((long)j * 256 + v) * K;
            for (int f = 0; f < K; f++) interaction -= ev[f] * ev[f];
        }
        interaction *= 0.5f;

        float logit = sum_w + interaction;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```