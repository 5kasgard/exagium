```hypothesis
Factorization Machines capture pairwise feature interactions via low-rank embeddings, crucial for tabular data like Adult where combinations (e.g., education×occupation) matter. With embedding size k=4, the model adds only ~14k parameters over linear, but inference requires just O(d·k) operations, keeping cost near linear. If dev accuracy does not exceed label-linear (0.8524) or cost exceeds label-mlp16 (2325 Ir/item), the approach fails.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Learns per-value linear weights and k-dim embeddings.
Weights format: uint32 magic 0x464D4C31, int32 d, k, C; float32 w0; float32 T_lin[d*256]; float32 T_emb[d*256*k]."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 100


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
    d = X.shape[1]
    C = card["classes"]
    assert C == 2
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Parameters
    w0 = np.float32(0.0)
    T_lin = np.zeros((d, 256), dtype=np.float32)
    T_emb = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)

    # Adam state
    m_w0, v_w0 = np.float32(0.0), np.float32(0.0)
    m_lin, v_lin = np.zeros_like(T_lin), np.zeros_like(T_lin)
    m_emb, v_emb = np.zeros_like(T_emb), np.zeros_like(T_emb)

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d) uint8
            yb = y[idx]          # (B,)
            B = len(idx)

            # Forward
            logits = w0 + np.sum(T_lin[np.arange(d), xb.T], axis=0)  # (B,)
            emb = T_emb[np.arange(d), xb.T]                          # (d, B, K)
            emb = np.transpose(emb, (1, 0, 2))                       # (B, d, K)
            sum_v = np.sum(emb, axis=1)                              # (B, K)
            sum_v2 = np.sum(emb * emb, axis=1)                       # (B, K)
            inter = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)     # (B,)
            logits += inter

            # Loss gradient (binary cross-entropy)
            p = 1.0 / (1.0 + np.exp(-logits))
            dlogits = (p - yb) / B                                   # (B,)

            # Gradients
            gw0 = np.sum(dlogits).astype(np.float32)

            g_lin = np.zeros_like(T_lin)
            g_emb = np.zeros_like(T_emb)

            # Linear term gradients
            for j in range(d):
                np.add.at(g_lin[j], xb[:, j], dlogits)

            # Embedding gradients from interaction term
            # d_inter/d_emb[i,j,f] = dlogits[i] * (sum_v[i,f] - emb[i,j,f])
            for f in range(K):
                # (B, d) += dlogits[:,None] * (sum_v[:,f:f+1] - emb[:,:,f])
                diff = sum_v[:, f:f+1] - emb[:, :, f]                # (B, d)
                g_emb[:, :, f] = np.transpose(diff * dlogits[:, None], (1, 0))  # (d, B) -> but we need per category
                # Actually we need to accumulate per category index. Use np.add.at per feature.
                # Let's do it feature by feature for clarity.
            # Recompute per feature to use np.add.at
            g_emb.fill(0.0)
            for j in range(d):
                v = xb[:, j]                                         # (B,)
                for f in range(K):
                    diff = dlogits * (sum_v[:, f] - emb[:, j, f])    # (B,)
                    np.add.at(g_emb[j, :, f], v, diff)

            # L2 regularization
            g_lin += L2 * T_lin
            g_emb += L2 * T_emb

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * gw0 * gw0
            m_hat = m_w0 / (1 - beta1 ** t)
            v_hat = v_w0 / (1 - beta2 ** t)
            w0 -= LR * m_hat / (np.sqrt(v_hat) + eps)

            # T_lin
            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            m_hat = m_lin / (1 - beta1 ** t)
            v_hat = v_lin / (1 - beta2 ** t)
            T_lin -= LR * m_hat / (np.sqrt(v_hat) + eps)

            # T_emb
            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            m_hat = m_emb / (1 - beta1 ** t)
            v_hat = v_emb / (1 - beta2 ** t)
            T_emb -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights
    magic = 0x464D4C31  # 'FM1'
    out = struct.pack("<4i f", magic, d, K, C, w0)
    out += T_lin.astype("<f4").tobytes()
    out += T_emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (uint8 per feature). Output: one byte class (0/1).
 * Weights: uint32 magic 0x464D4C31, int32 d, k, C; float32 w0; float32 T_lin[d*256]; float32 T_emb[d*256*k]. */
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
    if (nw < 4 + 4*3 + 4) { fprintf(stderr, "weights too small\n"); return 3; }

    uint32_t magic = *(uint32_t*)wb;
    if (magic != 0x464D4C31) { fprintf(stderr, "bad magic\n"); return 3; }
    int32_t *hdr = (int32_t*)(wb + 4);
    int d = hdr[0], k = hdr[1], C = hdr[2];
    if (C != 2 || d <= 0 || k <= 0) { fprintf(stderr, "bad header\n"); return 3; }
    float w0 = *(float*)(wb + 4 + 12);
    const float *T_lin = (const float*)(wb + 4 + 16);
    const float *T_emb = T_lin + (long)d * 256;
    long expected = 4 + 16 + 4L * d * 256 * (1 + k);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(k * sizeof(float));
    float *sum_v2 = malloc(k * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        float score = w0;
        for (int f = 0; f < k; f++) { sum_v[f] = 0.0f; sum_v2[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            int v = xi[j];
            long idx = (long)j * 256 + v;
            score += T_lin[idx];
            const float *emb = T_emb + idx * k;
            for (int f = 0; f < k; f++) {
                float e = emb[f];
                sum_v[f] += e;
                sum_v2[f] += e * e;
            }
        }
        float inter = 0.0f;
        for (int f = 0; f < k; f++) inter += sum_v[f] * sum_v[f] - sum_v2[f];
        score += 0.5f * inter;

        unsigned char cls = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```