```hypothesis
We propose "label-fm" (Factorization Machine) family. FM models pairwise feature interactions via low-rank latent vectors, which should capture interactions in Adult data (e.g., education × occupation) better than a linear model, while adding only O(k·d) parameters. With k=4, inference cost is ~14·4=56 multiply-adds per item, comparable to linear. We expect dev accuracy >0.855 (beating label-linear's 0.8524) at similar cost. If accuracy ≤0.8524 or cost >2× linear, the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np

SEED = 1
K = 4
LR = 0.01
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100


def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
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
    assert C == 2
    rng = np.random.default_rng(SEED)

    # Build encoding matrices per column
    enc = []
    d_cols = X.shape[1]
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None]  # (256, 1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float64)
            E[vals, np.arange(len(vals))] = 1.0
        enc.append(E.astype(np.float32))

    # Total dimension
    d = sum(E.shape[1] for E in enc)
    # Build full feature matrix Z (n x d)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n = Z.shape[0]

    # Initialize parameters
    w = (rng.uniform(-1, 1, d) * math.sqrt(6.0 / d)).astype(np.float32)
    V = (rng.uniform(-1, 1, (d, K)) * math.sqrt(6.0 / (d + K))).astype(np.float32)
    bias = np.float32(0.0)

    # Adam state
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    m_b, v_b = np.float32(0.0), np.float32(0.0)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z = Z[idx]
            yy = y[idx]
            batch_sz = len(yy)

            # Forward
            linear = z @ w + bias
            vx = z @ V
            vx_sq = np.sum(vx * vx, axis=1)
            z_sq = z * z
            V_sq = V * V
            z_sq_V_sq = z_sq @ V_sq
            z_sq_V_sq_sum = np.sum(z_sq_V_sq, axis=1)
            interaction = 0.5 * (vx_sq - z_sq_V_sq_sum)
            logit = linear + interaction
            p = 1.0 / (1.0 + np.exp(-logit))
            grad_logit = p - yy

            # Gradients
            grad_bias = np.mean(grad_logit)
            grad_w = (z.T @ grad_logit) / batch_sz + L2 * w
            g_exp = grad_logit[:, None]
            grad_V = (z.T @ (g_exp * vx) - z_sq.T @ (g_exp * V)) / batch_sz + L2 * V

            # Adam updates
            t += 1
            m_w = 0.9 * m_w + 0.1 * grad_w
            v_w = 0.999 * v_w + 0.001 * grad_w * grad_w
            w -= LR * (m_w / (1 - 0.9**t)) / (np.sqrt(v_w / (1 - 0.999**t)) + 1e-8)

            m_V = 0.9 * m_V + 0.1 * grad_V
            v_V = 0.999 * v_V + 0.001 * grad_V * grad_V
            V -= LR * (m_V / (1 - 0.9**t)) / (np.sqrt(v_V / (1 - 0.999**t)) + 1e-8)

            m_b = 0.9 * m_b + 0.1 * grad_bias
            v_b = 0.999 * v_b + 0.001 * grad_bias * grad_bias
            bias -= LR * (m_b / (1 - 0.9**t)) / (np.sqrt(v_b / (1 - 0.999**t)) + 1e-8)

    # Build per-column tables
    T_linear_list = []
    T_latent_list = []
    offset = 0
    for E in enc:
        d_j = E.shape[1]
        w_j = w[offset:offset + d_j]
        V_j = V[offset:offset + d_j]
        T_linear_list.append(E @ w_j)
        T_latent_list.append(E @ V_j)
        offset += d_j

    # Write weights
    magic = 0x464D4C31  # 'FML1'
    d_cols = len(enc)
    K_val = K
    C_val = C
    header = struct.pack("<4i", magic, d_cols, K_val, C_val)
    T_linear_flat = np.concatenate(T_linear_list).astype("<f4")
    T_latent_flat = np.concatenate([arr.ravel() for arr in T_latent_list]).astype("<f4")
    bias_arr = np.array([bias], dtype="<f4")
    out = header + T_linear_flat.tobytes() + T_latent_flat.tobytes() + bias_arr.tobytes()
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d_cols = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d_cols != 14 || K < 1 || K > 16) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected_floats = (long)d_cols * 256 * (1 + K) + 1;
    if (nw != 16 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T_linear = (const float *)(hd + 4);
    const float *T_latent = T_linear + (long)d_cols * 256;
    const float *bias_ptr = T_latent + (long)d_cols * 256 * K;
    float bias = *bias_ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *vx = malloc(sizeof(float) * K);
    unsigned char *rec = malloc(1);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d_cols) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d_cols); return 3; }

        float linear_sum = bias;
        for (int u = 0; u < K; u++) vx[u] = 0.0f;
        float vx_sq_sum = 0.0f;

        for (int j = 0; j < d_cols; j++) {
            unsigned char byte = xi[j];
            linear_sum += T_linear[(long)j * 256 + byte];
            const float *lat = T_latent + ((long)j * 256 + byte) * K;
            for (int u = 0; u < K; u++) {
                float val = lat[u];
                vx[u] += val;
                vx_sq_sum += val * val;
            }
        }
        float vx_norm_sq = 0.0f;
        for (int u = 0; u < K; u++) vx_norm_sq += vx[u] * vx[u];
        float interaction = 0.5f * (vx_norm_sq - vx_sq_sum);
        float score = linear_sum + interaction;
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        rec[0] = pred;
        put_rec(stdout, rec, 1);
    }
    return 0;
}
```