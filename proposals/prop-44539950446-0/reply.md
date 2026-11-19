```hypothesis
Family: label-fm (degree-2 Factorization Machine). Adult income has strong pairwise interactions (e.g., education×occupation, capital-gain×hours-per-week) that linear models miss. FM captures all pairwise interactions with O(Kd) parameters (K=8 latent dim, d≈105 expanded features). With only 14 active features per sample (one per original column), inference computes the interaction term via the efficient 0.5*sum_f((sum V_i x_i)^2 - sum (V_i x_i)^2) formula in ~91*K multiply-adds. This should exceed label-linear accuracy (0.8524) while staying well below label-mlp16 cost (2325 Ir/item). Failure if dev accuracy ≤ 0.8524 or Ir/item ≥ 2325.
```
```python
import json, math, struct, sys
import numpy as np

K = 8
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

    d_orig = card["input"]["bytes"]
    types = card["input"]["types"]
    C = card["classes"]

    rng = np.random.default_rng(SEED)

    col_type = np.zeros(d_orig, dtype=np.uint8)
    col_card = np.zeros(d_orig, dtype=np.uint16)
    col_offset = np.zeros(d_orig, dtype=np.uint32)
    z_tables = []

    expanded_dim = 0
    for j, t in enumerate(types):
        if t == "ordinal":
            col_type[j] = 0
            col_card[j] = 1
            col_offset[j] = expanded_dim
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            z_table = ((np.arange(256) - mu) / sd).astype(np.float32)
            z_tables.append(z_table)
            expanded_dim += 1
        else:
            col_type[j] = 1
            vals = np.unique(X[:, j])
            card = len(vals)
            col_card[j] = card
            col_offset[j] = expanded_dim
            z_tables.append(None)
            expanded_dim += card

    d = expanded_dim
    print(f"Expanded dim: {d}, latent: {K}", file=sys.stderr)

    w0 = np.float32(0.0)
    w = np.zeros(d, dtype=np.float32)
    V = (rng.normal(0, 0.01, (d, K))).astype(np.float32)

    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    sample_features = []
    for i in range(n):
        feats = []
        for j in range(d_orig):
            v = X[i, j]
            if col_type[j] == 0:
                idx = col_offset[j]
                val = z_tables[j][v]
                feats.append((idx, val))
            else:
                if v < col_card[j]:
                    idx = col_offset[j] + v
                    feats.append((idx, 1.0))
        sample_features.append(feats)

    t = 0
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            batch_size = len(idx)

            grad_w0 = 0.0
            grad_w = np.zeros_like(w)
            grad_V = np.zeros_like(V)

            for i in idx:
                feats = sample_features[i]
                yi = y[i]

                linear = w0
                sum_f = np.zeros(K, dtype=np.float32)
                sum_sq_f = np.zeros(K, dtype=np.float32)

                for exp_idx, val in feats:
                    linear += w[exp_idx] * val
                    Vf = V[exp_idx]
                    t_val = Vf * val
                    sum_f += t_val
                    sum_sq_f += t_val * t_val

                interaction = 0.5 * np.sum(sum_f * sum_f - sum_sq_f)
                score = linear + interaction

                if score >= 0:
                    p = 1.0 / (1.0 + math.exp(-score))
                else:
                    ez = math.exp(score)
                    p = ez / (1.0 + ez)

                g = np.float32(p - yi)

                grad_w0 += g
                for exp_idx, val in feats:
                    grad_w[exp_idx] += g * val
                    Vf = V[exp_idx]
                    grad_V[exp_idx] += g * val * (sum_f - Vf * val)

            grad_w0 /= batch_size
            grad_w /= batch_size
            grad_V /= batch_size

            grad_w += L2 * w
            grad_V += L2 * V

            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

            m_w0 = beta1 * m_w0 + (1 - beta1) * grad_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * grad_w0 * grad_w0
            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            w0 -= LR * m_w0_hat / (np.sqrt(v_w0_hat) + eps)

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

    magic = 0x464D3231
    num_ordinal = int(np.sum(col_type == 0))

    out = struct.pack("<5I", magic, d_orig, K, C, num_ordinal)
    out += col_type.tobytes()
    out += col_offset.astype("<I").tobytes()

    for j in range(d_orig):
        if col_type[j] == 0:
            out += z_tables[j].tobytes()

    for j in range(d_orig):
        if col_type[j] == 1:
            out += struct.pack("<H", col_card[j])

    out += struct.pack("<f", w0)
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()

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
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    const uint32_t *hdr = (const uint32_t *)wb;
    if (nw < 20 || hdr[0] != 0x464D3231) { fprintf(stderr, "bad weights magic\n