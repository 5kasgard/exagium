```hypothesis
Family: label-fm (factorization machine, degree 2, rank 8).
Why: The adult dataset has strong categorical feature interactions (e.g., education × occupation). A factorization machine captures pairwise interactions with only O(d·r) extra parameters, staying fast and small. It should beat the linear baseline (0.8524) and approach the MLP (0.8591) at lower inference cost.
Failure signal: If dev accuracy ≤ 0.8524 or instructions/item exceed label-linear by >2×, the interaction gain does not justify the cost.
```

```python
"""label-fm: Factorization Machine (degree 2, rank r) for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Exports: magic 0x464D4C31, d, r, C=1; float32 linear_table[d*256], V_table[d*256*r], w0."""
import json, math, struct, sys

import numpy as np

R = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.float32)  # 0 or 1
    d = X.shape[1]
    rng = np.random.default_rng(SEED)

    # Build encoding matrices E_j: 256 x k_j for each column
    enc = []
    k_list = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None]  # 256 x 1
            enc.append(E.astype(np.float32))
            k_list.append(1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
            k_list.append(len(vals))
    k = sum(k_list)

    # Initialize parameters
    w0 = 0.0
    w = (rng.uniform(-1, 1, k) * math.sqrt(6.0 / k)).astype(np.float32)
    V = (rng.uniform(-1, 1, (k, R)) * math.sqrt(6.0 / (k + R))).astype(np.float32)

    # Split w and V by column
    w_split = []
    V_split = []
    offset = 0
    for kj in k_list:
        w_split.append(w[offset:offset + kj])
        V_split.append(V[offset:offset + kj])
        offset += kj

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w = [np.zeros_like(wj) for wj in w_split]
    v_w = [np.zeros_like(wj) for wj in w_split]
    m_V = [np.zeros_like(Vj) for Vj in V_split]
    v_V = [np.zeros_like(Vj) for Vj in V_split]

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            bs = len(yb)

            # Forward
            linear = np.full(bs, w0, dtype=np.float32)
            S = np.zeros((bs, R), dtype=np.float32)
            for j in range(d):
                Ej = enc[j][xb[:, j]]  # bs x kj
                linear += Ej @ w_split[j]
                S += Ej @ V_split[j]

            sumS = S.sum(axis=1)
            sumS2 = (S * S).sum(axis=1)
            interaction = 0.5 * (sumS * sumS - sumS2)
            logit = linear + interaction
            prob = sigmoid(logit)

            # Loss gradient
            dlogit = prob - yb  # bs

            # Backward
            dw0 = dlogit.mean()
            dw_split = [np.zeros_like(wj) for wj in w_split]
            dV_split = [np.zeros_like(Vj) for Vj in V_split]

            for j in range(d):
                Ej = enc[j][xb[:, j]]  # bs x kj
                # Linear part
                dw_split[j] = (Ej.T @ dlogit) / bs + L2 * w_split[j]
                # Interaction part: d(interaction)/dS = sumS - S
                dS = dlogit[:, None] * (sumS[:, None] - S)  # bs x R
                dV_split[j] = (Ej.T @ dS) / bs + L2 * V_split[j]

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

            m_w0 = beta1 * m_w0 + (1 - beta1) * dw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * dw0 * dw0
            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            w0 -= LR * m_w0_hat / (np.sqrt(v_w0_hat) + eps)

            for j in range(d):
                m_w[j] = beta1 * m_w[j] + (1 - beta1) * dw_split[j]
                v_w[j] = beta2 * v_w[j] + (1 - beta2) * dw_split[j] * dw_split[j]
                m_hat = m_w[j] / (1 - beta1 ** t)
                v_hat = v_w[j] / (1 - beta2 ** t)
                w_split[j] -= LR * m_hat / (np.sqrt(v_hat) + eps)

                m_V[j] = beta1 * m_V[j] + (1 - beta1) * dV_split[j]
                v_V[j] = beta2 * v_V[j] + (1 - beta2) * dV_split[j] * dV_split[j]
                m_hat = m_V[j] / (1 - beta1 ** t)
                v_hat = v_V[j] / (1 - beta2 ** t)
                V_split[j] -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Precompute lookup tables for inference
    linear_table = np.zeros((d, 256), dtype=np.float32)
    V_table = np.zeros((d, 256, R), dtype=np.float32)
    for j in range(d):
        Ej = enc[j]  # 256 x kj
        linear_table[j] = Ej @ w_split[j]
        V_table[j] = Ej @ V_split[j]

    # Write weights.bin
    magic = 0x464D4C31  # "FML1"
    out = struct.pack("<4i", magic, d, R, 1)  # C=1 for binary logit
    out += linear_table.astype("<f4").tobytes()
    out += V_table.astype("<f4").tobytes()
    out += struct.pack("<f", w0)
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-fm infer: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class (0/1). */
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
    int d = hd[1], R = hd[2], C = hd[3];
    if (C != 1) { fprintf(stderr, "expected C=1\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 + (long)d * 256 * R + 1);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    const float *linear_table = (const float *)(hd + 4);
    const float *V_table = linear_table + (long)d * 256;
    const float w0 = V_table[(long)d * 256 * R][0];  // last float

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *S = malloc(sizeof(float) * R);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear_sum = w0;
        memset(S, 0, sizeof(float) * R);
        for (int j = 0; j < d; j++) {
            unsigned char b = xi[j];
            linear_sum += linear_table[(long)j * 256 + b];
            const float *Vrow = V_table + ((long)j * 256 + b) * R;
            for (int u = 0; u < R; u++) S[u] += Vrow[u];
        }
        float sumS = 0.0f, sumS2 = 0.0f;
        for (int u = 0; u < R; u++) { sumS += S[u]; sumS2 += S[u] * S[u]; }
        float interaction = 0.5f * (sumS * sumS - sumS2);
        float logit = linear_sum + interaction;
        float prob = 1.0f / (1.0f + expf(-logit));
        unsigned char cls = (prob > 0.5f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```