```hypothesis
Family: label-fm (factorization machine with linear term).
Why: Character prediction benefits from pairwise byte interactions (e.g., digrams/trigrams). Linear model misses these; GBRT captures them but is 18× slower. FM adds a low-rank interaction term (k=16) costing only ~2× linear inference, staying well under 20k Ir/item.
Failure: If dev accuracy ≤ 0.3375 (linear) or Ir/item > 20000, the model doesn't justify its complexity.
```

```python
"""Factorization Machine with linear term for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: int32 magic 0x464D4C31, d, k, C; float32 T_lin[d][256][C], b1[C], V[d][256][k], W2[k][C], b2[C]."""
import json, math, struct, sys

import numpy as np

K = 16
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
    d = X.shape[1]
    rng = np.random.default_rng(SEED)

    # Build per-feature one-hot encoders
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        if t == "ordinal":
            vals = np.arange(256, dtype=np.uint8)
        else:
            vals = np.unique(X[:, j])
        E = np.zeros((256, len(vals)), dtype=np.float32)
        E[vals, np.arange(len(vals))] = 1.0
        enc.append(E)

    # Concatenated one-hot matrix Z (n x D)
    Z_list = [E[X[:, j]] for j, E in enumerate(enc)]
    Z = np.concatenate(Z_list, axis=1).astype(np.float32)
    n, D = Z.shape

    # Initialize parameters
    W1 = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b1 = np.zeros(C, np.float32)
    V = (rng.uniform(-1, 1, (D, K)) * math.sqrt(6.0 / D)).astype(np.float32)
    W2 = (rng.uniform(-1, 1, (K, C)) * math.sqrt(6.0 / K)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [W1, b1, V, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            B = len(yy)

            # Forward
            linear_logits = z @ W1 + b1
            sum_v = z @ V
            sum_v_sq = z @ (V * V)
            interaction = 0.5 * (sum_v * sum_v - sum_v_sq)
            interaction_logits = interaction @ W2 + b2
            logits = linear_logits + interaction_logits

            # Softmax gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yy] -= 1.0
            dlogits = (probs / B).astype(np.float32)

            # Backward
            dW1 = z.T @ dlogits + L2 * W1
            db1 = dlogits.sum(axis=0)

            dinteraction_logits = dlogits
            dW2 = interaction.T @ dinteraction_logits + L2 * W2
            db2 = dinteraction_logits.sum(axis=0)

            dinteraction = dinteraction_logits @ W2.T
            dsum_v = dinteraction * sum_v
            dsum_v_sq = dinteraction * (-0.5)
            dV = z.T @ (dsum_v + 2.0 * V * dsum_v_sq) + L2 * V

            grads = [dW1, db1, dV, dW2, db2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Compress to per-feature lookup tables
    T_lin_list, V_list = [], []
    off = 0
    for E in enc:
        T_lin_list.append((E @ W1[off:off + E.shape[1]]).astype(np.float32))
        V_list.append((E @ V[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]

    T_lin = np.stack(T_lin_list)          # (d, 256, C)
    V_table = np.stack(V_list)            # (d, 256, K)

    # Write weights.bin
    magic = 0x464D4C31  # 'FML1'
    header = struct.pack("<4i", magic, d, K, C)
    out = header
    out += T_lin.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += V_table.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine with linear term: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index (argmax). */
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 256) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * C + C + (long)d * 256 * K + (long)K * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * C;
    const float *V = b1 + C;
    const float *W2 = V + (long)d * 256 * K;
    const float *b2 = W2 + (long)K * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *sum_v = malloc(sizeof(float) * K);
    float *sum_v_sq = malloc(sizeof(float) * K);
    float *interaction = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(scores, b1, C * sizeof(float));
        memset(sum_v, 0, K * sizeof(float));
        memset(sum_v_sq, 0, K * sizeof(float));

        for (int j = 0; j < d; j++) {
            unsigned char byte = xi[j];
            const float *t = T + ((long)j * 256 + byte) * C;
            for (int c = 0; c < C; c++) scores[c] += t[c];
            const float *v = V + ((long)j * 256 + byte) * K;
            for (int u = 0; u < K; u++) {
                sum_v[u] += v[u];
                sum_v_sq[u] += v[u] * v[u];
            }
        }

        for (int u = 0; u < K; u++)
            interaction[u] = 0.5f * (sum_v[u] * sum_v[u] - sum_v_sq[u]);

        for (int c = 0; c < C; c++) {
            float acc = b2[c];
            for (int u = 0; u < K; u++) acc += interaction[u] * W2[(long)u * C + c];
            scores[c] += acc;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```