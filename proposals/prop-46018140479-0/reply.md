```hypothesis
Family: label-fm (Factorization Machine). Adds low-rank pairwise feature interactions (k=4) to the linear per-byte
lookup tables, capturing combinatorial effects (e.g., education×occupation) crucial for Adult. The FM term costs only
O(d·k) extra FLOPs/item (d=14, k=4) — far cheaper than an MLP layer — while FMs are known to excel on sparse
tabular data. If dev accuracy ≤ label-linear (0.8524) or Ir/item ≥ 2000, the hypothesis fails.
```
```python
"""label-fm: Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Adds rank-k=4 FM interactions to the linear per-byte tables. Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Weights: magic 0x464D4C31, d, k, C; float32 T_lin[d][256][C], b_lin[C], T_fm[d][256][k], W_fm[k][C]."""
import json, math, struct, sys

import numpy as np

K = 4
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
    d = card["input"]["bytes"]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # Build per-column encoders (same as label-linear)
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals))); E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)

    # Linear part: each column -> C logits
    # FM part: each column -> K latent factors
    W_lin = []
    W_fm = []
    for j, E in enumerate(enc):
        in_dim = E.shape[1]
        # Linear weights
        W_lin.append((rng.uniform(-1, 1, (in_dim, C)) * math.sqrt(6.0 / in_dim)).astype(np.float32))
        # FM weights
        W_fm.append((rng.uniform(-1, 1, (in_dim, K)) * math.sqrt(6.0 / in_dim)).astype(np.float32))
    b_lin = np.zeros(C, np.float32)
    W_fm_out = (rng.uniform(-1, 1, (K, C)) * math.sqrt(6.0 / K)).astype(np.float32)

    params = [W_lin, [b_lin], W_fm, [W_fm_out]]
    state = []
    for group in params:
        gstate = []
        for p in group:
            gstate.append((np.zeros_like(p), np.zeros_like(p)))
        state.append(gstate)

    # Precompute encoded features
    Z_list = [E[X[:, j]].astype(np.float32) for j, E in enumerate(enc)]
    n = X.shape[0]

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits); P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            yy = y[idx]
            # Gather batch
            batch_Z = [Z[idx] for Z in Z_list]
            # Forward
            logits_lin = b_lin.copy()
            sum_fm = np.zeros((len(idx), K), np.float32)
            sum_fm_sq = np.zeros((len(idx), K), np.float32)
            for j, Zj in enumerate(batch_Z):
                logits_lin += Zj @ W_lin[j]
                v = Zj @ W_fm[j]          # (B, K)
                sum_fm += v
                sum_fm_sq += v * v
            fm_inter = 0.5 * (sum_fm * sum_fm - sum_fm_sq)   # (B, K)
            logits_fm = fm_inter @ W_fm_out                    # (B, C)
            logits = logits_lin + logits_fm

            # Backward
            g = softmax_grad(logits, yy)   # (B, C)

            # Gradients for linear part
            grad_W_lin = []
            for j, Zj in enumerate(batch_Z):
                grad_W_lin.append(Zj.T @ g + L2 * W_lin[j])
            grad_b_lin = g.sum(0) + L2 * b_lin

            # Gradients for FM part
            # dL/dW_fm_out = fm_inter.T @ g
            grad_W_fm_out = fm_inter.T @ g + L2 * W_fm_out
            # dL/d fm_inter = g @ W_fm_out.T
            g_fm_inter = g @ W_fm_out.T   # (B, K)
            # dL/d sum_fm = g_fm_inter * sum_fm
            g_sum_fm = g_fm_inter * sum_fm
            # dL/d sum_fm_sq = -0.5 * g_fm_inter
            g_sum_fm_sq = -0.5 * g_fm_inter
            # dL/d v = g_sum_fm + 2 * v * g_sum_fm_sq
            grad_W_fm = []
            for j, Zj in enumerate(batch_Z):
                v = Zj @ W_fm[j]
                g_v = g_sum_fm + 2.0 * v * g_sum_fm_sq
                grad_W_fm.append(Zj.T @ g_v + L2 * W_fm[j])

            grads = [grad_W_lin, [grad_b_lin], grad_W_fm, [grad_W_fm_out]]

            # Adam update
            t += 1
            for gi, (pgroup, ggroup, sgroup) in enumerate(zip(params, grads, state)):
                for pi, (p, gr, (m, v)) in enumerate(zip(pgroup, ggroup, sgroup)):
                    m *= 0.9; m += 0.1 * gr
                    v *= 0.999; v += 0.001 * gr * gr
                    m_hat = m / (1 - 0.9 ** t)
                    v_hat = v / (1 - 0.999 ** t)
                    p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export per-byte lookup tables
    # T_lin[j][val] = E[val] @ W_lin[j]  -> shape (d, 256, C)
    T_lin = np.zeros((d, 256, C), np.float32)
    T_fm = np.zeros((d, 256, K), np.float32)
    for j, E in enumerate(enc):
        T_lin[j] = (E.astype(np.float32) @ W_lin[j]).astype(np.float32)
        T_fm[j] = (E.astype(np.float32) @ W_fm[j]).astype(np.float32)

    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    out += T_lin.astype("<f4").tobytes()
    out += b_lin.astype("<f4").tobytes()
    out += T_fm.astype("<f4").tobytes()
    out += W_fm_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-fm: Factorization Machine for label tasks. ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: 1 byte class (argmax).
 * Weights format: magic 0x464D4C31, int32 d, K, C; float32 T_lin[d][256][C], b_lin[C], T_fm[d][256][K], W_fm[K][C]. */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 64) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * C + C + (long)d * 256 * K + (long)K * C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *T_lin = p; p += (long)d * 256 * C;
    const float *b_lin = p; p += C;
    const float *T_fm = p; p += (long)d * 256 * K;
    const float *W_fm = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc_lin = malloc(sizeof(float) * C);
    float *sum_fm = malloc(sizeof(float) * K);
    float *sum_fm_sq = malloc(sizeof(float) * K);
    float *fm_inter = malloc(sizeof(float) * K);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Linear part
        for (int c = 0; c < C; c++) acc_lin[c] = b_lin[c];
        for (int j = 0; j < d; j++) {
            const float *t = T_lin + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) acc_lin[c] += t[c];
        }

        // FM part
        for (int f = 0; f < K; f++) { sum_fm[f] = 0.0f; sum_fm_sq[f] = 0.0f; }
        for (int j = 0; j < d; j++) {
            const float *t = T_fm + ((long)j * 256 + xi[j]) * K;
            for (int f = 0; f < K; f++) {
                float v = t[f];
                sum_fm[f] += v;
                sum_fm_sq[f] += v * v;
            }
        }
        for (int f = 0; f < K; f++) {
            fm_inter[f] = 0.5f * (sum_fm[f] * sum_fm[f] - sum_fm_sq[f]);
        }
        for (int c = 0; c < C; c++) {
            logits[c] = acc_lin[c];
            for (int f = 0; f < K; f++) logits[c] += fm_inter[f] * W_fm[(long)f * C + c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```