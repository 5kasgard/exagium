```hypothesis
Family: label-fm (Field-aware Factorization Machine with rank k=8).
Why: Adult income has strong pairwise interactions (education×occupation, marital-status×relationship, capital-gain×hours-per-week) that linear models miss. FM captures all O(d²) interactions via low-rank embeddings (14 columns × 256 values × 8 dims × 2 classes) at O(C·d·k) ≈ 224 FLOPs/item using the identity Σ_{i<j}⟨v_i,v_j⟩ = ½(‖Σv_i‖² - Σ‖v_i‖²). This adds minimal cost over label-linear (719 Ir/item) while modeling interactions GBDT captures at 7570 Ir/item.
Failure: dev accuracy ≤ 0.8524 (label-linear) or Ir/item > 1500.
```
```python
"""label-fm: Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Exports: magic 0x464D4C31, d, k, C; float32 T[d][256][C], b[C], E[d][256][k][C]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-3, 1e-4, 64, 600000, 100


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
    d, C = X.shape[1], card["classes"]
    rng = np.random.default_rng(SEED)

    # params: T[d,256,C], b[C], E[d,256,K,C]
    T = (rng.normal(0, 0.01, (d, 256, C))).astype(np.float32)
    b = np.zeros(C, np.float32)
    E = (rng.normal(0, 0.01, (d, 256, K, C))).astype(np.float32)

    params = [T, b, E]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # linear logits [bs, C]
            lin = b + np.sum(T[np.arange(d)[:, None, None], xb.T[:, :, None], :], axis=0)

            # embeddings [bs, d, K, C]
            emb = E[np.arange(d)[:, None, None, None], xb.T[:, :, None, None], :, :]
            # sum over d -> [bs, K, C]
            emb_sum = np.sum(emb, axis=1)
            # sum of squares over d and K -> [bs, C]
            emb_sq_sum = np.sum(emb * emb, axis=(1, 2))

            # interaction = 0.5 * (||sum||^2 - sum|| ||^2)  [bs, C]
            inter = 0.5 * (np.sum(emb_sum * emb_sum, axis=1) - emb_sq_sum)

            logits = lin + inter
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grad_logits = (probs / bs).astype(np.float32)  # [bs, C]

            # grads
            g_T = np.zeros_like(T)
            g_b = np.zeros_like(b)
            g_E = np.zeros_like(E)

            # linear grads
            for j in range(d):
                vals = xb[:, j]
                np.add.at(g_T, (j, vals, slice(None)), grad_logits)
            g_b[:] = grad_logits.sum(axis=0)

            # embedding grads
            # d(inter)/d(emb_sum) = emb_sum  [bs, K, C]
            # d(inter)/d(emb) for each j = emb_sum - emb_j
            for j in range(d):
                vals = xb[:, j]
                emb_j = emb[:, j]  # [bs, K, C]
                g_emb_j = emb_sum - emb_j  # [bs, K, C]
                # multiply by grad_logits [bs, C] -> [bs, K, C]
                g_emb_j *= grad_logits[:, None, :]
                np.add.at(g_E, (j, vals, slice(None), slice(None)), g_emb_j)

            # L2
            g_T += L2 * T
            g_E += L2 * E

            grads = [g_T, g_b, g_E]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # export
    magic = 0x464D4C31  # "FML1"
    out = struct.pack("<4i", magic, d, K, C)
    out += T.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    out += E.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-fm: Factorization Machine inference. ./infer weights.bin in.rec N */
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
    if (C < 1 || C > 256 || K < 1 || K > 64 || d < 1 || d > 256) {
        fprintf(stderr, "bad dimensions\n"); return 3;
    }
    long expected = 16L + 4L * ((long)d * 256 * C + C + (long)d * 256 * K * C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *b = T + (long)d * 256 * C;
    const float *E = b + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *lin = malloc(sizeof(float) * C);
    float *emb_sum = malloc(sizeof(float) * K * C);
    float *emb_sq_sum = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            lin[c] = b[c];
            emb_sq_sum[c] = 0.0f;
            for (int u = 0; u < K; u++) emb_sum[u * C + c] = 0.0f;
        }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *Tj = T + ((long)j * 256 + v) * C;
            const float *Ej = E + ((long)j * 256 + v) * K * C;
            for (int c = 0; c < C; c++) {
                lin[c] += Tj[c];
                for (int u = 0; u < K; u++) {
                    float e = Ej[u * C + c];
                    emb_sum[u * C + c] += e;
                    emb_sq_sum[c] += e * e;
                }
            }
        }

        float best_score = -1e30f;
        unsigned char best_cls = 0;
        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int u = 0; u < K; u++) {
                float s = emb_sum[u * C + c];
                inter += s * s;
            }
            inter = 0.5f * (inter - emb_sq_sum[c]);
            float score = lin[c] + inter;
            if (score > best_score) { best_score = score; best_cls = (unsigned char)c; }
        }
        put_rec(stdout, &best_cls, 1);
    }
    return 0;
}
```