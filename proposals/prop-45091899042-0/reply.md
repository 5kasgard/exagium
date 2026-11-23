```hypothesis
Family: label-fm (Factorization Machine). Captures pairwise feature interactions via low-rank embeddings, crucial for tabular data like Adult where feature combinations (education×occupation, marital×relationship) drive income. Uses shared embeddings + class-specific linear terms + interaction projection. Expected to beat label-linear (0.8524) by modeling interactions, approaching label-mlp16 (0.8591) at similar inference cost (O(d·K) vs O(d·H)). Failure: dev accuracy ≤ 0.853 or Ir/item ≥ 3000.
```
```python
"""Factorization Machine for tabular label tasks: linear + pairwise interactions via embeddings.
python3 -s -P train.py card.json train.in train.key weights.bin  (deterministic: seed 1, single thread)
Exports: int32 magic 0x464D31, d, K, C; float32 T_lin[d][256][C], b[C], T_emb[d][256][K], W2[K][C]."""
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
    rng = np.random.default_rng(SEED)

    enc = []
    val_counts = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j]
        if t == "ordinal":
            vals = np.arange(256)
            E = np.eye(256, dtype=np.float32)
            enc.append(E)
            val_counts.append(256)
        else:
            vals = np.unique(v)
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
            val_counts.append(len(vals))

    total_vals = sum(val_counts)
    offsets = np.cumsum([0] + val_counts)

    W_lin = rng.normal(0, 0.01, (total_vals, C)).astype(np.float32)
    E_emb = rng.normal(0, 0.01, (total_vals, K)).astype(np.float32)
    W2 = rng.normal(0, 0.01, (K, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    params = [W_lin, E_emb, W2, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            lin_batch = np.zeros((bs, C), dtype=np.float32)
            emb_batch = np.zeros((bs, K), dtype=np.float32)
            emb_sq_batch = np.zeros((bs, K), dtype=np.float32)

            for j in range(X.shape[1]):
                off = offsets[j]
                cnt = val_counts[j]
                vals = xb[:, j]
                lin_batch += W_lin[off + vals]
                e = E_emb[off + vals]
                emb_batch += e
                emb_sq_batch += e * e

            inter = 0.5 * (emb_batch * emb_batch - emb_sq_batch)
            logits = lin_batch + inter @ W2 + b

            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grad_logits = probs / bs

            grad_W_lin = np.zeros_like(W_lin)
            grad_E_emb = np.zeros_like(E_emb)
            grad_W2 = inter.T @ grad_logits
            grad_b = grad_logits.sum(axis=0)

            dL_dinter = grad_logits @ W2.T
            for j in range(X.shape[1]):
                off = offsets[j]
                vals = xb[:, j]
                np.add.at(grad_W_lin, off + vals, grad_logits)
                dinter_dE = emb_batch - E_emb[off + vals]
                grad_E = dL_dinter * dinter_dE
                np.add.at(grad_E_emb, off + vals, grad_E)

            grad_W_lin += L2 * W_lin
            grad_E_emb += L2 * E_emb
            grad_W2 += L2 * W2

            grads = [grad_W_lin, grad_E_emb, grad_W2, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    T_lin = np.zeros((X.shape[1], 256, C), dtype=np.float32)
    T_emb = np.zeros((X.shape[1], 256, K), dtype=np.float32)
    for j in range(X.shape[1]):
        off = offsets[j]
        cnt = val_counts[j]
        if card["input"]["types"][j] == "ordinal":
            T_lin[j] = W_lin[off:off + cnt]
            T_emb[j] = E_emb[off:off + cnt]
        else:
            vals = np.unique(X[:, j])
            T_lin[j, vals] = W_lin[off:off + cnt]
            T_emb[j, vals] = E_emb[off:off + cnt]

    magic = 0x464D31
    out = struct.pack("<4i", magic, X.shape[1], K, C)
    out += T_lin.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    out += T_emb.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: 1 byte class (argmax). Format: see train.py. */
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
    if (nw < 16 || hd[0] != 0x464D31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }

    long expected = 16L + 4L * ((long)d * 256 * C + C + (long)d * 256 * K + (long)K * C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T_lin = (const float *)(hd + 4);
    const float *b = T_lin + (long)d * 256 * C;
    const float *T_emb = b + C;
    const float *W2 = T_emb + (long)d * 256 * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *lin_acc = malloc(sizeof(float) * C);
    float *s = malloc(sizeof(float) * K);
    float *sq = malloc(sizeof(float) * K);
    float *inter = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) lin_acc[c] = b[c];
        for (int j = 0; j < d; j++) {
            const float *t = T_lin + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) lin_acc[c] += t[c];
        }

        for (int f = 0; f < K; f++) { s[f] = 0.0f; sq[f] = 0.0f; }
        for (int j = 0; j < d; j++) {
            const float *e = T_emb + ((long)j * 256 + xi[j]) * K;
            for (int f = 0; f < K; f++) {
                float v = e[f];
                s[f] += v;
                sq[f] += v * v;
            }
        }

        for (int f = 0; f < K; f++) inter[f] = 0.5f * (s[f] * s[f] - sq[f]);

        for (int c = 0; c < C; c++) scores[c] = lin_acc[c];
        for (int f = 0; f < K; f++) {
            float iv = inter[f];
            if (iv != 0.0f) {
                for (int c = 0; c < C; c++) scores[c] += iv * W2[(long)f * C + c];
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```