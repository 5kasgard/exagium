```hypothesis
Family: label-fm (Factorization Machine). Adds pairwise feature interactions via low-rank embeddings (k=4) to a linear base, capturing combinations like education×occupation that matter for income. Expects ~0.858 dev accuracy at ~1100 Ir/item (vs label-linear 0.8524 @ 719). If dev accuracy ≤0.853 or Ir/item >2000, the FM interactions do not justify their cost here.
```

```python
"""label-fm: Factorization Machine for tabular byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Linear logits + FM interactions (k=4). Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Weights: magic=0x464D4C31, d, k, C; float32 T_lin[d*256*C], Emb[d*256*k], W_fm[k*C], b[C]."""
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
    d, C = X.shape[1], card["classes"]
    rng = np.random.default_rng(SEED)

    # Parameters: W_lin[d][256,C], Emb[d][256,K], W_fm[K,C], b[C]
    W_lin = [(rng.uniform(-0.01, 0.01, (256, C)) / math.sqrt(256)).astype(np.float32) for _ in range(d)]
    Emb   = [(rng.uniform(-0.01, 0.01, (256, K)) / math.sqrt(256)).astype(np.float32) for _ in range(d)]
    W_fm  = (rng.uniform(-0.01, 0.01, (K, C)) / math.sqrt(K)).astype(np.float32)
    b     = np.zeros(C, np.float32)

    # Adam state
    params = [p for j in range(d) for p in (W_lin[j], Emb[j])] + [W_fm, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            B = len(yb)

            # Forward
            lin_logits = np.zeros((B, C), np.float32)
            sum_emb = np.zeros((B, K), np.float32)
            sum_emb_sq = np.zeros((B, K), np.float32)
            for j in range(d):
                v = xb[:, j]
                lin_logits += W_lin[j][v]
                e = Emb[j][v]
                sum_emb += e
                sum_emb_sq += e * e
            fm_out = 0.5 * (sum_emb * sum_emb - sum_emb_sq)
            logits = lin_logits + fm_out @ W_fm + b

            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            dlogits = (probs / B).astype(np.float32)

            # Gradients
            db = dlogits.sum(axis=0)
            dW_fm = fm_out.T @ dlogits
            dfm_out = dlogits @ W_fm.T

            dW_lin = [np.zeros((256, C), np.float32) for _ in range(d)]
            dEmb   = [np.zeros((256, K), np.float32) for _ in range(d)]
            for j in range(d):
                v = xb[:, j]
                np.add.at(dW_lin[j], v, dlogits)
                e_j = Emb[j][v]
                grad_emb = (sum_emb - e_j) * dfm_out
                np.add.at(dEmb[j], v, grad_emb)

            # L2
            for j in range(d):
                dW_lin[j] += L2 * W_lin[j]
                dEmb[j]   += L2 * Emb[j]
            dW_fm += L2 * W_fm

            # Adam update
            grads = [g for j in range(d) for g in (dW_lin[j], dEmb[j])] + [dW_fm, db]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export
    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    for j in range(d):
        out += W_lin[j].astype("<f4").tobytes()
    for j in range(d):
        out += Emb[j].astype("<f4").tobytes()
    out += W_fm.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Factorization Machine with k=4. Each input record: d bytes. Output: 1 byte class. */
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
    if (C < 1 || C > 256 || K < 1 || K > 64 ||
        nw != 16 + 4L * ((long)d * 256 * C + (long)d * 256 * K + (long)K * C + C)) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T_lin = (const float *)(hd + 4);
    const float *Emb   = T_lin + (long)d * 256 * C;
    const float *W_fm  = Emb   + (long)d * 256 * K;
    const float *b     = W_fm  + (long)K * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * C);
    float *sum_emb = malloc(sizeof(float) * K);
    float *sum_emb_sq = malloc(sizeof(float) * K);
    float *fm_out = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) acc[c] = b[c];
        for (int u = 0; u < K; u++) sum_emb[u] = sum_emb_sq[u] = 0.0f;
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            const float *t = T_lin + ((long)j * 256 + v) * C;
            const float *e = Emb   + ((long)j * 256 + v) * K;
            for (int c = 0; c < C; c++) acc[c] += t[c];
            for (int u = 0; u < K; u++) {
                float eu = e[u];
                sum_emb[u] += eu;
                sum_emb_sq[u] += eu * eu;
            }
        }
        for (int u = 0; u < K; u++) {
            fm_out[u] = 0.5f * (sum_emb[u] * sum_emb[u] - sum_emb_sq[u]);
        }
        for (int u = 0; u < K; u++) {
            float fu = fm_out[u];
            const float *w = W_fm + (long)u * C;
            for (int c = 0; c < C; c++) acc[c] += fu * w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```