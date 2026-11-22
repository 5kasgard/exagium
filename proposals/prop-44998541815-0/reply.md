```hypothesis
Family: label-fm (Factorization Machine with shared embeddings, rank k=4).
Why: The Adult dataset has known feature interactions (e.g., education × occupation, marital-status × relationship). A rank-4 FM adds only ~14k embedding parameters and ~100 FLOPs/item to the linear model, capturing all pairwise interactions via the FM trick (sum^2 - sum_of_squares) without per-pair tables. This should beat label-linear (0.8524) by 0.5–1% accuracy while staying well below label-mlp16's 2325 Ir/item.
Failure signal: Dev accuracy ≤ 0.853 or Ir/item > 1500 would indicate the FM interactions are not helping or the implementation is too slow.
```
```python
"""Factorization Machine for tabular label tasks: linear + shared rank-k pairwise interactions.
python3 -s -P train.py card.json train.in train.key weights.bin  (deterministic: seed 1, single thread)
Exports: int32 magic 0x464D3130, d, k, C; float32 L[d][256][C], V[d][256][k]."""
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
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Linear weights L[j][v][c] and embeddings V[j][v][f]
    L = np.zeros((d, 256, C), dtype=np.float32)
    V = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    mL, vL = np.zeros_like(L), np.zeros_like(L)
    mV, vV = np.zeros_like(V), np.zeros_like(V)
    t = 0

    def forward(Xb):
        """Xb: (bs, d) uint8 -> logits (bs, C)"""
        bs = Xb.shape[0]
        logits = np.zeros((bs, C), dtype=np.float32)
        sum_v = np.zeros((bs, K), dtype=np.float32)
        sum_v2 = np.zeros((bs, K), dtype=np.float32)
        for j in range(d):
            v = Xb[:, j]
            logits += L[j, v]                 # (bs, C)
            v_emb = V[j, v]                   # (bs, K)
            sum_v += v_emb
            sum_v2 += v_emb * v_emb
        fm = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1, keepdims=True)  # (bs, 1)
        logits += fm                          # broadcast to both classes
        return logits

    def softmax_grad(logits, yb):
        """Cross-entropy gradient wrt logits: (P - one_hot) / bs"""
        bs = logits.shape[0]
        maxl = logits.max(axis=1, keepdims=True)
        exp = np.exp(logits - maxl)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(bs), yb] -= 1.0
        return probs / bs

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            bs = len(yb)

            # Forward
            logits = np.zeros((bs, C), dtype=np.float32)
            sum_v = np.zeros((bs, K), dtype=np.float32)
            sum_v2 = np.zeros((bs, K), dtype=np.float32)
            for j in range(d):
                v = Xb[:, j]
                logits += L[j, v]
                v_emb = V[j, v]
                sum_v += v_emb
                sum_v2 += v_emb * v_emb
            fm = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1, keepdims=True)
            logits += fm

            # Gradient
            g = softmax_grad(logits, yb)          # (bs, C)
            # dL/dlogits = g
            # dlogits/dL[j,v,c] = 1 if Xb[i,j]==v else 0
            # dlogits/dV[j,v,f] = (sum_v[f] - V[j,v,f]) if Xb[i,j]==v else 0
            grad_L = np.zeros_like(L)
            grad_V = np.zeros_like(V)
            for j in range(d):
                v = Xb[:, j]
                np.add.at(grad_L[j], v, g)                     # (256, C)
                v_emb = V[j, v]                                # (bs, K)
                # sum_v - v_emb per example
                diff = sum_v - v_emb                           # (bs, K)
                np.add.at(grad_V[j], v, diff * g[:, 0:1])      # fm adds same to both classes, so g[:,0] == g[:,1]? No, fm is added to both logits equally, so dlogits_c/dV = d(fm)/dV. g_c = dL/dlogits_c. Sum_c g_c * dlogits_c/dV = (sum_c g_c) * d(fm)/dV.
                # Actually: logits_c = linear_c + fm. dlogits_c/dV = d(fm)/dV.
                # So dLoss/dV = sum_c g_c * d(fm)/dV = (g.sum(axis=1)) * d(fm)/dV.
                # d(fm)/dV[j,v,f] = sum_v[f] - V[j,v,f] for examples where Xb[:,j]==v.
            # Wait, need to recompute sum_v without the current example's contribution? No, fm = 0.5 * (sum_v^2 - sum_v2).
            # d(fm)/dV_ijf = sum_v[f] - V_ijf. This is correct for each example i.
            # So for each example i, for its feature j, value v, the gradient for V[j,v] is (sum_v[i] - V[j,v]).
            # We can accumulate per example.

            # Let's redo gradient computation more carefully:
            # For each example i, for each field j, value v = Xb[i,j]:
            #   dL/dV[j,v,f] = (sum_v[i,f] - V[j,v,f]) * sum_c g[i,c]
            #   dL/dL[j,v,c] = g[i,c]
            # We'll compute sum_c g[i,c] first.

            g_sum = g.sum(axis=1)  # (bs,)

            grad_L = np.zeros_like(L)
            grad_V = np.zeros_like(V)
            for j in range(d):
                v = Xb[:, j]
                # Linear gradient
                for c in range(C):
                    np.add.at(grad_L[j, :, c], v, g[:, c])
                # FM gradient: (sum_v - v_emb) * g_sum[:, None]
                v_emb = V[j, v]
                diff = sum_v - v_emb
                scaled = diff * g_sum[:, None]
                for f in range(K):
                    np.add.at(grad_V[j, :, f], v, scaled[:, f])

            # L2 regularization
            grad_L += L2 * L
            grad_V += L2 * V

            t += 1
            # Adam update
            for param, grad, m, v in [(L, grad_L, mL, vL), (V, grad_V, mV, vV)]:
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x464D3130  # "FM10"
    out = struct.pack("<4i", magic, d, K, C)
    out += L.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d bytes per record (one per field, 0..255).
 * Output: one byte per record (class index 0 or 1).
 * Weights format: int32 magic=0x464D3130, d, k, C; float32 L[d][256][C], V[d][256][k]. */
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || k < 1 || k > 16 || d != 14) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * C + (long)d * 256 * k);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, want %ld\n", nw, expected); return 3; }

    const float *L = (const float *)(hd + 4);
    const float *V = L + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t reclen; const unsigned char *xi = next_rec(x, nx, &pos, &reclen);
        if ((int)reclen != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, reclen, d); return 3; }

        float logit[2] = {0.0f, 0.0f};
        float sum_v[16];     /* k <= 16 */
        float sum_v2[16];
        for (int f = 0; f < k; f++) { sum_v[f] = 0.0f; sum_v2[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *Ljv = L + ((long)j * 256 + v) * C;
            logit[0] += Ljv[0];
            logit[1] += Ljv[1];
            const float *Vjv = V + ((long)j * 256 + v) * k;
            for (int f = 0; f < k; f++) {
                float ve = Vjv[f];
                sum_v[f]  += ve;
                sum_v2[f] += ve * ve;
            }
        }
        float fm = 0.0f;
        for (int f = 0; f < k; f++) fm += sum_v[f] * sum_v[f] - sum_v2[f];
        fm *= 0.5f;
        logit[0] += fm;
        logit[1] += fm;

        unsigned char best = (logit[1] > logit[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```