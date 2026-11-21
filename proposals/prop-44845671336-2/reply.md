```hypothesis
Family: label-fm (factorization machine with linear output).  
Each of the 16 context positions gets a learned embedding (size E). The model computes the sum of embeddings S and the sum of squared embeddings Q, then forms pairwise interactions as 0.5*(S^2 - Q). A linear layer on [S, pairwise] predicts the next character.  
This captures pairwise position interactions cheaply (O(16*E) inference).  
If dev accuracy ≤ label-linear (0.3375) or inference cost exceeds 2× label-linear, the idea fails.
```

```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports: magic 0x464D3130, d, E, C; float32 embeddings[d][256][E], W[2*E][C], b[C]."""
import json, math, struct, sys

import numpy as np

E = 16
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 50
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8


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
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # embeddings: (d, 256, E)
    emb = (rng.normal(0, 0.1, (d, 256, E))).astype(np.float32)
    # output weights: (2*E, C)
    W = (rng.normal(0, 0.1, (2 * E, C))).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    # Adam state
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # forward: gather embeddings for each position
            S = np.zeros((bs, E), dtype=np.float32)
            Q = np.zeros((bs, E), dtype=np.float32)
            for p in range(d):
                e = emb[p, xb[:, p]]          # (bs, E)
                S += e
                Q += e * e
            P = 0.5 * (S * S - Q)            # (bs, E)
            feats = np.concatenate([S, P], axis=1)   # (bs, 2E)
            logits = feats @ W + b            # (bs, C)

            # softmax cross-entropy
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grad_logits = probs / bs          # (bs, C)

            # gradients for output layer
            grad_W = feats.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)

            # gradients for features
            grad_feats = grad_logits @ W.T    # (bs, 2E)
            grad_S = grad_feats[:, :E]
            grad_P = grad_feats[:, E:]

            # gradients for embeddings
            grad_emb = np.zeros_like(emb)
            for p in range(d):
                e = emb[p, xb[:, p]]          # (bs, E)
                # dL/de = grad_S + grad_P * (S - e)
                g = grad_S + grad_P * (S - e)
                np.add.at(grad_emb[p], xb[:, p], g)
            grad_emb += L2 * emb

            # Adam update
            t += 1
            for param, grad, m, v in [
                (emb, grad_emb, m_emb, v_emb),
                (W, grad_W, m_W, v_W),
                (b, grad_b, m_b, v_b)
            ]:
                m *= BETA1; m += (1 - BETA1) * grad
                v *= BETA2; v += (1 - BETA2) * grad * grad
                m_hat = m / (1 - BETA1 ** t)
                v_hat = v / (1 - BETA2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + EPS)

    # write weights
    magic = 0x464D3130  # 'FM10'
    out = struct.pack("<4i", magic, d, E, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index.
 * Weights format: magic 0x464D3130, int32 d, E, C; float32 emb[d][256][E], W[2*E][C], b[C]. */
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
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 16 || C != 65 || E <= 0 || E > 128) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + 2L * E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + 2L * E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *S = malloc(sizeof(float) * E);
    float *Q = malloc(sizeof(float) * E);
    float *P = malloc(sizeof(float) * E);
    float *feats = malloc(sizeof(float) * 2 * E);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int e = 0; e < E; e++) { S[e] = 0.0f; Q[e] = 0.0f; }
        for (int p = 0; p < d; p++) {
            const float *eptr = emb + ((long)p * 256 + xi[p]) * E;
            for (int e = 0; e < E; e++) {
                float v = eptr[e];
                S[e] += v;
                Q[e] += v * v;
            }
        }
        for (int e = 0; e < E; e++) {
            P[e] = 0.5f * (S[e] * S[e] - Q[e]);
        }
        for (int e = 0; e < E; e++) {
            feats[e] = S[e];
            feats[E + e] = P[e];
        }

        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int f = 0; f < 2 * E; f++) {
            float wf = feats[f];
            if (wf == 0.0f) continue;
            const float *wptr = W + (long)f * C;
            for (int c = 0; c < C; c++) scores[c] += wf * wptr[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```