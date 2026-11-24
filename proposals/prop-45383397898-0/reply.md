```hypothesis
Family: label-fm (Factorization Machine with class-specific embeddings).
Why: Adult income prediction relies heavily on feature interactions (e.g., education×occupation, marital-status×relationship). A factorization machine captures all pairwise interactions via low-rank embeddings (k=8) while sharing statistical strength across the 14 fields. With only 2 classes, class-specific embeddings add minimal parameters (~143 KB weights) but allow interaction patterns to differ per class. Inference computes interactions in O(d·k) via the (sum² - sum of squares) identity, keeping per-item cost well below label-linear's 719 Ir/item.
Failure signal: Dev accuracy ≤ 0.8524 (label-linear) or Ir/item ≥ 700.
```

```python
"""label-fm: Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, numpy only. Exports per-field lookup tables.
Format: int32 magic 0x464D4C31 ('FML1'), d, k, C; float32 bias[C], linear[C][d*256], emb[C][d*256][k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 64, 600000, 100


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

    # Parameters: bias[C], linear[C, d, 256], emb[C, d, 256, K]
    bias = np.zeros(C, np.float32)
    linear = np.zeros((C, d, 256), np.float32)
    emb = (rng.normal(0, 0.01, (C, d, 256, K))).astype(np.float32)

    params = [bias, linear, emb]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # [B, d]
            yb = y[idx]          # [B]
            B = xb.shape[0]

            # Forward pass
            logits = np.zeros((B, C), np.float32)
            sum_emb = np.zeros((B, C, K), np.float32)    # for backward
            for c in range(C):
                # linear term
                lin = linear[c, np.arange(d), xb.T].sum(axis=0)   # [B]
                # embeddings: [B, d, K]
                e = emb[c, np.arange(d), xb.T].transpose(1, 0, 2)
                se = e.sum(axis=1)                 # [B, K]
                sesq = (e * e).sum(axis=1)         # [B, K]
                inter = 0.5 * ((se * se - sesq).sum(axis=1))   # [B]
                logits[:, c] = bias[c] + lin + inter
                sum_emb[:, c] = se

            # Softmax + loss gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            probs /= B  # [B, C]

            # Backward: accumulate gradients
            dbias = probs.sum(axis=0).astype(np.float32)                     # [C]
            dlinear = np.zeros_like(linear)
            demb = np.zeros_like(emb)

            for c in range(C):
                gc = probs[:, c]                      # [B]
                # linear gradient
                for j in range(d):
                    vals = xb[:, j]
                    np.add.at(dlinear[c, j], vals, gc)
                # embedding gradient
                # For each feature j, factor f: dV = gc * (sum_emb[:,c,f] - emb[c,j,vals,f])
                # sum_emb[:,c] is [B, K]
                for j in range(d):
                    vals = xb[:, j]
                    se_c = sum_emb[:, c]              # [B, K]
                    # broadcast gc [B] to [B, K]
                    diff = (se_c - emb[c, j, vals]) * gc[:, None]   # [B, K]
                    np.add.at(demb[c, j], vals, diff)

            # L2 regularization
            dlinear += L2 * linear
            demb += L2 * emb

            grads = [dbias, dlinear, demb]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Flatten for export: bias[C], linear[C][d*256], emb[C][d*256][K]
    linear_flat = linear.reshape(C, d * 256)
    emb_flat = emb.reshape(C, d * 256, K)

    magic = 0x464D4C31  # 'FML1'
    header = struct.pack("<4i", magic, d, K, C)
    body = (bias.astype("<f4").tobytes() +
            linear_flat.astype("<f4").tobytes() +
            emb_flat.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Factorization Machine with class-specific embeddings.
 * Input: d bytes per record. Output: one byte class index (argmax). */
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
    if (C < 1 || C > 256 || K < 1 || K > 64 || d < 1 || d > 256) {
        fprintf(stderr, "bad weights dims\n"); return 3;
    }
    long expected = 16L + 4L * (C + (long)C * d * 256 + (long)C * d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld != %ld\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + C;
    const float *emb = linear + (long)C * d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(C * sizeof(float));
    float *sum_emb = malloc(C * K * sizeof(float));
    float *sum_emb_sq = malloc(C * K * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            logits[c] = bias[c];
            for (int f = 0; f < K; f++) {
                sum_emb[c * K + f] = 0.0f;
                sum_emb_sq[c * K + f] = 0.0f;
            }
        }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            long base = (long)j * 256 + v;
            for (int c = 0; c < C; c++) {
                logits[c] += linear[(long)c * d * 256 + base];
                const float *e = emb + ((long)c * d * 256 + base) * K;
                float *se = sum_emb + c * K;
                float *sesq = sum_emb_sq + c * K;
                for (int f = 0; f < K; f++) {
                    float val = e[f];
                    se[f] += val;
                    sesq[f] += val * val;
                }
            }
        }

        for (int c = 0; c < C; c++) {
            float *se = sum_emb + c * K;
            float *sesq = sum_emb_sq + c * K;
            float inter = 0.0f;
            for (int f = 0; f < K; f++) inter += se[f] * se[f] - sesq[f];
            logits[c] += 0.5f * inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    return 0;
}
```