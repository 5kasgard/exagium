```hypothesis
Family: label-lookup. A linear model with a per-feature lookup table over all 256 possible byte values (treating ordinal and categorical uniformly). This captures non-linear effects on quantized ordinal features while keeping inference extremely cheap (14 table lookups and 28 additions per record). It should beat label-linear (0.8524) because ordinal features are not constrained to be linear in the quantile bins. If it does not beat label-linear, it would indicate that the ordinal features are well-approximated by linear functions after quantile binning, or that the extra capacity overfits despite regularization.
```
```python
"""label-lookup: linear model with per-feature 256-value lookup tables.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format: int32 magic 0x4C4F4F4B, d, C; float32 W[d][256][C], b[C]."""
import json, math, struct, sys

import numpy as np

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
    d = X.shape[1]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # Parameters: W[d, 256, C], b[C]
    W = (rng.uniform(-0.01, 0.01, (d, 256, C))).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # Forward: scores = b + sum_j W[j, xb[:,j], :]
            scores = b.copy()
            for j in range(d):
                scores += W[j, xb[:, j], :]

            # Softmax gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grad_scores = (probs / bs).astype(np.float32)   # (bs, C)

            # Gradients
            grad_W = np.zeros_like(W)
            grad_b = grad_scores.sum(axis=0)
            for j in range(d):
                np.add.at(grad_W[j], xb[:, j], grad_scores)
            grad_W += L2 * W
            grads = [grad_W, grad_b]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x4C4F4F4B  # "LOOK"
    out = struct.pack("<3i", magic, d, C)
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-lookup inference: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: one byte class index (argmax). */
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
    if (nw < 12 || hd[0] != 0x4C4F4F4B) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2];
    if (C < 1 || C > 256) { fprintf(stderr, "bad class count\n"); return 3; }
    long expected = 12 + (long)d * 256 * C * 4 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W = (const float *)(hd + 3);
    const float *b = W + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(scores, b, C * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *w = W + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```