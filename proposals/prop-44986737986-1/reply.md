```hypothesis
Family: label-1hot (per-pixel value lookup table). Each of the 64 pixels has 17 possible values (0..16); we learn a 10-d weight vector for every (pixel, value) pair plus a class bias. This captures non-monotonic pixel effects that linear z-scoring misses, while inference remains a simple 64×10 accumulate (≈640 adds/item). Trained with Adam (same schedule as label-linear) on the one-hot expanded features. Expect to beat label-linear (0.9613) and approach label-mlp16 (0.9770) at ~5k Ir/item. Failure: dev accuracy ≤ 0.9613.
```

```python
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x4C4F4F4B  # "LOOK"

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
    d, V, C = X.shape[1], 17, card["classes"]          # 64, 17, 10
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # table[d, V, C] and bias[C]
    scale = math.sqrt(6.0 / (d * V))
    table = (rng.uniform(-1, 1, (d, V, C)) * scale).astype(np.float32)
    bias  = np.zeros(C, np.float32)

    # Adam state
    m_table, v_table = np.zeros_like(table), np.zeros_like(table)
    m_bias,  v_bias  = np.zeros_like(bias),  np.zeros_like(bias)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # forward: scores = bias + sum_j table[j, xb[:,j], :]
            scores = np.tile(bias, (bsz, 1))
            for j in range(d):
                scores += table[j, xb[:, j], :]

            # softmax + gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores, dtype=np.float32)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yb] -= 1.0
            probs /= bsz                                      # dL/dscores

            # gradients
            grad_table = np.zeros_like(table)
            grad_bias  = probs.sum(axis=0).astype(np.float32)
            for j in range(d):
                vals = xb[:, j]
                for v in range(V):
                    mask = (vals == v)
                    if mask.any():
                        grad_table[j, v] = probs[mask].sum(axis=0)

            # L2
            grad_table += L2 * table
            grad_bias  += L2 * bias

            # Adam update
            t += 1
            for param, grad, m, v in [(table, grad_table, m_table, v_table),
                                      (bias,  grad_bias,  m_bias,  v_bias)]:
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights
    out = struct.pack("<4i", MAGIC, d, V, C)
    out += table.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
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
    if (nw < 16 || hd[0] != 0x4C4F4F4B) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], V = hd[2], C = hd[3];
    if (C < 1 || C > 256 || V != 17 || d != 64) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * V * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *table = (const float *)(hd + 4);
    const float *bias  = table + (long)d * V * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(scores, bias, C * sizeof(float));
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            const float *row = table + ((long)j * V + v) * C;
            for (int c = 0; c < C; c++) scores[c] += row[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```