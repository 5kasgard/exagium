```hypothesis
Family: label-adjpoly2 (linear on z-scored pixels + 112 adjacent pairwise products).
Why: optdigits 8x8 strokes have local structure (edges, corners) captured by adjacent pixel products; linear model misses these interactions. Adds only 112 features (2.75x linear's 64), inference ~14k Ir/item (vs linear 5k, mlp16 8.7k).
Failure: If dev accuracy <= 0.9613 (label-linear) or Ir/item > 20000, the cost/accuracy tradeoff is not worthwhile.
```

```python
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
    C = card["classes"]
    d = card["input"]["bytes"]  # 64
    n = X.shape[0]

    # Per-pixel z-scoring
    Xf = X.astype(np.float32)
    mean = Xf.mean(axis=0)
    std = Xf.std(axis=0)
    std[std == 0] = 1.0
    inv_std = 1.0 / std
    Z = (Xf - mean) * inv_std  # (n, 64)

    # Adjacent pairs: horizontal (56) + vertical (56) = 112
    pairs = []
    for r in range(8):
        for c in range(7):
            i = r * 8 + c
            j = r * 8 + c + 1
            pairs.append((i, j))
    for r in range(7):
        for c in range(8):
            i = r * 8 + c
            j = (r + 1) * 8 + c
            pairs.append((i, j))
    pairs = np.array(pairs, dtype=np.int32)  # (112, 2)
    n_pairs = len(pairs)

    # Pair features: product of z-scored values
    Z_pairs = Z[:, pairs[:, 0]] * Z[:, pairs[:, 1]]  # (n, 112)

    # Concatenate: 64 linear + 112 pairwise = 176 features
    Z_all = np.concatenate([Z, Z_pairs], axis=1).astype(np.float32)
    k = Z_all.shape[1]  # 176

    # Train linear softmax with Adam (same hyperparams as label-linear)
    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z_all[idx], y[idx]
            logits = z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad_logits = (probs / len(yy)).astype(np.float32)
            grads = [z.T @ grad_logits + L2 * W, grad_logits.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export weights: magic, d, n_pairs, C; mean[d], inv_std[d], linear_w[d*C], pair_w[n_pairs*C], bias[C]
    magic = 0x41444A32  # "ADJ2"
    linear_w = W[:d]
    pair_w = W[d:]
    out = struct.pack("<4i", magic, d, n_pairs, C)
    out += mean.astype("<f4").tobytes()
    out += inv_std.astype("<f4").tobytes()
    out += linear_w.astype("<f4").tobytes()
    out += pair_w.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
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
   