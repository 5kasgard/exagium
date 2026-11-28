```hypothesis
Family: label-polyadj (logistic regression with adjacent-pixel pairwise interactions).
Why: Optdigits 8x8 digits have local spatial structure; adding products of neighboring pixels captures stroke curvature and thickness, improving linear separability with only 176 features.
Failure: If dev accuracy ≤ 0.9613 (label-linear) or inference cost exceeds label-linear by >20%.
```

```python
"""label-polyadj: logistic regression with adjacent-pixel pairwise interactions.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x504F4C59  # "POLY" little-endian

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_pairs():
    """Return list of (idx1, idx2) for horizontal then vertical adjacent pairs in 8x8."""
    pairs = []
    for r in range(8):
        for c in range(7):
            pairs.append((r*8 + c, r*8 + c + 1))
    for r in range(7):
        for c in range(8):
            pairs.append((r*8 + c, (r+1)*8 + c))
    return pairs

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert d == 64, "This family expects 64-byte input (8x8)."

    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]

    pairs = build_pairs()
    F = d + len(pairs)  # 64 + 112 = 176

    # Compute raw features
    X_raw = np.zeros((n, F), dtype=np.float32)
    X_raw[:, :d] = X_bytes.astype(np.float32)
    for k, (i, j) in enumerate(pairs):
        X_raw[:, d + k] = X_bytes[:, i].astype(np.float32) * X_bytes[:, j].astype(np.float32)

    # Normalize features
    mean = X_raw.mean(axis=0)
    std = X_raw.std(axis=0)
    std[std == 0] = 1.0
    inv_std = 1.0 / std
    Z = (X_raw - mean) * inv_std  # (n, F)

    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (F, C)) * math.sqrt(6.0 / F)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # softmax gradient
            scores = z @ W + b
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad = (probs / len(yy)).astype(np.float32)

            grad_W = z.T @ grad + L2 * W
            grad_b = grad.sum(axis=0)

            t += 1
            for p, gr, (m, v) in zip(params, [grad_W, grad_b], state):
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    # Format: magic(u32), F(u32), C(u32), mean[F](f32), inv_std[F](f32), W[F*C](f32), b[C](f32)
    out = struct.pack("<3I", MAGIC, F, C)
    out += mean.astype("<f4").tobytes()
    out += inv_std.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-polyadj inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (8x8 pixels). Output record: 1 byte (class 0..9). */
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
    unsigned char *wb = slurp(argv[1], &nw);
    uint32_t *hd = (uint32_t *)wb;
    if (nw < 12 || hd[0] != 0x504F4C59) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int F = hd[1], C = hd[2];
    if (C < 1 || C > 256 || F != 176) { fprintf(stderr, "bad weights dimensions\n"); return 3; }
    long expected = 12L + 4L * F * 2 + 4L * F * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(hd + 3);
    const float *inv_std = mean + F;
    const float *W = inv_std + F;
    const float *b = W + (long)F * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    /* Precomputed adjacent pairs for 8x8: horizontal then vertical */
    uint8_t pair_i[112], pair_j[112];
    int pidx = 0;
    for (int r = 0; r < 8; r++) for (int c = 0; c < 7; c++) { pair_i[pidx] = r*8+c; pair_j[pidx] = r*8+c+1; pidx++; }
    for (int r = 0; r < 7; r++) for (int c = 0; c < 8; c++) { pair_i[pidx] = r*8+c; pair_j[pidx] = (r+1)*8+c; pidx++; }

    float *feat = malloc(sizeof(float) * F);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }

        /* Raw features */
        for (int f = 0; f < 64; f++) feat[f] = xi[f];
        for (int p = 0; p < 112; p++) feat[64 + p] = (float)xi[pair_i[p]] * (float)xi[pair_j[p]];

        /* Normalize */
        for (int f = 0; f < F; f++) feat[f] = (feat[f] - mean[f]) * inv_std[f];

        /* Linear scores */
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int f = 0; f < F; f++) {
            float fv = feat[f];
            const float *wf = W + (long)f * C;
            for (int c = 0; c < C; c++) scores[c] += fv * wf[c];
        }

        /* Argmax (first max wins) */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(feat); free(scores); free(wb); if (x) free(x);
    return 0;
}
```