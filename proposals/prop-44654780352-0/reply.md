```hypothesis
Family: label-emb (embedding-based linear model). All 14 columns treated as categorical with learned embeddings (dim=8), summed per record, followed by a linear output layer. This captures non-linear ordinal effects and category similarities with fewer parameters than one-hot MLP. If dev accuracy ≤ 0.8524 (label-linear) or inference cost > label-linear, the hypothesis fails.
```

```python
"""label-emb: embedding-based linear model for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Architecture: per-column embedding tables (256 values x E), sum embeddings, linear output.
Exported weights: magic 0x454D424C, d, E, C; float32 embeddings[d][256][E], W_out[C][E], b_out[C]."""
import json, math, struct, sys

import numpy as np

E = 8
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

    # embeddings: (d, 256, E)
    emb = (rng.uniform(-0.01, 0.01, (d, 256, E)) * math.sqrt(6.0 / E)).astype(np.float32)
    W_out = (rng.uniform(-0.01, 0.01, (C, E)) * math.sqrt(6.0 / (E + C))).astype(np.float32)
    b_out = np.zeros(C, np.float32)

    params = [emb, W_out, b_out]
    state_m = [np.zeros_like(p) for p in params]
    state_v = [np.zeros_like(p) for p in params]
    t = 0

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            B = len(yb)

            # forward: sum embeddings
            sum_emb = np.zeros((B, E), np.float32)
            for j in range(d):
                sum_emb += emb[j, xb[:, j], :]

            logits = sum_emb @ W_out.T + b_out
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            # gradient wrt logits
            grad_logits = probs
            grad_logits[np.arange(B), yb] -= 1.0
            grad_logits /= B

            # gradients
            grad_W_out = grad_logits.T @ sum_emb + L2 * W_out
            grad_b_out = grad_logits.sum(axis=0) + L2 * b_out
            grad_sum_emb = grad_logits @ W_out

            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_sum_emb)
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W_out, grad_b_out]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state_m[i], state_v[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights
    magic = 0x454D424C  # 'LMBE' little-endian
    header = struct.pack("<4i", magic, d, E, C)
    emb_bytes = emb.astype("<f4").tobytes()
    wout_bytes = W_out.astype("<f4").tobytes()
    bout_bytes = b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + emb_bytes + wout_bytes + bout_bytes)


if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: d bytes (one per column). Output: one byte class index.
 * Weights format: magic 0x454D424C, int32 d, E, C; float32 embeddings[d][256][E], W_out[C][E], b_out[C]. */
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
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d <= 0 || E <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)C * E + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W_out = emb + (long)d * 256 * E;
    const float *b_out = W_out + (long)C * E;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = (float *)malloc(sizeof(float) * E);
    float *logits = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int e = 0; e < E; e++) sum_emb[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) sum_emb[e] += eptr[e];
        }
        for (int c = 0; c < C; c++) {
            float acc = b_out[c];
            const float *wptr = W_out + (long)c * E;
            for (int e = 0; e < E; e++) acc += sum_emb[e] * wptr[e];
            logits[c] = acc;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```