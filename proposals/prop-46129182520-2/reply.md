```hypothesis
Family: label-emb (low-rank logistic regression via per-position character embeddings).
Why it should work: TinyShakespeare character prediction benefits from modeling character similarities (e.g., vowels, punctuation clusters). One-hot linear models (label-linear) use 16×256×65=266K params and cannot share statistical strength across characters. An embedding model factorizes the weight matrix: each (position, byte) gets an E-dim vector (16×256×E), and each class gets an E-dim vector (65×E). For E=8 this is only 33K params, acts as a strong regularizer, captures character analogies, and reduces inference ops (81E vs 1040 multiply-adds). Tuning on the last 20% of train.in (E=8, lr=0.05, Adam, L2=1e-4) should yield >34% dev accuracy at lower CPU cost.
Failure signal: Dev accuracy ≤ 0.3375 (label-linear) or inference ns/item ≥ label-linear.
```

```python
"""label-emb: Low-rank logistic regression via per-position character embeddings.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Embedding dim E=8. Adam lr=0.05, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Exports: magic 0x4C424D45, d, E, C; float32 Emb[d][256][E], U[C][E], b[C]."""
import json, math, struct, sys

import numpy as np

E = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.05, 1e-4, 64, 600000, 100


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
    # Emb: [d, 256, E], U: [C, E], b: [C]
    Emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    U = (rng.normal(0, 0.01, (C, E))).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = [Emb, U, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # Forward: Z = sum_j Emb[j, xb[:, j]]  -> [bs, E]
            Z = np.zeros((bs, E), np.float32)
            for j in range(d):
                Z += Emb[j, xb[:, j]]

            logits = Z @ U.T + b  # [bs, C]
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs

            # Backward
            dU = probs.T @ Z + L2 * U
            db = probs.sum(axis=0)
            dZ = probs @ U  # [bs, E]

            dEmb = np.zeros_like(Emb)
            for j in range(d):
                # scatter-add dZ into dEmb[j, xb[:, j]]
                np.add.at(dEmb[j], xb[:, j], dZ)
            dEmb += L2 * Emb

            grads = [dEmb, dU, db]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export weights
    # Magic, d, E, C
    out = struct.pack("<4i", 0x4C424D45, d, E, C)
    out += Emb.astype("<f4").tobytes()
    out += U.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emb: Low-rank logistic regression via per-position character embeddings.
 * ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins).
 * Weights format: int32 magic 0x4C424D45, d, E, C; float32 Emb[d][256][E], U[C][E], b[C]. */
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
    if (nw < 16 || hd[0] != 0x4C424D45) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16 || E < 1 || E > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)C * E + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *Emb = (const float *)(hd + 4);
    const float *U = Emb + (long)d * 256 * E;
    const float *b = U + (long)C * E;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *Z = malloc(sizeof(float) * E);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int e = 0; e < E; e++) Z[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *emb = Emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) Z[e] += emb[e];
        }

        for (int c = 0; c < C; c++) {
            float sum = b[c];
            const float *uc = U + (long)c * E;
            for (int e = 0; e < E; e++) sum += Z[e] * uc[e];
            logits[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```