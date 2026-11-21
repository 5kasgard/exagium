```hypothesis
Family: label-embedding-linear. Replaces one-hot/categorical encodings with per-feature embeddings (dim=8) for all 14 columns, then a single linear layer. Captures non-linear bin effects for ordinal features and pairwise interactions via the linear layer on concatenated embeddings. Expects >0.855 accuracy at ~1000 Ir/item (2x label-linear cost, < label-mlp16). Fails if embeddings overfit rare bins or ordinal quantiles need monotonic constraints.
```
```python
"""label-embedding-linear: per-feature embeddings + linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Exports: magic 0x454D4244 (EMBD), d, D, C; float32 E[d][256][D], W[d*D][C], b[C]."""
import json, math, struct, sys

import numpy as np

D = 8          # embedding dimension per feature
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

    # embeddings: [d, 256, D]
    E = (rng.normal(0, 0.01, (d, 256, D))).astype(np.float32)
    # linear weights: [d*D, C], bias: [C]
    W = (rng.normal(0, 0.01, (d * D, C))).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = [E, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi, yy = X[idx], y[idx]
            bsz = xi.shape[0]

            # gather embeddings: [bsz, d, D] -> [bsz, d*D]
            x_emb = E[np.arange(d)[None, :], xi]          # [bsz, d, D]
            x_flat = x_emb.reshape(bsz, d * D)            # [bsz, d*D]

            # forward
            logits = x_flat @ W + b                        # [bsz, C]
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            # gradient dL/dlogits
            grad_logits = probs
            grad_logits[np.arange(bsz), yy] -= 1.0
            grad_logits /= bsz

            # gradients
            grad_W = x_flat.T @ grad_logits + L2 * W       # [d*D, C]
            grad_b = grad_logits.sum(axis=0)               # [C]
            grad_x_flat = grad_logits @ W.T                # [bsz, d*D]
            grad_x_emb = grad_x_flat.reshape(bsz, d, D)    # [bsz, d, D]

            # scatter embedding gradients
            grad_E = np.zeros_like(E)
            np.add.at(grad_E, (np.arange(d)[None, :], xi), grad_x_emb)
            grad_E += L2 * E

            grads = [grad_E, grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # pack weights: magic, d, D, C; E[d][256][D]; W[d*D][C]; b[C]
    magic = 0x454D4244  # 'EMBD'
    out = struct.pack("<4i", magic, d, D, C)
    out += E.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-embedding-linear infer: ./infer weights.bin in.rec N
 * Input record: d bytes (one per feature). Output: 1 byte class index.
 * Weights format: int32 magic(0x454D4244), d, D, C; float32 E[d][256][D], W[d*D][C], b[C]. */
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
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d <= 0 || D <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * D + (long)d * D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *E = (const float *)(hd + 4);
    const float *W = E + (long)d * 256 * D;
    const float *b = W + (long)d * D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_flat = malloc(sizeof(float) * (size_t)d * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // gather embeddings into x_flat
        for (int j = 0; j < d; j++) {
            const float *src = E + ((long)j * 256 + xi[j]) * D;
            float *dst = x_flat + (long)j * D;
            memcpy(dst, src, sizeof(float) * D);
        }

        // linear layer: scores = x_flat @ W + b
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int k = 0; k < d * D; k++) sum += x_flat[k] * W[(long)k * C + c];
            scores[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```