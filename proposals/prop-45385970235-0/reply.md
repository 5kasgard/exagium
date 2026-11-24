```hypothesis
Family: label-emb (embedding + linear). Treat all 14 features as categorical with 256 values (quantile bins + unseen). Learn a D=8 embedding per feature-value, concatenate (112-d), then a linear layer to 2 classes. This captures non-linear category similarities that one-hot linear misses, with far fewer parameters than label-mlp16. Expect dev accuracy > 0.8524 (label-linear) and similar speed. Failure: accuracy ≤ label-linear or inference cost > label-mlp16.
```
```python
"""label-emb: per-feature embeddings + linear classifier.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Exports: magic 0x454D424C, int32 d, D, C; float32 emb[d][256][D], W[(d*D)][C], b[C]."""
import json, math, struct, sys

import numpy as np

D = 8                 # embedding dimension per feature
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

    # embeddings: (d, 256, D)
    emb = (rng.normal(0, 1.0 / math.sqrt(D), (d, 256, D))).astype(np.float32)
    # linear weights: (d*D, C), bias: (C,)
    W = (rng.normal(0, 1.0 / math.sqrt(d * D), (d * D, C))).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)

            # forward: gather embeddings and flatten
            B = xb.shape[0]
            # emb_gathered: (B, d, D)
            emb_gathered = emb[xb]          # advanced indexing
            flat = emb_gathered.reshape(B, d * D)   # (B, d*D)
            logits = flat @ W + b            # (B, C)

            # softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = probs / B           # (B, C)

            # gradients
            grad_W = flat.T @ grad_logits + L2 * W          # (d*D, C)
            grad_b = grad_logits.sum(axis=0)                # (C,)
            grad_flat = grad_logits @ W.T                   # (B, d*D)
            grad_emb_gathered = grad_flat.reshape(B, d, D)  # (B, d, D)

            # scatter-add to emb gradients
            grad_emb = np.zeros_like(emb)
            np.add.at(grad_emb, (xb[:, :, None], np.arange(D)), grad_emb_gathered.transpose(0, 2, 1))
            # Note: np.add.at with (B, d) indices and (B, d, D) values needs correct indexing.
            # Simpler: loop over features (d=14 small)
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_emb_gathered[:, j, :])
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # export
    magic = 0x454D424C  # 'EMBL'
    out = struct.pack("<4i", magic, d, D, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index. */
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
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 14) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * D + (long)d * D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * D;
    const float *b = W + (long)d * D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * D;
            float *dst = flat + j * D;
            memcpy(dst, src, D * sizeof(float));
        }

        // linear layer
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int w = 0; w < d * D; w++) sum += flat[w] * W[(long)w * C + c];
            scores[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```