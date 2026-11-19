```hypothesis
Family: label-embed (per-position byte embeddings + linear classifier).  
Why: The 16-byte context has strong positional patterns; one-hot linear (label-linear) uses 4096×65=266k params. Embedding each byte into 8 dimensions per position reduces params to 16×256×8 + 16×8×65 ≈ 41k, acting as a strong regularizer that should improve generalization on 24k training items. Inference cost drops from 266k to 8.3k multiply-adds per item.  
Failure: Dev accuracy ≤ 0.3375 (label-linear) or instructions/item ≥ label-linear (7.4k).
```

```python
"""label-embed: per-position byte embeddings (dim D=8) + linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Weights format: magic 0x454D4244, int32 d, D, C; float32 emb[d][256][D], W[d*D][C], b[C]."""
import json, math, struct, sys

import numpy as np

D = 8
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
    d = X.shape[1]
    rng = np.random.default_rng(SEED)

    # embeddings: (d, 256, D)
    emb = rng.normal(0, 0.01, size=(d, 256, D)).astype(np.float32)
    # linear: (d*D, C)
    W = rng.normal(0, 0.01, size=(d * D, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            B = len(idx)
            # gather embeddings -> (B, d*D)
            emb_out = np.zeros((B, d * D), dtype=np.float32)
            for j in range(d):
                emb_out[:, j * D:(j + 1) * D] = emb[j, xb[:, j], :]
            logits = emb_out @ W + b
            # softmax cross-entropy gradient
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = probs / B
            # gradients
            grad_W = emb_out.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
            grad_emb_out = grad_logits @ W.T  # (B, d*D)
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_emb_out[:, j * D:(j + 1) * D])
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
    # write weights
    out = struct.pack("<4i", 0x454D4244, d, D, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Each input record: 16 bytes. Output: 1 byte class index (argmax). */
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
    if (C < 1 || C > 256 || d != 16 || D != 8) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected_floats = (long)d * 256 * D + (long)d * D * C + C;
    if (nw != 16 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * D;
    const float *b = W + (long)d * D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *vec = malloc(sizeof(float) * d * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // gather embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            memcpy(vec + j * D, e, D * sizeof(float));
        }
        // linear
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int k = 0; k < d * D; k++) {
            float v = vec[k];
            if (v == 0.0f) continue;
            const float *w = W + (long)k * C;
            for (int c = 0; c < C; c++) scores[c] += v * w[c];
        }
        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```