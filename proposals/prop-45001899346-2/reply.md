```hypothesis
Family: label-emb (positional byte embeddings + linear classifier).  
Why: The 16-byte context has strong local dependencies; one-hot linear (label-linear) ignores byte similarity and position interactions. Learned embeddings capture semantic similarity between characters and position-specific patterns with far fewer parameters (16*256*D + 16*D*65 vs 4096*65), reducing overfitting on 24k samples and cutting inference cost (memory bandwidth and FLOPs) by ~30x.  
Failure signal: If dev accuracy ≤ 0.3375 or instructions/item ≥ label-linear's 7359, the factorization is too restrictive or hyperparameters poorly chosen.
```

```python
"""label-emb: positional byte embeddings + linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format (little-endian):
  uint32 magic 0x454D424C ('LBME')
  int32 d (16), int32 D (embedding dim), int32 C (65)
  float32 emb[d][256][D]   (position, byte, dim)
  float32 W_out[d*D][C]
  float32 b_out[C]
Hyperparameters chosen on tuning split (last 20% of train.in): D=8, lr=0.01, L2=1e-4, batch=64, epochs=50.
"""
import json, math, struct, sys

import numpy as np

# Hyperparameters (fixed after tuning)
D = 8
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 50
SAMPLES = 600000  # same as label-linear for comparable compute budget

MAGIC = 0x454D424C  # 'LBME' little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert d == 16 and C == 65

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize embeddings: [d, 256, D]
    emb = (rng.normal(0, 1.0 / math.sqrt(256), size=(d, 256, D))).astype(np.float32)
    # Initialize output weights: [d*D, C]
    W_out = (rng.normal(0, 1.0 / math.sqrt(d * D), size=(d * D, C))).astype(np.float32)
    b_out = np.zeros(C, dtype=np.float32)

    params = [emb, W_out, b_out]
    # Adam state: m, v for each param
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # [B, d]
            yb = y[idx]          # [B]

            # Forward: gather embeddings -> [B, d*D]
            # emb[xb] gives [B, d, D]; reshape to [B, d*D]
            h = emb[xb].reshape(-1, d * D)   # [B, d*D]
            logits = h @ W_out + b_out       # [B, C]

            # Softmax + cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_logits = (probs / len(yb)).astype(np.float32)   # [B, C]

            # Gradients
            grad_W_out = h.T @ grad_logits + L2 * W_out          # [d*D, C]
            grad_b_out = grad_logits.sum(axis=0)                 # [C]
            grad_h = grad_logits @ W_out.T                       # [B, d*D]
            grad_emb = np.zeros_like(emb)                        # [d, 256, D]
            # For each position, accumulate gradient for the seen bytes
            for pos in range(d):
                bytes_pos = xb[:, pos]          # [B]
                # grad_h[:, pos*D:(pos+1)*D] is [B, D]
                np.add.at(grad_emb[pos], bytes_pos, grad_h[:, pos * D:(pos + 1) * D])
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W_out, grad_b_out]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= beta1; m += (1 - beta1) * gr
                v *= beta2; v += (1 - beta2) * gr * gr
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Pack weights
    out = struct.pack("<4i", MAGIC, d, D, C)
    out += emb.astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    out += b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Each input record is 16 bytes; output is one byte (class index).
 * Weights format (little-endian):
 *   uint32 magic 0x454D424C ('LBME')
 *   int32 d (16), int32 D (embedding dim), int32 C (65)
 *   float32 emb[d][256][D]
 *   float32 W_out[d*D][C]
 *   float32 b_out[C]
 */
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
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C != 65 || D <= 0 || D > 64) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * D + (long)d * D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W_out = emb + (long)d * 256 * D;
    const float *b_out = W_out + (long)d * D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * d * D);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build h = concatenation of embeddings for each position
        for (int pos_i = 0; pos_i < d; pos_i++) {
            const float *e = emb + ((long)pos_i * 256 + xi[pos_i]) * D;
            float *dst = h + pos_i * D;
            for (int j = 0; j < D; j++) dst[j] = e[j];
        }

        // logits = h @ W_out + b_out
        for (int c = 0; c < C; c++) logits[c] = b_out[c];
        for (int idx = 0; idx < d * D; idx++) {
            float hv = h[idx];
            if (hv == 0.0f) continue;
            const float *w = W_out + (long)idx * C;
            for (int c = 0; c < C; c++) logits[c] += hv * w[c];
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(h); free(logits);
    return 0;
}
```