```hypothesis
Family: label-embed-mlp (positional embeddings + one ReLU layer).
Why: Character-level language modeling benefits from dense embeddings that capture semantic similarity (e.g., letters vs punctuation) which one-hot linear models cannot. A small hidden layer (H=64) on top of 8-dim embeddings per position (16*8=128 inputs) adds minimal inference cost (≈5k FLOPs/item) while modeling pairwise interactions via the hidden layer. Precomputing the embedding→hidden projection into a lookup table makes inference pure table lookups + one dense matvec.
Failure signal: Dev accuracy ≤ 0.3375 (label-linear) or instructions/item ≥ 2× label-linear.
```

```python
"""label-embed-mlp: positional embeddings (8-dim) + one ReLU layer (H=64) for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, numpy+stdlib only.
Weights format (little-endian):
  magic(0x454D424D), d, H, C, Edim (int32)
  emb[d][256][Edim] (float32)
  W1[d*Edim][H], b1[H], W2[H][C], b2[C] (float32)"""
import json, math, struct, sys

import numpy as np

EDIM, H, SEED = 8, 64, 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100

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
    rng = np.random.default_rng(SEED)

    # params
    emb = (rng.normal(0, 1/math.sqrt(EDIM), (d, 256, EDIM))).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(2.0/(d*EDIM)), (d*EDIM, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0/H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # forward
            # gather embeddings: [bs, d, EDIM]
            e = emb[:, xb.T, :].transpose(1, 0, 2)  # [bs, d, EDIM]
            flat = e.reshape(bs, d*EDIM)            # [bs, d*EDIM]
            h_pre = flat @ W1 + b1                   # [bs, H]
            h = np.maximum(h_pre, 0)
            logits = h @ W2 + b2                     # [bs, C]

            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grad_out = (probs / bs).astype(np.float32)   # [bs, C]

            # backprop
            grad_W2 = h.T @ grad_out + L2 * W2
            grad_b2 = grad_out.sum(axis=0)
            grad_h = grad_out @ W2.T
            grad_h[h_pre <= 0] = 0.0
            grad_W1 = flat.T @ grad_h + L2 * W1
            grad_b1 = grad_h.sum(axis=0)
            grad_flat = grad_h @ W1.T                     # [bs, d*EDIM]
            grad_e = grad_flat.reshape(bs, d, EDIM)       # [bs, d, EDIM]

            # accumulate embedding gradients
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_e[:, j, :])

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * g
                v *= 0.999; v += 0.001 * g * g
                m_hat = m / (1 - 0.9**t)
                v_hat = v / (1 - 0.999**t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights
    magic = 0x454D424D  # 'EMBM'
    hdr = struct.pack("<5i", magic, d, H, C, EDIM)
    out = hdr
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights: magic(0x454D424D), d, H, C, Edim; emb[d][256][Edim]; W1[d*Edim][H]; b1[H]; W2[H][C]; b2[C] (all float32 LE).
 * Precomputes T[j][byte][h] = sum_k emb[j][byte][k] * W1[j*Edim+k][h] at load time. */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], Edim = hd[4];
    if (d != 16 || C != 65 || H <= 0 || Edim <= 0) { fprintf(stderr, "bad header\n"); return 3; }

    size_t emb_sz = (size_t)d * 256 * Edim;
    size_t W1_sz  = (size_t)d * Edim * H;
    size_t b1_sz  = H;
    size_t W2_sz  = (size_t)H * C;
    size_t b2_sz  = C;
    long expected = 20 + 4L * (emb_sz + W1_sz + b1_sz + W2_sz + b2_sz);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *emb = p; p += emb_sz;
    const float *W1  = p; p += W1_sz;
    const float *b1  = p; p += b1_sz;
    const float *W2  = p; p += W2_sz;
    const float *b2  = p;

    // Precompute T[j][byte][h] = emb[j][byte] dot W1_slice
    float *T = malloc((size_t)d * 256 * H * sizeof(float));
    for (int j = 0; j < d; j++) {
        const float *W1_j = W1 + (size_t)j * Edim * H;
        for (int byte = 0; byte < 256; byte++) {
            const float *e = emb + ((size_t)j * 256 + byte) * Edim;
            float *Tjb = T + ((size_t)j * 256 + byte) * H;
            for (int h = 0; h < H; h++) {
                float sum = 0.0f;
                for (int k = 0; k < Edim; k++) sum += e[k] * W1_j[k * H + h];
                Tjb[h] = sum;
            }
        }
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(H * sizeof(float));
    float *sc  = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int h = 0; h < H; h++) acc[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *Tjb = T + ((size_t)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) acc[h] += Tjb[h];
        }
        // ReLU + output
        for (int c = 0; c < C; c++) sc[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float hv = acc[h] > 0.0f ? acc[h] : 0.0f;
            const float *W2h = W2 + (size_t)h * C;
            for (int c = 0; c < C; c++) sc[c] += hv * W2h[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```