```hypothesis
Propose label-mlp-tanh: a one-hidden-layer MLP with 32 tanh units, inputs scaled to [0,1] (divide by 16). The 8x8 digit images have smooth spatial structure that tanh can model better than linear; min-max scaling matches the 0..16 pixel range. If dev accuracy ≤ 0.9613 (label-linear) or cost exceeds label-linear, the family does not improve on the baseline.
```

```python
"""label-mlp-tanh: one-hidden-layer MLP with tanh, 32 units, min-max scaled inputs.
python3 -s -P train.py card.json train.in train.key weights.bin
Export format: magic 0x54424C31, d, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

H = 32
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
    rng = np.random.default_rng(SEED)

    # min-max scaling to [0,1] for ordinal (pixel counts 0..16)
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        if t == "ordinal":
            # scale by 16.0, but keep table for all 256 byte values
            scale = 1.0 / 16.0
            enc.append((np.arange(256, dtype=np.float32) * scale)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)

    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, k = Z.shape

    # Xavier init for tanh
    W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / (k + H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / (H + C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]

            # forward
            a = z @ W1 + b1
            h = np.tanh(a)
            logits = h @ W2 + b2

            # softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)  # dL/dlogits

            # backward
            grad_W2 = h.T @ g + L2 * W2
            grad_b2 = g.sum(0)
            gh = g @ W2.T * (1.0 - h * h)  # tanh derivative
            grad_W1 = z.T @ gh + L2 * W1
            grad_b1 = gh.sum(0)
            grads = [grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1.0 - 0.9 ** t)
                v_hat = v / (1.0 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Build lookup tables T[j][v] = contribution of byte v at feature j to hidden layer (length H)
    off, T_list = 0, []
    for E in enc:
        # E shape (256, feat_dim), W1 slice (feat_dim, H) -> (256, H)
        T_list.append((E @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    T = np.stack(T_list)  # (d, 256, H)

    out = struct.pack("<4i", 0x54424C31, X.shape[1], H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-mlp-tanh inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte class (argmax). */
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
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * H + H + (long)H * C + C)) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * H);
    float *s = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int h = 0; h < H; h++) acc[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) acc[h] += t[h];
        }
        for (int h = 0; h < H; h++) acc[h] = tanhf(acc[h]);

        for (int c = 0; c < C; c++) s[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float ah = acc[h];
            for (int c = 0; c < C; c++) s[c] += ah * W2[(long)h * C + c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (s[c] > s[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```