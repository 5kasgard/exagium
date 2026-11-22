```hypothesis
Family: label-embedmlp (per-position embeddings + one ReLU hidden layer).
Why: The 16-byte context has strong positional patterns; per-position embeddings capture byte-position interactions more efficiently than one-hot linear. A small hidden layer (H=128) adds non-linearity with modest cost.
Failure: If dev accuracy ≤ 0.3375 (label-linear) or CPU cost per item exceeds label-linear by >20%.
```

```python
"""label-embedmlp: per-position embeddings + one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Exports: magic(0x454D4244), d, D, H, C; float32 embed[d][256][D], W1[(d*D)][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

D = 32          # embedding dimension per position
H = 128         # hidden units
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

    # per-position embeddings: d x 256 x D
    embed = (rng.uniform(-1, 1, (d, 256, D)) * math.sqrt(6.0 / 256)).astype(np.float32)

    # hidden layer: (d*D) -> H
    W1 = (rng.uniform(-1, 1, (d * D, H)) * math.sqrt(6.0 / (d * D))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # output layer: H -> C
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [embed, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi, yy = X[idx], y[idx]
            # forward
            # embed lookup: (batch, d, D)
            e = embed[np.arange(d)[None, :], xi[:, :, None], np.arange(D)[None, None, :]]  # (B, d, D)
            x_flat = e.reshape(len(xi), -1)  # (B, d*D)
            h_pre = x_flat @ W1 + b1
            h = np.maximum(h_pre, 0)
            logits = h @ W2 + b2

            # softmax + cross-entropy gradient
            logits = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)  # (B, C)

            # backprop
            gW2 = h.T @ g + L2 * W2
            gb2 = g.sum(0)
            gh = g @ W2.T
            gh[h_pre <= 0] = 0.0
            gW1 = x_flat.T @ gh + L2 * W1
            gb1 = gh.sum(0)
            # embed gradients: (d, 256, D)
            g_flat = gh @ W1.T  # (B, d*D)
            g_embed = np.zeros_like(embed)
            for j in range(d):
                # g_flat[:, j*D:(j+1)*D] is gradient for position j
                gj = g_flat[:, j*D:(j+1)*D]  # (B, D)
                # scatter-add into g_embed[j, xi[:, j], :]
                np.add.at(g_embed[j], xi[:, j], gj)
            g_embed += L2 * embed

            grads = [g_embed, gW1, gb1, gW2, gb2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # export
    magic = 0x454D4244  # 'EMBD'
    out = struct.pack("<5i", magic, d, D, H, C)
    out += embed.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embedmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte class index.
 * Weights format: magic(0x454D4244), d, D, H, C (int32); then float32 arrays:
 *   embed[d][256][D], W1[(d*D)][H], b1[H], W2[H][C], b2[C] */
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * D + (long)d * D * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *embed = (const float *)(hd + 5);
    const float *W1 = embed + (long)d * 256 * D;
    const float *b1 = W1 + (long)d * D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * H);
    float *s = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // hidden layer
        for (int u = 0; u < H; u++) h[u] = b1[u];
        for (int j = 0; j < d; j++) {
            const float *e = embed + ((long)j * 256 + xi[j]) * D;
            for (int u = 0; u < H; u++) {
                float sum = 0.0f;
                for (int v = 0; v < D; v++) sum += e[v] * W1[((long)j * D + v) * H + u];
                h[u] += sum;
            }
        }
        // ReLU
        for (int u = 0; u < H; u++) if (h[u] < 0.0f) h[u] = 0.0f;

        // output
        for (int c = 0; c < C; c++) s[c] = b2[c];
        for (int u = 0; u < H; u++) {
            float hu = h[u];
            for (int c = 0; c < C; c++) s[c] += hu * W2[(long)u * C + c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (s[c] > s[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```