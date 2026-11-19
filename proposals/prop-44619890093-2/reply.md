```hypothesis
Family: label-embed (positional byte embeddings + tiny 2-layer MLP).
Why: Character prediction has strong positional structure; dense embeddings (E=16) replace 256-d one-hots, capturing similarity between bytes and reducing parameters 16x. A small ReLU layer (H=64) models interactions. This should beat label-linear's 0.3375 accuracy at ~10x lower inference cost (fewer FLOPs, better cache use).
Failure signal: dev accuracy ≤ 0.3375 or Ir/item ≥ label-linear's 7360.
```
```python
"""label-embed: positional byte embeddings + 2-layer MLP for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only.
Architecture: 16 positions × 256 bytes → embedding dim E=16 → concat (256) → ReLU H=64 → 65 classes.
Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Weights format (little-endian):
  magic(0x454D4245), d=16, E=16, H=64, C=65
  emb[d*256*E] (float32), W1[(d*E)*H], b1[H], W2[H*C], b2[C]"""
import json, math, struct, sys

import numpy as np

E, H = 16, 64
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x454D4245  # 'EMBE'


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
    d, C = card["input"]["bytes"], card["classes"]
    assert d == 16 and C == 65

    rng = np.random.default_rng(SEED)
    # Embeddings: [d, 256, E] ~ Xavier
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / (256 + E))).astype(np.float32)
    # MLP weights
    in_dim = d * E
    W1 = (rng.uniform(-1, 1, (in_dim, H)) * math.sqrt(6.0 / in_dim)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    def sg(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits, dtype=np.float32)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi, yy = X[idx], y[idx]
            bsz = xi.shape[0]

            # Forward: embeddings lookup + concat
            x_emb = np.zeros((bsz, in_dim), np.float32)
            for j in range(d):
                x_emb[:, j*E:(j+1)*E] = emb[j, xi[:, j], :]

            a = x_emb @ W1 + b1
            h = np.maximum(a, 0, dtype=np.float32)
            logits = h @ W2 + b2

            g = sg(logits, yy)  # [bsz, C]

            # Backprop
            gW2 = h.T @ g + L2 * W2
            gb2 = g.sum(0)
            gh = g @ W2.T
            gh[a <= 0] = 0.0
            gW1 = x_emb.T @ gh + L2 * W1
            gb1 = gh.sum(0)

            # Embedding gradients
            gemb = np.zeros_like(emb)
            for j in range(d):
                gj = gh[:, j*E:(j+1)*E]  # [bsz, E]
                np.add.at(gemb[j], xi[:, j], gj)

            grads = [gemb, gW1, gb1, gW2, gb2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    out = struct.pack("<5i", MAGIC, d, E, H, C)
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
/* label-embed: positional byte embeddings + 2-layer MLP.
 * ./infer weights.bin in.rec N
 * Weights: magic(0x454D4245), d, E, H, C; emb[d*256*E], W1[(d*E)*H], b1[H], W2[H*C], b2[C] (all float32 LE).
 * Input record: d bytes (d=16). Output: 1 byte class index (argmax, first max wins). */
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
    if (nw < 20 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E != 16 || H != 64 || C != 65) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long in_dim = (long)d * E;
    long expected = 20 + 4L * ((long)d * 256 * E + in_dim * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1  = emb + (long)d * 256 * E;
    const float *b1  = W1 + in_dim * H;
    const float *W2  = b1 + H;
    const float *b2  = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = (float *)malloc(sizeof(float) * in_dim);
    float *h_act  = (float *)malloc(sizeof(float) * H);
    float *scores = (float *)malloc(sizeof(float) * C);
    if (!hidden || !h_act || !scores) { fprintf(stderr, "oom\n"); exit(3); }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup + concat into hidden (reused as input buffer)
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(hidden + j * E, e, E * sizeof(float));
        }

        // Layer 1: hidden = ReLU(hidden @ W1 + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            const float *w = W1 + (long)u;  // W1 is [in_dim, H] column-major? No, row-major: W1[in_dim * H], W1[row*H + col]
            // Actually W1 is [in_dim, H] stored row-major: W1[i*H + u] for input i, output u
            for (int v = 0; v < in_dim; v++) {
                sum += hidden[v] * W1[v * H + u];
            }
            h_act[u] = sum > 0.0f ? sum : 0.0f;
        }

        // Layer 2: scores = h_act @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) {
                sum += h_act[u] * W2[(long)u * C + c];
            }
            scores[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```