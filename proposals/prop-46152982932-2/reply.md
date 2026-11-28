```hypothesis
Family: label-embmlp (positional embeddings + one ReLU layer, with first-layer folding for fast inference).
Why: Character prediction benefits from non-linear interactions between context positions. By learning small per-position embeddings (E=8) and a hidden layer (H=16), then folding the embedding×input-weight product into a single lookup table per position (16×256×H), inference does only 16×H adds + H×C MACs (≈1.3k ops), similar to label-linear's 16×C adds but with a ReLU non-linearity. This should exceed label-linear's 0.3375 accuracy at comparable CPU cost.
Failure: If dev accuracy ≤ 0.3375 or instructions/item ≥ 1.5× label-linear, the non-linearity doesn't help enough for this task.
```

```python
"""label-embmlp: positional embeddings + one ReLU layer, folded for fast inference.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, stdlib+numpy only.
Exports: magic 0x454D4231, d, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

E, H = 8, 16
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
    d, C = X.shape[1], card["classes"]
    rng = np.random.default_rng(SEED)

    # params: emb[d,256,E], W1[d*E,H], b1[H], W2[H,C], b2[C]
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return (probs / len(yy)).astype(np.float32)

    t = 0
    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            # forward
            # emb_lookup: (B, d, E)
            emb_seq = emb[:, xb.T].transpose(1, 0, 2)  # (B, d, E)
            flat = emb_seq.reshape(len(yb), d * E)     # (B, d*E)
            hidden_pre = flat @ W1 + b1                 # (B, H)
            hidden = np.maximum(hidden_pre, 0)
            logits = hidden @ W2 + b2                   # (B, C)
            # backward
            g = softmax_grad(logits, yb)                # (B, C)
            gW2 = hidden.T @ g + L2 * W2
            gb2 = g.sum(0)
            gh = g @ W2.T * (hidden_pre > 0)            # (B, H)
            gW1 = flat.T @ gh + L2 * W1
            gb1 = gh.sum(0)
            # emb grad: gh (B,H) @ W1.T (H, d*E) -> (B, d*E) -> reshape (B,d,E) -> scatter to emb
            gflat = gh @ W1.T                           # (B, d*E)
            gemb_seq = gflat.reshape(len(yb), d, E)     # (B, d, E)
            # accumulate per position and byte value
            gemb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(gemb[j], xb[:, j], gemb_seq[:, j])
            gemb += L2 * emb
            grads = [gemb, gW1, gb1, gW2, gb2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # fold first layer: T[j, byte, h] = emb[j, byte] @ W1[j*E:(j+1)*E, h]
    T = np.zeros((d, 256, H), np.float32)
    for j in range(d):
        T[j] = emb[j] @ W1[j * E:(j + 1) * E]

    # pack: magic, d, H, C; T (d*256*H), b1 (H), W2 (H*C), b2 (C)
    out = struct.pack("<4i", 0x454D4231, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index.
 * Weights format: magic 0x454D4231, int32 d, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 16 || hd[0] != 0x454D4231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * H);
    float *s = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int u = 0; u < H; u++) h[u] = b1[u];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H;
            for (int u = 0; u < H; u++) h[u] += t[u];
        }
        for (int u = 0; u < H; u++) if (h[u] < 0.0f) h[u] = 0.0f;
        for (int c = 0; c < C; c++) s[c] = b2[c];
        for (int u = 0; u < H; u++) {
            float hu = h[u];
            const float *w = W2 + (long)u * C;
            for (int c = 0; c < C; c++) s[c] += hu * w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (s[c] > s[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```