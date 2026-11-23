```hypothesis
Family: label-embmlp (embedding MLP with one hidden ReLU layer).
Why: Character-level language modeling benefits from distributed representations; embeddings capture semantic similarity between bytes (e.g., letters vs punctuation) and drastically reduce parameters vs one-hot linear (16*256*E + E*H + H*C vs 16*256*C). A small hidden layer (H=64) can model pairwise interactions between context positions. With 24k training examples, E=16, H=64 (~70k params) should generalize better than label-linear (266k params) while being faster at inference (dense matvec vs sparse lookup sum).
Failure: Dev accuracy ≤ 0.3375 (label-linear) or instructions/item > 15000 (2× label-linear).
```

```python
"""label-embmlp: embedding MLP for fixed-size byte sequence classification.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, numpy only.
Architecture: per-position byte embeddings (16*256*E) -> sum -> ReLU(E->H) -> linear(H->C).
Exported weights format (little-endian):
  int32 magic=0x454D424D ('EMBM'), d=16, E, H, C
  float32 embeddings[d][256][E] (row-major: position, byte, emb_dim)
  float32 W1[E][H], b1[H]
  float32 W2[H][C], b2[C]
If H=0 (linear on embeddings): W1/b1 omitted, W2[E][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
E = 16          # embedding dimension
H = 64          # hidden units (0 for linear on embeddings)
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 20


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
    assert d == 16

    rng = np.random.default_rng(SEED)

    # Initialize embeddings: (d, 256, E)
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)

    if H:
        W1 = (rng.normal(0, math.sqrt(2.0 / E), (E, H))).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W1, b1, W2, b2]
    else:
        W2 = (rng.normal(0, math.sqrt(2.0 / E), (E, C))).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W2, b2]

    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    n = X.shape[0]
    t = 0

    def forward(Xb):
        # Xb: (B, d) uint8
        B = Xb.shape[0]
        # embedding lookup and sum: (B, E)
        x = np.zeros((B, E), np.float32)
        for j in range(d):
            x += emb[j, Xb[:, j]]
        if H:
            h = x @ W1 + b1
            h = np.maximum(h, 0)
            logits = h @ W2 + b2
            return x, h, logits
        else:
            logits = x @ W2 + b2
            return x, None, logits

    def backward(Xb, yb, x, h, logits):
        B = Xb.shape[0]
        # softmax gradient
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits, dtype=np.float32)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(B), yb] -= 1.0
        probs /= B

        grads = [np.zeros_like(p) for p in params]
        if H:
            # grad W2, b2
            grads[3] = h.T @ probs + L2 * W2
            grads[4] = probs.sum(axis=0)
            # grad hidden
            gh = probs @ W2.T
            gh *= (h > 0)
            # grad W1, b1
            grads[1] = x.T @ gh + L2 * W1
            grads[2] = gh.sum(axis=0)
            # grad embeddings
            gx = gh @ W1.T
            for j in range(d):
                np.add.at(grads[0][j], Xb[:, j], gx)
        else:
            grads[1] = x.T @ probs + L2 * W2
            grads[2] = probs.sum(axis=0)
            gx = probs @ W2.T
            for j in range(d):
                np.add.at(grads[0][j], Xb[:, j], gx)
        return grads

    for _ in range(MAX_EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            x, h, logits = forward(Xb)
            grads = backward(Xb, yb, x, h, logits)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    out = struct.pack("<5i", 0x454D424D, d, E, H, C)
    out += emb.astype("<f4").tobytes()
    if H:
        out += params[1].astype("<f4").tobytes()  # W1
        out += params[2].astype("<f4").tobytes()  # b1
        out += params[3].astype("<f4").tobytes()  # W2
        out += params[4].astype("<f4").tobytes()  # b2
    else:
        out += params[1].astype("<f4").tobytes()  # W2
        out += params[2].astype("<f4").tobytes()  # b2
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 raw bytes. Output record: 1 byte class index (argmax).
 * Weights format (little-endian):
 *   int32 magic=0x454D424D, d, E, H, C
 *   float32 emb[d][256][E]
 *   if H>0: float32 W1[E][H], b1[H], W2[H][C], b2[C]
 *   else:   float32 W2[E][C], b2[C] */
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
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C < 1 || C > 256 || E < 1 || H < 0) { fprintf(stderr, "bad header dims\n"); return 3; }

    size_t emb_sz = (size_t)d * 256 * E;
    size_t w1_sz = H ? (size_t)E * H : 0;
    size_t b1_sz = H ? H : 0;
    size_t w2_sz = H ? (size_t)H * C : (size_t)E * C;
    size_t b2_sz = C;
    long expected = 20 + 4L * (emb_sz + w1_sz + b1_sz + w2_sz + b2_sz);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + emb_sz;
    const float *b1 = W1 + w1_sz;
    const float *W2 = b1 + b1_sz;
    const float *b2 = W2 + w2_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_emb = malloc(sizeof(float) * E);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // embedding sum
        for (int e = 0; e < E; e++) x_emb[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((size_t)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) x_emb[e] += eptr[e];
        }

        if (H) {
            // hidden = ReLU(x_emb @ W1 + b1)
            for (int h = 0; h < H; h++) {
                float sum = b1[h];
                for (int e = 0; e < E; e++) sum += x_emb[e] * W1[(size_t)e * H + h];
                hidden[h] = sum > 0.0f ? sum : 0.0f;
            }
            // scores = hidden @ W2 + b2
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int h = 0; h < H; h++) sum += hidden[h] * W2[(size_t)h * C + c];
                scores[c] = sum;
            }
        } else {
            // scores = x_emb @ W2 + b2
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int e = 0; e < E; e++) sum += x_emb[e] * W2[(size_t)e * C + c];
                scores[c] = sum;
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```