```hypothesis
Family: label-hashngram (linear or 1-hidden-layer MLP on hashed unigram+bigram features).
Why: tinyshakespeare next-char depends on local n-grams; label-linear uses only unigrams (one-hot per position).
Hashing bigrams (15 overlapping pairs) into a shared embedding space captures pairwise interactions
with far fewer parameters than a full one-hot bigram model. A hidden layer adds non-linearity.
If dev accuracy ≤ 0.3375 (label-linear) or inference cost > 2× label-linear, the idea fails.
```

```python
"""label-hashngram: hashed unigram+bigram features → linear/MLP.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Feature space: D=2048. 16 unigram hashes (pos*256 + byte) + 15 bigram hashes
(pos*65536 + b1*256 + b2) all modulo D. Hidden size H=64 (0 = linear).
Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Weights: magic 0x48415348, d, D, H, C; float32 W1[D][W], b1[W]; if H: W2[H][C], b2[C] (W=H or C)."""
import json, math, struct, sys

import numpy as np

D = 2048
H = 64
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def featurize(X, d, D):
    """X: (n, d) uint8. Returns (n, 31) int32 hash indices."""
    n = X.shape[0]
    idx = np.empty((n, 31), dtype=np.int32)
    # unigrams
    for p in range(d):
        idx[:, p] = (p * 256 + X[:, p]) % D
    # bigrams
    for p in range(d - 1):
        idx[:, 16 + p] = (16 * 256 + p * 65536 + X[:, p] * 256 + X[:, p + 1]) % D
    return idx


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # hash features
    idx = featurize(X, d, D)  # (n, 31)

    rng = np.random.default_rng(SEED)
    W = H if H else C
    W1 = (rng.uniform(-1, 1, (D, W)) * math.sqrt(6.0 / D)).astype(np.float32)
    b1 = np.zeros(W, np.float32)
    if H:
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [W1, b1, W2, b2]
    else:
        params = [W1, b1]

    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            batch_idx = order[s:s + BATCH]
            idb = idx[batch_idx]          # (b, 31)
            yy = y[batch_idx]
            b = len(yy)
            # forward
            acc = np.zeros((b, W), np.float32)
            for f in range(31):
                acc += W1[idb[:, f]]
            acc += b1
            if H:
                h = np.maximum(acc, 0)
                logits = h @ W2 + b2
            else:
                logits = acc
            # softmax gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(b), yy] -= 1.0
            grad_out = (probs / b).astype(np.float32)
            # backward
            if H:
                gW2 = h.T @ grad_out + L2 * W2
                gb2 = grad_out.sum(0)
                gh = grad_out @ W2.T * (acc > 0)
                gW1 = np.zeros_like(W1)
                for f in range(31):
                    np.add.at(gW1, idb[:, f], gh)
                gW1 += L2 * W1
                gb1 = gh.sum(0)
                grads = [gW1, gb1, gW2, gb2]
            else:
                gW1 = np.zeros_like(W1)
                for f in range(31):
                    np.add.at(gW1, idb[:, f], grad_out)
                gW1 += L2 * W1
                gb1 = grad_out.sum(0)
                grads = [gW1, gb1]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights
    out = struct.pack("<4i", 0x48415348, d, D, H, C)
    out += W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    if H:
        out += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-hashngram infer: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte class index.
 * Weights format: magic 0x48415348, int32 d, D, H, C; float32 W1[D][W], b1[W];
 * if H>0: float32 W2[H][C], b2[C] (W = H if H>0 else C). */
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
    if (nw < 20 || hd[0] != 0x48415348) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    int W = H ? H : C;
    long expected = 20L + 4L * ((long)D * W + W + (H ? (long)H * C + C : 0));
    if (C < 1 || C > 256 || nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W1 = (const float *)(hd + 5);
    const float *b1 = W1 + (long)D * W;
    const float *W2 = b1 + W;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // compute 31 hash indices
        int idx[31];
        for (int p = 0; p < d; p++) idx[p] = (p * 256 + xi[p]) % D;
        for (int p = 0; p < d - 1; p++) idx[16 + p] = (16 * 256 + p * 65536 + xi[p] * 256 + xi[p + 1]) % D;

        // accumulate W1 rows
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int f = 0; f < 31; f++) {
            const float *row = W1 + (long)idx[f] * W;
            for (int w = 0; w < W; w++) acc[w] += row[w];
        }

        const float *logits;
        if (H) {
            for (int h = 0; h < H; h++) hidden[h] = acc[h] > 0.0f ? acc[h] : 0.0f;
            for (int c = 0; c < C; c++) scores[c] = b2[c];
            for (int h = 0; h < H; h++) {
                float hv = hidden[h];
                const float *w2row = W2 + (long)h * C;
                for (int c = 0; c < C; c++) scores[c] += hv * w2row[c];
            }
            logits = scores;
        } else {
            logits = acc;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```