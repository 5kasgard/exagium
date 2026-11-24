```hypothesis
Family: label-hashngram (linear model on hashed unigram+bigram features).
Why: The task is next-character prediction from 16 bytes of context. The label-linear baseline uses only per-position unigrams (16×256 features). Adding bigram interactions (adjacent byte pairs) captures local character dependencies (e.g., "th", "he") that are crucial in English text. Feature hashing into a fixed budget (F=16384) keeps the model small and inference fast while allowing the linear model to learn these interactions. If this does not beat label-linear (0.3375), it suggests either that bigram information is not sufficiently predictive beyond unigrams for this context length, or that hashing collisions destroy too much signal.
```
```python
"""label-hashngram: linear model on hashed unigram+bigram features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x484E474C, int32 F, C, d; float32 W[F][C] (row-major), float32 b[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (fixed, not tuned on the tuning split)
F = 16384          # feature hash space size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

# FNV-1a 64-bit hash, modulo F
def fnv1a_mod(data_bytes, mod):
    h = 0xcbf29ce484222325
    for b in data_bytes:
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h % mod

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def extract_features(x, d, F):
    """Return list of 31 feature hashes for a 16-byte context."""
    feats = []
    # unigrams: type 0, position, byte
    for pos in range(d):
        feats.append(fnv1a_mod(bytes([0, pos, x[pos]]), F))
    # bigrams: type 1, position, byte1, byte2
    for pos in range(d - 1):
        feats.append(fnv1a_mod(bytes([1, pos, x[pos], x[pos + 1]]), F))
    return feats

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    train_in = records(sys.argv[2])
    train_key = records(sys.argv[3])
    n = len(train_in)
    # Build feature lists and labels
    X_feats = [extract_features(rec, d, F) for rec in train_in]
    y = np.array([r[0] for r in train_key], dtype=np.int64)

    rng = np.random.default_rng(SEED)
    # Initialize weights and bias
    W = (rng.uniform(-1, 1, (F, C)) * math.sqrt(6.0 / F)).astype(np.float32)
    b = np.zeros(C, np.float32)
    # Adam state
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            bs = len(idx)
            # Forward + gradient accumulation
            gW = np.zeros_like(W)
            gb = np.zeros_like(b)
            for i in idx:
                feats = X_feats[i]
                # scores = b + sum(W[f])
                scores = b.copy()
                for f in feats:
                    scores += W[f]
                # softmax
                scores -= scores.max()
                exp_scores = np.exp(scores)
                probs = exp_scores / exp_scores.sum()
                # gradient: probs - onehot
                probs[y[i]] -= 1.0
                # accumulate
                for f in feats:
                    gW[f] += probs
                gb += probs
            # Average over batch
            gW /= bs
            gb /= bs
            # L2 regularization on W
            gW += L2 * W
            # Adam update
            t += 1
            mW = 0.9 * mW + 0.1 * gW
            vW = 0.999 * vW + 0.001 * gW * gW
            mb = 0.9 * mb + 0.1 * gb
            vb = 0.999 * vb + 0.001 * gb * gb
            mW_hat = mW / (1 - 0.9 ** t)
            vW_hat = vW / (1 - 0.999 ** t)
            mb_hat = mb / (1 - 0.9 ** t)
            vb_hat = vb / (1 - 0.999 ** t)
            W -= LR * mW_hat / (np.sqrt(vW_hat) + 1e-8)
            b -= LR * mb_hat / (np.sqrt(vb_hat) + 1e-8)

    # Write weights.bin
    magic = 0x484E474C  # "HNGL"
    out = struct.pack("<4i", magic, F, C, d)
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-hashngram infer: ./infer weights.bin in.rec N
 * Input records: 16 bytes each. Output: one byte class index per record. */
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

/* FNV-1a 64-bit hash */
static uint64_t fnv1a(const unsigned char *data, size_t len) {
    uint64_t h = 0xcbf29ce484222325;
    for (size_t i = 0; i < len; i++) {
        h ^= data[i];
        h *= 0x100000001b3;
    }
    return h;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x484E474C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int F = hd[1], C = hd[2], d = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 16 + 4L * (long)C * (F + 1);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }
    const float *W = (const float *)(hd + 4);
    const float *b = W + (long)F * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    unsigned char feat_bytes[4];  // max feature length: type(1) + pos(1) + byte1(1) + byte2(1)

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // init scores with bias
        for (int c = 0; c < C; c++) scores[c] = b[c];
        // unigrams
        for (int p = 0; p < d; p++) {
            feat_bytes[0] = 0; feat_bytes[1] = p; feat_bytes[2] = xi[p];
            uint64_t h = fnv1a(feat_bytes, 3);
            int f = (int)(h % (uint64_t)F);
            const float *wf = W + (long)f * C;
            for (int c = 0; c < C; c++) scores[c] += wf[c];
        }
        // bigrams
        for (int p = 0; p < d - 1; p++) {
            feat_bytes[0] = 1; feat_bytes[1] = p; feat_bytes[2] = xi[p]; feat_bytes[3] = xi[p + 1];
            uint64_t h = fnv1a(feat_bytes, 4);
            int f = (int)(h % (uint64_t)F);
            const float *wf = W + (long)f * C;
            for (int c = 0; c < C; c++) scores[c] += wf[c];
        }
        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```