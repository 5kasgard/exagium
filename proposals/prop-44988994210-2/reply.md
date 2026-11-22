```hypothesis
Family: label-hashngram (hashed n-gram linear model).
Why: The task is next-character prediction with 16-byte context. Label-linear uses only position-specific unigrams (16×256 features). Adding bigrams and trigrams captures local character interactions crucial for language. Feature hashing into a fixed bucket array (B=16384) keeps the model small and inference fast (≈45 hashes + 45×65 additions per item), while the linear model on hashed features can be trained with the same Adam routine. This should exceed label-linear's 33.75% accuracy at modestly higher cost (target <15k Ir/item).
Failure signal: Dev accuracy ≤ 0.34 or Ir/item > 20000.
```

```python
"""Hashed n-gram linear model for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Features: unigrams (pos, byte), bigrams (pos, byte, byte), trigrams (pos, byte, byte, byte)
Hashed into B=16384 buckets with FNV-1a. Linear model (logistic regression) trained with Adam.
Weights format: magic 0x484E474C, int32 d,H,C,B; float32 W[B][C], b[C]."""
import json, math, struct, sys

import numpy as np

B = 16384          # hash buckets
SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
N_FEAT = 45        # 16 unigrams + 15 bigrams + 14 trigrams


def fnv1a_32(data: bytes) -> int:
    h = 0x811c9dc5
    for byte in data:
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def extract_features(X: np.ndarray) -> np.ndarray:
    """X: (n, 16) uint8 -> (n, 45) int32 bucket indices."""
    n = X.shape[0]
    feats = np.empty((n, N_FEAT), dtype=np.uint32)
    for i in range(n):
        x = X[i]
        idx = 0
        # unigrams
        for pos in range(16):
            key = bytes([0, pos, x[pos]])
            feats[i, idx] = fnv1a_32(key) % B
            idx += 1
        # bigrams
        for pos in range(15):
            key = bytes([1, pos, x[pos], x[pos + 1]])
            feats[i, idx] = fnv1a_32(key) % B
            idx += 1
        # trigrams
        for pos in range(14):
            key = bytes([2, pos, x[pos], x[pos + 1], x[pos + 2]])
            feats[i, idx] = fnv1a_32(key) % B
            idx += 1
    return feats


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n = X.shape[0]

    # Feature extraction
    feats = extract_features(X)          # (n, 45) uint32

    # Model parameters
    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (B, C)) * math.sqrt(6.0 / B)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            batch_feats = feats[idx]          # (batch, 45)
            yy = y[idx]
            batch_size = len(idx)

            # Forward: logits = sum_k W[feats[i,k]] + b
            logits = np.zeros((batch_size, C), dtype=np.float32)
            for i in range(batch_size):
                logits[i] = W[batch_feats[i]].sum(axis=0)
            logits += b

            # Softmax loss gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(batch_size), yy] -= 1.0
            grad_logits = (probs / batch_size).astype(np.float32)   # (batch, C)

            # Gradients for W (sparse) and b (dense)
            grad_W = np.zeros_like(W)
            # Accumulate per bucket: for each example, each feature gets the same grad_logits[ex]
            flat_feats = batch_feats.ravel()
            ex_idx = np.repeat(np.arange(batch_size), N_FEAT)
            np.add.at(grad_W, flat_feats, grad_logits[ex_idx])
            grad_W += L2 * W
            grad_b = grad_logits.sum(axis=0) + L2 * b
            grads = [grad_W, grad_b]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Save weights
    magic = 0x484E474C  # 'HNGL'
    d = card["input"]["bytes"]
    H = 0
    out = struct.pack("<5i", magic, d, H, C, B)
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Hashed n-gram linear model inference: ./infer weights.bin in.rec N
 * Each input record: 16 bytes. Output: 1 byte class index.
 * Weights format: magic 0x484E474C, int32 d,H,C,B; float32 W[B][C], b[C]. */
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

/* FNV-1a 32-bit hash */
static uint32_t fnv1a(const unsigned char *data, int len) {
    uint32_t h = 0x811c9dc5;
    for (int i = 0; i < len; i++) {
        h ^= data[i];
        h *= 0x01000193;
    }
    return h;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x484E474C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], B = hd[4];
    if (d != 16 || H != 0 || C < 1 || C > 256) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 20L + 4L * ((long)B * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W = (const float *)(hd + 5);
    const float *b = W + (long)B * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* Clear scores with bias */
        memcpy(scores, b, C * sizeof(float));

        /* Unigrams: 16 */
        unsigned char key[5];
        for (int p = 0; p < 16; p++) {
            key[0] = 0; key[1] = p; key[2] = xi[p];
            uint32_t bucket = fnv1a(key, 3) % (uint32_t)B;
            const float *w = W + (long)bucket * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        /* Bigrams: 15 */
        for (int p = 0; p < 15; p++) {
            key[0] = 1; key[1] = p; key[2] = xi[p]; key[3] = xi[p+1];
            uint32_t bucket = fnv1a(key, 4) % (uint32_t)B;
            const float *w = W + (long)bucket * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        /* Trigrams: 14 */
        for (int p = 0; p < 14; p++) {
            key[0] = 2; key[1] = p; key[2] = xi[p]; key[3] = xi[p+1]; key[4] = xi[p+2];
            uint32_t bucket = fnv1a(key, 5) % (uint32_t)B;
            const float *w = W + (long)bucket * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }

        /* Argmax (first max wins) */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```