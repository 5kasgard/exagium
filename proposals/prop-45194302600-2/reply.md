```hypothesis
The label-hashngram family uses a linear model on hashed character n-grams (unigrams, bigrams, trigrams) from the 16-byte context. This captures local sequential patterns (like "th", "ing") that the position-independent one-hot linear model misses, while keeping the model size small (H=2048 buckets) and inference fast (only ~45 hash lookups and vector adds per item). It should beat label-linear's 0.3375 accuracy on dev. If it does not, it would show that n-gram interactions are not sufficient or the hashing collisions hurt too much.
```

```python
"""Hash n-gram linear model for next-character prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Features: all unigrams, bigrams, trigrams from the 16-byte context, hashed to H=2048 buckets.
Linear model with Adam (lr=3e-3, L2=1e-4, batch=64, epochs up to 100)."""
import json, math, struct, sys

import numpy as np

H = 2048
NGRAM_LENS = (1, 2, 3)
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x484E4731  # "HNG1"

def fnv1a_32(data: bytes) -> int:
    h = 0x811c9dc5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def compute_buckets(ctx: bytes) -> np.ndarray:
    """Return array of bucket indices for all n-grams in context."""
    buckets = []
    for n in NGRAM_LENS:
        for start in range(0, 17 - n):
            # encode: length (1 byte), start (1 byte), then the n bytes
            key = bytes([n, start]) + ctx[start:start + n]
            buckets.append(fnv1a_32(key) % H)
    return np.array(buckets, dtype=np.int32)

def main():
    card = json.load(open(sys.argv[1]))
    X_recs = records(sys.argv[2])
    y_recs = records(sys.argv[3])
    X = np.frombuffer(b"".join(X_recs), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in y_recs], dtype=np.int64)
    C = card["classes"]
    n = X.shape[0]

    # Precompute buckets for all training examples
    all_buckets = np.empty((n, 45), dtype=np.int32)  # 16+15+14 = 45
    for i in range(n):
        all_buckets[i] = compute_buckets(X[i].tobytes())

    rng = np.random.default_rng(SEED)
    # Initialize weights: W[H, C], b[C]
    W = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    # Adam state
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    beta1, beta2 = 0.9, 0.999
    t = 0

    steps_per_epoch = math.ceil(n / BATCH)
    max_steps = min(MAX_EPOCHS * steps_per_epoch, math.ceil(SAMPLES / BATCH))

    for step in range(max_steps):
        idx = rng.permutation(n)[:BATCH]
        buckets = all_buckets[idx]          # (BATCH, 45)
        yy = y[idx]

        # Forward: logits = b + sum_{k in buckets} W[k]
        logits = b + np.sum(W[buckets], axis=1)   # (BATCH, C)

        # Softmax gradient
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits, dtype=np.float32)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(BATCH), yy] -= 1.0
        probs /= BATCH

        # Gradients
        grad_W = np.zeros_like(W)
        np.add.at(grad_W, buckets.ravel(), np.repeat(probs, 45, axis=0))
        grad_W += L2 * W
        grad_b = probs.sum(axis=0)

        # Adam update
        t += 1
        m_W = beta1 * m_W + (1 - beta1) * grad_W
        v_W = beta2 * v_W + (1 - beta2) * grad_W * grad_W
        m_b = beta1 * m_b + (1 - beta1) * grad_b
        v_b = beta2 * v_b + (1 - beta2) * grad_b * grad_b
        m_W_hat = m_W / (1 - beta1 ** t)
        v_W_hat = v_W / (1 - beta2 ** t)
        m_b_hat = m_b / (1 - beta1 ** t)
        v_b_hat = v_b / (1 - beta2 ** t)
        W -= LR * m_W_hat / (np.sqrt(v_W_hat) + 1e-8)
        b -= LR * m_b_hat / (np.sqrt(v_b_hat) + 1e-8)

    # Write weights.bin
    out = struct.pack("<4i", MAGIC, H, C, len(NGRAM_LENS))
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Hash n-gram linear model inference: ./infer weights.bin in.rec N */
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

static uint32_t fnv1a_32(const unsigned char *data, int len) {
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
    uint8_t *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x484E4731) { fprintf(stderr, "bad magic\n"); return 3; }
    int H = hd[1], C = hd[2], ngram_cnt = hd[3];
    if (C < 1 || C > 256 || ngram_cnt != 3) { fprintf(stderr, "bad header\n"); return 3; }
    size_t expected = 16 + (size_t)H * C * 4 + (size_t)C * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    float *W = (float *)(wb + 16);
    float *b = W + (size_t)H * C;

    uint8_t *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * C);
    uint8_t best;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }

        for (int c = 0; c < C; c++) acc[c] = b[c];

        /* unigrams, bigrams, trigrams */
        for (int n = 1; n <= 3; n++) {
            for (int start = 0; start <= 16 - n; start++) {
                unsigned char key[18];
                key[0] = n; key[1] = start;
                memcpy(key + 2, xi + start, n);
                uint32_t bucket = fnv1a_32(key, n + 2) % H;
                const float *w = W + (size_t)bucket * C;
                for (int c = 0; c < C; c++) acc[c] += w[c];
            }
        }

        best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (uint8_t)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```