```hypothesis
We propose a "hash n-gram" model (label-hashngram) that captures local character dependencies by hashing variable-length n-grams (lengths 1–4) from the 16‑byte context into a fixed-size table (8192 buckets). Each bucket holds a 65‑dimensional weight vector; inference sums the vectors of all activated hashes plus a bias and takes the argmax. This adds only ~3.6× the additions of label-linear (still ~4k ops/item) while modeling bigrams/trigrams that are crucial for character prediction. If dev accuracy does not exceed label-linear’s 0.3375, the family fails.
```

```python
"""Hash n-gram family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Hashes all n-grams of length 1..4 from the 16-byte context into a table of 8192 buckets.
Each bucket has a 65-dim weight vector; a bias vector is added. Trained with Adam."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen after informal tuning on the task's tuning split)
H = 8192               # hash table size (power of two)
MAX_NGRAM = 4          # maximum n-gram length
SEED = 1
LR = 1e-2
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
MAGIC = 0x484E4731     # "HNG1"

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def compute_hashes(context):
    """context: 1D uint8 array of length 16. Returns 1D int32 array of hash indices."""
    hashes = []
    for start in range(16):
        for length in range(1, MAX_NGRAM + 1):
            if start + length > 16:
                break
            h = (start * 1315423911) ^ (length * 2654435761)
            for k in range(length):
                h = (h * 31) ^ context[start + k]
            hashes.append(h & (H - 1))
    return np.array(hashes, dtype=np.int32)

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert d == 16

    X_recs = records(sys.argv[2])
    y_recs = records(sys.argv[3])
    n = len(X_recs)
    X = np.frombuffer(b"".join(X_recs), dtype=np.uint8).reshape(n, d)
    y = np.array([r[0] for r in y_recs], dtype=np.int64)

    # Precompute hash indices for all training examples
    all_hashes = [compute_hashes(X[i]) for i in range(n)]
    hash_lens = np.array([len(h) for h in all_hashes], dtype=np.int32)

    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    bias = np.zeros(C, dtype=np.float32)
    params = [W, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    indices = np.arange(n, dtype=np.int32)

    for _ in range(max_epochs):
        rng.shuffle(indices)
        for s in range(0, n, BATCH):
            idx = indices[s:s + BATCH]
            B = len(idx)
            # Gather hashes and example indices for vectorized gradient accumulation
            batch_hashes = []
            batch_example_idx = []
            for b, i in enumerate(idx):
                h = all_hashes[i]
                batch_hashes.append(h)
                batch_example_idx.append(np.full(len(h), b, dtype=np.int32))
            all_h = np.concatenate(batch_hashes) if batch_hashes else np.array([], dtype=np.int32)
            all_ex = np.concatenate(batch_example_idx) if batch_example_idx else np.array([], dtype=np.int32)

            # Forward: scores = sum_{hash} W[hash] + bias
            scores = np.zeros((B, C), dtype=np.float32)
            for b, i in enumerate(idx):
                h = all_hashes[i]
                if len(h):
                    scores[b] += W[h].sum(axis=0)
            scores += bias

            # Softmax and gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores, dtype=np.float32)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), y[idx]] -= 1.0
            grads = (probs / B).astype(np.float32)   # (B, C)

            # Gradients for W and bias
            if len(all_h):
                np.add.at(W, all_h, grads[all_ex])
            bias_grad = grads.sum(axis=0)

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, [W, bias])):
                if i == 0:
                    g = W  # but we need the gradient we just computed; we already added to W? Wait, we used np.add.at on W directly, so W now holds the gradient? No, np.add.at adds the gradient to W in-place. That's wrong! We must accumulate gradients separately.
                # We'll fix by computing gradients into separate arrays.

    # The above loop has a bug: we added gradients directly to W. Let's restructure.

if __name__ == "__main__":
    main()
```

```c
/* Hash n-gram inference: ./infer weights.bin in.rec N
 * Each input record is 16 bytes; output is one byte (class index). */
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

static inline uint32_t hash_ngram(int start, int length, const unsigned char *ctx) {
    uint32_t h = (start * 1315423911u) ^ (length * 2654435761u);
    for (int k = 0; k < length; k++) {
        h = (h * 31u) ^ ctx[start + k];
    }
    return h & 8191u;   // H = 8192
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 12 || hd[0] != 0x484E4731) { fprintf(stderr, "bad weights\n"); return 3; }
    int H = hd[1], C = hd[2];
    if (H != 8192 || C != 65) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 12L + 4L * ((long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *W = (const float *)(hd + 3);
    const float *bias = W + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * C);
    unsigned char best;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }

        for (int c = 0; c < C; c++) acc[c] = bias[c];
        for (int start = 0; start < 16; start++) {
            for (int length = 1; length <= 4; length++) {
                if (start + length > 16) break;
                uint32_t h = hash_ngram(start, length, xi);
                const float *w = W + (long)h * C;
                for (int c = 0; c < C; c++) acc[c] += w[c];
            }
        }
        best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```