```hypothesis
Family: label-ngram — linear model over hashed positional n-grams (unigrams, bigrams, trigrams).
Why: Character-level language modeling benefits from local sequence patterns (e.g., "th", "ing", "tion"). The current label-linear uses only position-specific unigrams (16×256 features). Adding bigrams/trigrams captures short-range dependencies with minimal cost: ~48 active features per example hashed to 8192 dimensions. Inference remains a sparse dot-product + argmax, similar speed to label-linear.
Failure: If dev accuracy ≤ 0.3375 (label-linear) or latency exceeds label-linear by >20%.
```
```python
"""label-ngram: linear model over hashed positional n-grams (unigrams, bigrams, trigrams).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Exports: int32 magic=0x4E475231, d=16, H=0, C=65, feat_dim=8192;
float32 W[feat_dim][C], b[C] (row-major)."""
import json, math, struct, sys

import numpy as np

FEAT_DIM = 8192
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def fnv1a_32(key_bytes):
    """Deterministic 32-bit FNV-1a hash."""
    h = 0x811c9dc5
    for b in key_bytes:
        h ^= b
        h = (h * 0x01000193) & 0xffffffff
    return h


def extract_features(X, d):
    """X: (n, d) uint8. Returns list of (indices, values) for each example."""
    n = X.shape[0]
    all_indices = []
    for i in range(n):
        idx = []
        row = X[i]
        # unigrams: (pos, byte)
        for pos in range(d):
            key = bytes([0x01, pos, row[pos]])
            idx.append(fnv1a_32(key) % FEAT_DIM)
        # bigrams: (pos, byte1, byte2)
        for pos in range(d - 1):
            key = bytes([0x02, pos, row[pos], row[pos + 1]])
            idx.append(fnv1a_32(key) % FEAT_DIM)
        # trigrams: (pos, byte1, byte2, byte3)
        for pos in range(d - 2):
            key = bytes([0x03, pos, row[pos], row[pos + 1], row[pos + 2]])
            idx.append(fnv1a_32(key) % FEAT_DIM)
        # deduplicate (keep first occurrence)
        seen = set()
        uniq = []
        for h in idx:
            if h not in seen:
                seen.add(h)
                uniq.append(h)
        all_indices.append(np.array(uniq, dtype=np.int32))
    return all_indices


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    assert d == 16

    rng = np.random.default_rng(SEED)
    feat_indices = extract_features(X, d)
    n = X.shape[0]

    W = (rng.uniform(-1, 1, (FEAT_DIM, C)) * math.sqrt(6.0 / FEAT_DIM)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            yy = y[idx]
            # forward
            scores = b.copy()
            for i, ex_idx in enumerate(idx):
                scores += W[feat_indices[ex_idx]].sum(axis=0)
            # softmax grad
            scores -= scores.max(axis=1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad = (probs / len(yy)).astype(np.float32)  # (B, C)
            # backward
            grad_W = np.zeros_like(W)
            grad_b = grad.sum(axis=0)
            for i, ex_idx in enumerate(idx):
                fi = feat_indices[ex_idx]
                grad_W[fi] += grad[i]
            grad_W += L2 * W
            grads = [grad_W, grad_b]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # write weights: magic, d, H=0, C, feat_dim; then W (feat_dim*C), b (C)
    out = struct.pack("<5i", 0x4E475231, d, 0, C, FEAT_DIM)
    out += W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-ngram inference: ./infer weights.bin in.rec N
 * Weights format: int32 magic=0x4E475231, d, H=0, C, feat_dim;
 * float32 W[feat_dim][C] (row-major), b[C].
 * Input records: d bytes each. Output: 1 byte class index per record. */
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

static uint32_t fnv1a_32(const unsigned char *data, size_t len) {
    uint32_t h = 0x811c9dc5;
    for (size_t i = 0; i < len; i++) {
        h ^= data[i];
        h *= 0x01000193;
    }
    return h;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x4E475231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], FEAT_DIM = hd[4];
    if (H != 0 || C < 1 || C > 256 || d != 16 ||
        nw != 20 + 4L * ((long)FEAT_DIM * C + C)) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *W = (const float *)(hd + 5);
    const float *b = W + (long)FEAT_DIM * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    unsigned char keybuf[5];  // max len: type(1) + pos(1) + 3 bytes = 5

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(scores, b, C * sizeof(float));
        // unigrams
        for (int p = 0; p < d; p++) {
            keybuf[0] = 1; keybuf[1] = p; keybuf[2] = xi[p];
            uint32_t h = fnv1a_32(keybuf, 3) % FEAT_DIM;
            const float *w = W + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        // bigrams
        for (int p = 0; p < d - 1; p++) {
            keybuf[0] = 2; keybuf[1] = p; keybuf[2] = xi[p]; keybuf[3] = xi[p + 1];
            uint32_t h = fnv1a_32(keybuf, 4) % FEAT_DIM;
            const float *w = W + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        // trigrams
        for (int p = 0; p < d - 2; p++) {
            keybuf[0] = 3; keybuf[1] = p; keybuf[2] = xi[p]; keybuf[3] = xi[p + 1]; keybuf[4] = xi[p + 2];
            uint32_t h = fnv1a_32(keybuf, 5) % FEAT_DIM;
            const float *w = W + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```