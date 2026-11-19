```hypothesis
Family: label-ngram (n-gram embedding model with hashing).
Why: Character-level language modeling benefits from local n-gram statistics (e.g., "the", "ing"). A linear model over independent positions misses these interactions. Hashing n-grams into a fixed embedding table captures them efficiently.
Failure: If dev accuracy <= 0.3375 (label-linear) or inference cost exceeds label-linear significantly.
```

```python
"""label-ngram: n-gram embedding model for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Architecture: extract n-grams of lengths 3,4,5 from 16-byte context, hash to V buckets (FNV-1a),
sum embeddings (V x D), linear to C classes. Adam lr=0.01, L2=1e-4, batch=64, epochs min(100, ceil(600000/n)).
Weights format: magic 0x4E47524D, V, D, C; float32 E[V][D], W[D][C], b[C] (all little-endian)."""
import json, math, struct, sys

import numpy as np

V = 8192          # vocabulary size (hash buckets)
D = 32            # embedding dimension
NGRAM_LENS = (3, 4, 5)
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 100
MAGIC = 0x4E47524D  # 'NGRM' in hex (little-endian bytes: 4D 52 47 4E)

def fnv1a_32(buf: bytes) -> int:
    h = 0x811c9dc5
    for b in buf:
        h ^= b
        h = (h * 0x01000193) & 0xffffffff
    return h

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def extract_ngrams(ctx: bytes):
    """Return list of n-gram hashes for a 16-byte context."""
    hashes = []
    for L in NGRAM_LENS:
        for s in range(0, 17 - L):
            hashes.append(fnv1a_32(ctx[s:s+L]) % V)
    return hashes

def main():
    card = json.load(open(sys.argv[1]))
    X_recs = records(sys.argv[2])
    y_recs = records(sys.argv[3])
    n = len(X_recs)
    assert all(len(r) == card["input"]["bytes"] for r in X_recs)
    assert all(len(r) == 1 for r in y_recs)
    C = card["classes"]

    # Precompute n-gram indices for all training examples
    ngram_indices = np.empty((n, 39), dtype=np.int32)  # 14+13+12 = 39
    y = np.empty(n, dtype=np.int64)
    for i, (x, yi) in enumerate(zip(X_recs, y_recs)):
        ngram_indices[i] = extract_ngrams(x)
        y[i] = yi[0]

    rng = np.random.default_rng(SEED)
    # Initialize parameters
    scale_e = 0.1
    E = rng.uniform(-scale_e, scale_e, (V, D)).astype(np.float32)
    scale_w = math.sqrt(6.0 / (D + C))
    W = rng.uniform(-scale_w, scale_w, (D, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    params = [E, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def softmax_cross_entropy_grad(logits, targets):
        # logits: (B, C), targets: (B,)
        B = logits.shape[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits, dtype=np.float32)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(B), targets] -= 1.0
        return probs / B  # (B, C)

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            B = len(idx)
            ng_idx = ngram_indices[idx]           # (B, 39)
            yy = y[idx]                           # (B,)

            # Forward
            emb = E[ng_idx]                       # (B, 39, D)
            summed = emb.sum(axis=1)              # (B, D)
            logits = summed @ W + b               # (B, C)

            # Backward
            g_logits = softmax_cross_entropy_grad(logits, yy)  # (B, C)
            g_summed = g_logits @ W.T             # (B, D)
            g_W = summed.T @ g_logits + L2 * W    # (D, C)
            g_b = g_logits.sum(axis=0)            # (C,)
            # Gradient for E: scatter-add g_summed to n-gram positions
            g_E = np.zeros_like(E)
            np.add.at(g_E, ng_idx.reshape(-1), np.repeat(g_summed, 39, axis=0))
            g_E += L2 * E

            grads = [g_E, g_W, g_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    out = struct.pack("<4i", MAGIC, V, D, C)
    out += E.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-ngram inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic 0x4E47524D, V, D, C; float32 E[V][D], W[D][C], b[C]. */
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

static uint32_t fnv1a_32(const unsigned char *buf, int len) {
    uint32_t h = 0x811c9dc5;
    for (int i = 0; i < len; i++) {
        h ^= buf[i];
        h *= 0x01000193;
    }
    return h;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x4E47524D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int V = hd[1], D = hd[2], C = hd[3];
    if (V <= 0 || D <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)V * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *E = (const float *)(hd + 4);
    const float *W = E + (long)V * D;
    const float *b = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * D);
    float *logits = malloc(sizeof(float) * C);
    if (!acc || !logits) { fprintf(stderr, "malloc failed\n"); exit(3); }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }

        for (int d = 0; d < D; d++) acc[d] = 0.0f;
        /* n-grams of lengths 3,4,5 */
        for (int L = 3; L <= 5; L++) {
            for (int s = 0; s <= 16 - L; s++) {
                uint32_t h = fnv1a_32(xi + s, L);
                int idx = h % (uint32_t)V;
                const float *e = E + (long)idx * D;
                for (int d = 0; d < D; d++) acc[d] += e[d];
            }
        }
        for (int c = 0; c < C; c++) logits[c] = b[c];
        for (int d = 0; d < D; d++) {
            float ad = acc[d];
            if (ad == 0.0f) continue;
            const float *w = W + (long)d * C;
            for (int c = 0; c < C; c++) logits[c] += ad * w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```