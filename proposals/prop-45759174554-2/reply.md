```hypothesis
Family: label-hashed-ngram (fastText-style hashed n-gram embeddings + linear output).
Why: Character-level language modeling relies heavily on local n-gram statistics (1–3 grams). A hashed embedding table captures these interactions compactly (8192×16 floats) and is faster than GBT while richer than positional one-hot linear. The model sums embeddings for all 45 n-grams in the 16-byte context, then applies a linear layer.
Failure signal: If dev accuracy ≤ 34.0%, the n-gram features add no predictive value over label-linear's positional unigrams.
```

```python
import json, math, struct, sys
import numpy as np

HASH_SIZE = 8192
D = 16
NGRAM_LENS = (1, 2, 3)
SEED = 1
LR = 0.05
L2 = 1e-5
BATCH = 64
EPOCHS = 10
SAMPLES = 600000

def fnv1a_64(data):
    h = 14695981039346656037
    for b in data:
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def get_ngram_hashes(context):
    hashes = []
    for L in NGRAM_LENS:
        for start in range(0, 17 - L):
            ngram = context[start:start+L]
            h = fnv1a_64(ngram) % HASH_SIZE
            hashes.append(h)
    return hashes

def main():
    card = json.load(open(sys.argv[1]))
    train_in = sys.argv[2]
    train_key = sys.argv[3]
    weights_out = sys.argv[4]

    contexts = records(train_in)
    labels = np.array([r[0] for r in records(train_key)], dtype=np.int64)
    N = len(contexts)
    C = card["classes"]
    assert C == 65

    ngram_indices = np.zeros((N, 45), dtype=np.int32)
    for i, ctx in enumerate(contexts):
        ngram_indices[i] = get_ngram_hashes(ctx)

    rng = np.random.default_rng(SEED)
    embeddings = (rng.uniform(-1, 1, (HASH_SIZE, D)) * math.sqrt(6.0 / D)).astype(np.float32)
    W_out = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b_out = np.zeros(C, dtype=np.float32)

    params = [embeddings, W_out, b_out]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    max_epochs = min(EPOCHS, math.ceil(SAMPLES / N))
    for _ in range(max_epochs):
        order = rng.permutation(N)
        for s in range(0, N, BATCH):
            idx = order[s:s+BATCH]
            batch_ngram_idx = ngram_indices[idx]
            yy = labels[idx]

            emb = embeddings[batch_ngram_idx]
            sum_emb = emb.sum(axis=1)

            logits = sum_emb @ W_out + b_out
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            dlogits = probs / len(yy)

            dW_out = sum_emb.T @ dlogits + L2 * W_out
            db_out = dlogits.sum(axis=0)
            dsum_emb = dlogits @ W_out.T

            flat_idx = batch_ngram_idx.ravel()
            repeated_grad = np.repeat(dsum_emb, 45, axis=0)
            dembeddings = np.zeros_like(embeddings)
            np.add.at(dembeddings, flat_idx, repeated_grad)
            dembeddings += L2 * embeddings

            grads = [dembeddings, dW_out, db_out]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    magic = 0x484E4731
    out = struct.pack("<4i", magic, HASH_SIZE, D, C)
    out += embeddings.astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    out += b_out.astype("<f4").tobytes()
    open(weights_out, "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
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

static uint64_t fnv1a_64(const unsigned char *data, int len) {
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < len; i++) {
        h ^= data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x484E4731) { fprintf(stderr, "bad weights\n"); return 3; }
    int hash_size = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256 || D < 1 || hash_size < 1) {
        fprintf(stderr, "bad weights dims\n"); return 3;
    }
    long expected = 16 + 4L * ((long)hash_size * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *embeddings = (const float *)(hd + 4);
    const float *W_out = embeddings + (long)hash_size * D;
    const float *b_out = W_out + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * D);
    float *logits = malloc(sizeof(float) * C);

    const int ng_lens[3] = {1, 2, 3};

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }

        for (int d = 0; d < D; d++) sum_emb[d] = 0.0f;

        for (int li = 0; li < 3; li++) {
            int L = ng_lens[li];
            for (int start = 0; start <= 16 - L; start++) {
                uint64_t h = fnv1a_64(xi + start, L);
                int idx = (int)(h % (uint64_t)hash_size);
                const float *emb = embeddings + (long)idx * D;
                for (int d = 0; d < D; d++) sum_emb[d] += emb[d];
            }
        }

        for (int c = 0; c < C; c++) {
            float acc = b_out[c];
            for (int d = 0; d < D; d++) acc += sum_emb[d] * W_out[(long)d * C + c];
            logits[c] = acc;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```