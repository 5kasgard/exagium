```hypothesis
Family: label-hashngram (linear model on hashed n-gram features up to length 3).
Why: The task is character-level language modeling on TinyShakespeare. A linear model on one-hot per-position bytes (label-linear) ignores local character interactions. Hashing 1/2/3-grams into a fixed feature table captures n-gram statistics without blowing up memory, and stays a single matrix-vector multiply at inference. With 4096 features it matches label-linear's weight size but models context better.
Failure signal: If dev accuracy ≤ 0.3375 (label-linear) or inference cost exceeds label-linear by >20%.
```

```python
"""Hash n-gram linear model for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Features: all 1/2/3-grams of the 16-byte context, hashed into F=4096 bins.
Linear multinomial logistic regression with Adam (lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n))).
Weights format: magic 0x484E4731 ('HNG1'), d=16, H=0, C=65, F=4096; then float32 W[F][C] row-major, then float32 b[C]."""
import json, math, struct, sys

import numpy as np

F = 4096
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def hash_ngram(ngram):
    h = 0
    for b in ngram:
        h = (h * 31 + b) & 0xFFFFFFFF
    return h % F

def extract_features(seq):
    """Return sorted unique feature indices for all 1/2/3-grams in seq (len=16)."""
    idxs = set()
    L = len(seq)
    for n in (1, 2, 3):
        for i in range(L - n + 1):
            idxs.add(hash_ngram(seq[i:i+n]))
    return sorted(idxs)

def main():
    card = json.load(open(sys.argv[1]))
    X_recs = records(sys.argv[2])
    y_recs = records(sys.argv[3])
    X = np.frombuffer(b"".join(X_recs), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in y_recs], dtype=np.int64)
    C = card["classes"]
    n = X.shape[0]

    # Precompute sparse feature indices for each example
    feat_idx = [extract_features(X[i]) for i in range(n)]

    rng = np.random.default_rng(SEED)
    W = np.zeros((F, C), dtype=np.float32)
    b = np.zeros(C, dtype=np.float32)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    mb = np.zeros_like(b); vb = np.zeros_like(b)

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            yy = y[idx]
            # forward
            scores = np.zeros((len(idx), C), dtype=np.float32)
            for b_i, ex_idx in enumerate(idx):
                for f in feat_idx[ex_idx]:
                    scores[b_i] += W[f]
            scores += b
            # softmax + gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores, dtype=np.float32)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            probs /= len(yy)  # gradient of CE loss w.r.t. scores

            # accumulate gradients for W (sparse) and b (dense)
            gW = np.zeros_like(W)
            for b_i, ex_idx in enumerate(idx):
                g = probs[b_i]
                for f in feat_idx[ex_idx]:
                    gW[f] += g
            gW += L2 * W
            gb = probs.sum(axis=0) + L2 * b

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            mW *= beta1; mW += (1 - beta1) * gW
            vW *= beta2; vW += (1 - beta2) * gW * gW
            mb *= beta1; mb += (1 - beta1) * gb
            vb *= beta2; vb += (1 - beta2) * gb * gb
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            mb_hat = mb / (1 - beta1 ** t)
            vb_hat = vb / (1 - beta2 ** t)
            W -= LR * mW_hat / (np.sqrt(vW_hat) + 1e-8)
            b -= LR * mb_hat / (np.sqrt(vb_hat) + 1e-8)

    # Write weights
    magic = 0x484E4731  # 'HNG1'
    d = card["input"]["bytes"]
    H = 0
    out = struct.pack("<5i", magic, d, H, C, F)
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Hash n-gram linear model inference: ./infer weights.bin in.rec N
 * Input record: 16 raw bytes. Output record: 1 byte class index (argmax).
 * Weights format: magic 0x484E4731, d, H, C, F; float32 W[F][C] row-major; float32 b[C]. */
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

static uint32_t hash_ngram(const unsigned char *seq, int len) {
    uint32_t h = 0;
    for (int i = 0; i < len; i++) h = h * 31 + seq[i];
    return h;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x484E4731) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], F = hd[4];
    if (d != 16 || H != 0 || C < 1 || C > 256) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 20L + 4L * ((long)F * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W = (const float *)(hd + 5);
    const float *b = W + (long)F * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    uint32_t feat_idx[45];  // max 16+15+14 = 45 n-grams
    int feat_cnt;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // extract unique hashed n-gram features (1,2,3-grams)
        feat_cnt = 0;
        for (int n = 1; n <= 3; n++) {
            for (int j = 0; j <= d - n; j++) {
                uint32_t h = hash_ngram(xi + j, n) % F;
                // deduplicate (small linear search, 45 items max)
                int dup = 0;
                for (int f = 0; f < feat_cnt; f++) if (feat_idx[f] == h) { dup = 1; break; }
                if (!dup) feat_idx[feat_cnt++] = h;
            }
        }

        // compute scores = b + sum_{f in feats} W[f]
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int f = 0; f < feat_cnt; f++) {
            const float *wf = W + (long)feat_idx[f] * C;
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