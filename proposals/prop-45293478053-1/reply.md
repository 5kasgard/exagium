```hypothesis
Family: label-mnb (Multinomial Naive Bayes with add-alpha smoothing).
Why: Optdigits features are pixel counts 0-16, a natural fit for multinomial NB; inference is ~640 adds/item (10 classes * 64 features), far cheaper than label-linear's 5058 Ir/item while capturing per-class feature distributions.
Failure signal: dev accuracy ≤ 0.9613 (label-linear) or Ir/item ≥ 5000.
```

```python
import json, struct, sys
import numpy as np

ALPHAS = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

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
    d = card["input"]["bytes"]   # 64
    C = card["classes"]          # 10
    V = 17                       # pixel values 0..16

    n = len(X)
    split = int(0.8 * n)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    n_train, n_val = len(X_train), len(X_val)

    # Count on training split (independent of alpha)
    class_count = np.zeros(C, dtype=np.int64)
    feat_count = np.zeros((C, d, V), dtype=np.int64)
    for i in range(n_train):
        c = y_train[i]
        class_count[c] += 1
        for j in range(d):
            v = X_train[i, j]
            feat_count[c, j, v if v < V else V-1] += 1

    best_alpha, best_acc = ALPHAS[0], -1.0
    for alpha in ALPHAS:
        log_prior = np.log((class_count + alpha) / (n_train + C * alpha)).astype(np.float32)
        log_prob = np.zeros((C, d, V), dtype=np.float32)
        for c in range(C):
            denom = class_count[c] + V * alpha
            log_prob[c] = np.log((feat_count[c] + alpha) / denom)

        # Vectorized validation
        scores = np.tile(log_prior, (n_val, 1))
        for j in range(d):
            v = np.clip(X_val[:, j], 0, V-1)
            scores += log_prob[:, j, v].T
        acc = np.mean(np.argmax(scores, axis=1) == y_val)
        if acc > best_acc:
            best_acc, best_alpha = acc, alpha

    # Retrain on all data with best_alpha
    class_count = np.zeros(C, dtype=np.int64)
    feat_count = np.zeros((C, d, V), dtype=np.int64)
    for i in range(n):
        c = y[i]
        class_count[c] += 1
        for j in range(d):
            v = X[i, j]
            feat_count[c, j, v if v < V else V-1] += 1

    log_prior = np.log((class_count + best_alpha) / (n + C * best_alpha)).astype(np.float32)
    log_prob = np.zeros((C, d, V), dtype=np.float32)
    for c in range(C):
        denom = class_count[c] + V * best_alpha
        log_prob[c] = np.log((feat_count[c] + best_alpha) / denom)

    # Write weights: magic, d, C, V, alpha, log_prior[C], log_prob[C*d*V]
    magic = 0x4D4E4231  # MNB1
    out = bytearray()
    out += struct.pack("<4i", magic, d, C, V)
    out += struct.pack("<f", best_alpha)
    out += log_prior.astype("<f4").tobytes()
    out += log_prob.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x4D4E4231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], V = hd[3];
    if (d != 64 || C != 10 || V != 17) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16 + 4 + (long)C*4 + (long)C*d*V*4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    float *ptr = (float *)(wb + 16);
    float *log_prior = ptr + 1;
    float *log_prob = ptr + 1 + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = log_prior[c];
        for (int j = 0; j < d; j++) {
            int v = xi[j]; if (v >= V) v = V-1;
            for (int c = 0; c < C; c++)
                scores[c] += log_prob[(long)c*d*V + j*V + v];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```