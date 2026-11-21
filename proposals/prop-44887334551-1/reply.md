```hypothesis
Family: label-nb (Naive Bayes with per-pixel categorical distributions).
Why: Optdigits pixels are 0–16 ordinal counts; a generative model with Laplace smoothing captures per-class pixel histograms efficiently. It requires only 64×256×10 log-probability tables and 10 log-priors, so inference is ~640 additions per item (far cheaper than label-linear’s 64×10 multiply-adds). Expected to match or exceed label-linear’s 96.1% accuracy. Failure sign: dev accuracy below 96%.
```
```python
"""Naive Bayes for label tasks with fixed-size byte items (categorical per feature).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic: fixed seed, single thread. Exports log-priors and log-probabilities per feature value per class."""
import json, struct, sys
import numpy as np

MAGIC = 0x4E424C31  # 'NLB1'
SEED = 1

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
    d, C = X.shape[1], card["classes"]
    # Count occurrences: cnt[c, j, v]
    cnt = np.ones((C, d, 256), dtype=np.int64)  # Laplace smoothing (+1)
    prior = np.ones(C, dtype=np.int64)
    for c in range(C):
        mask = (y == c)
        prior[c] += mask.sum()
        if mask.any():
            Xc = X[mask]
            for j in range(d):
                vals, freq = np.unique(Xc[:, j], return_counts=True)
                cnt[c, j, vals] += freq
    # Convert to log-probabilities
    log_prior = np.log(prior.astype(np.float64) / prior.sum()).astype(np.float32)
    # log_prob[c, j, v] = log( cnt[c,j,v] / cnt[c,j,:].sum() )
    denom = cnt.sum(axis=2, keepdims=True).astype(np.float64)
    log_prob = np.log(cnt.astype(np.float64) / denom).astype(np.float32)  # shape (C, d, 256)
    # Reorder to (d, 256, C) for infer.c's access pattern
    log_prob = np.transpose(log_prob, (1, 2, 0))
    # Pack weights
    out = struct.pack("<3i", MAGIC, d, C)
    out += log_prior.tobytes()
    out += log_prob.tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Naive Bayes inference: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: one byte class index (argmax log-posterior). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
    if (nw < 12 || hd[0] != 0x4E424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2];
    if (C < 1 || C > 256 || nw != 12 + 4L * C + 4L * d * 256 * C) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *log_prior = (const float *)(hd + 3);
    const float *log_prob = log_prior + C;  // [d][256][C]
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(scores, log_prior, C * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *tbl = log_prob + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += tbl[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```