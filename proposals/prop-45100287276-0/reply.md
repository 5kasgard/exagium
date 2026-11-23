```hypothesis
Family: label-nb (Naive Bayes with Laplace smoothing). Adult features are quantized to 256 bins each, making them naturally categorical. Naive Bayes exploits per-class feature distributions, is deterministic, trains in one pass, and has tiny weights (14*256*2 floats). Inference is ~30 adds/item, far cheaper than label-linear's 719 Ir/item. If dev accuracy is below 0.8524, the conditional independence assumption hurts too much.
```

```python
import json, struct, sys
import numpy as np

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    V = 256
    X_bytes = b"".join(records(sys.argv[2]))
    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(-1, d)
    y_bytes = b"".join(records(sys.argv[3]))
    y = np.frombuffer(y_bytes, dtype=np.uint8)
    n = X.shape[0]
    class_counts = np.zeros(C, dtype=np.int64)
    feature_counts = np.zeros((C, d, V), dtype=np.int64)
    for i in range(n):
        c = y[i]
        class_counts[c] += 1
        for j in range(d):
            v = X[i, j]
            feature_counts[c, j, v] += 1
    alpha = 1.0
    log_prior = np.log((class_counts + alpha) / (n + C * alpha)).astype(np.float32)
    denom = class_counts[:, None] + V * alpha
    prob = (feature_counts + alpha) / denom[:, :, None]
    log_prob = np.log(prob).astype(np.float32)
    magic = 0x4E424159
    header = struct.pack("<Iiii", magic, d, C, V)
    with open(sys.argv[4], "wb") as f:
        f.write(header)
        f.write(log_prior.tobytes())
        f.write(log_prob.tobytes())

if __name__ == "__main__":
    main()
```

```c
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | (uint32_t)wb[3] << 24;
    if (magic != 0x4E424159) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int32_t *hd = (int32_t *)(wb + 4);
    int d = hd[0], C = hd[1], V = hd[2];
    if (C < 1 || C > 256 || d < 1 || d > 256 || V != 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * C * (1 + (long)d * V);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *log_prior = (const float *)(wb + 16);
    const float *log_prob = log_prior + C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(C * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = log_prior[c];
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            for (int c = 0; c < C; c++) {
                scores[c] += log_prob[((long)c * d + j) * V + v];
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```