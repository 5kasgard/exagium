```hypothesis
Family: label-naivebayes. Multinomial Naive Bayes with per-pixel per-class histograms (Laplace smoothing).  
Optdigits pixels are ordinal 0..16; generative model captures per-class pixel distributions well.  
Inference: 64*10 log-prob lookups + adds per item, similar cost to linear.  
Fails if accuracy <= label-linear (0.961) or cost > 1.5x linear.
```

```python
import json, struct, sys
import numpy as np

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
    d = card["input"]["bytes"]
    C = card["classes"]
    V = 17  # pixel values 0..16

    # Count
    class_counts = np.zeros(C, dtype=np.int64)
    pixel_counts = np.zeros((d, V, C), dtype=np.int64)
    for i in range(X.shape[0]):
        c = y[i]
        class_counts[c] += 1
        for j in range(d):
            v = X[i, j]
            pixel_counts[j, v, c] += 1

    # Laplace smoothing
    pixel_counts += 1
    total = X.shape[0]
    log_prior = np.log(class_counts / total).astype(np.float32)

    # Log probabilities
    denom = class_counts.astype(np.float64) + V  # (C,)
    log_prob = np.log(pixel_counts.astype(np.float64) / denom[None, None, :]).astype(np.float32)

    # Write weights
    magic = 0x4E424159  # 'YABN'
    out = struct.pack("<4i", magic, d, C, V)
    out += log_prior.tobytes()
    out += log_prob.tobytes()  # flattened (d, V, C) in C order
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
    if (nw < 16 || hd[0] != 0x4E424159) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], V = hd[3];
    if (C < 1 || C > 256 || V != 17 || d != 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * C + 4L * d * V * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *log_prior = (const float *)(hd + 4);
    const float *log_prob = log_prior + C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float best_score = -1e30f;
        unsigned char best_class = 0;
        for (int c = 0; c < C; c++) {
            float score = log_prior[c];
            for (int j = 0; j < d; j++) {
                int v = xi[j];
                if (v >= V) v = V - 1;
                score += log_prob[((long)j * V + v) * C + c];
            }
            if (score > best_score) {
                best_score = score;
                best_class = (unsigned char)c;
            }
        }
        put_rec(stdout, &best_class, 1);
    }
    return 0;
}
```