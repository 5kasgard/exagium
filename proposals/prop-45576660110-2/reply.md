```hypothesis
The label-ngram family implements an interpolated n-gram language model (orders 1–3) with additive smoothing, tuned on the last 20% of training data. For next-character prediction on tinyshakespeare, local sequential statistics dominate: n-grams capture the strong positional-invariant dependencies (e.g., "the " → " ") that position-specific one-hot linear models miss, while interpolation gracefully backs off from sparse longer contexts. Expected to exceed label-linear's 0.3375 dev accuracy at comparable inference cost (three binary searches + small dense adds). Failure: dev accuracy ≤ 0.3375 or instruction count > 2× label-linear.
```
```python
import json, struct, sys, math
import numpy as np

SEED = 1
MAX_ORDER = 3
ALPHA = 0.1
LAMBDA_STEP = 0.05

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def pack_ctx(ctx, k):
    v = 0
    for b in ctx:
        v = (v << 8) | b
    return v

def unpack_ctx(v, k):
    ctx = bytearray(k)
    for i in range(k - 1, -1, -1):
        ctx[i] = v & 0xFF
        v >>= 8
    return ctx

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    K = MAX_ORDER

    X_recs = records(sys.argv[2])
    y_recs = records(sys.argv[3])
    n = len(X_recs)
    X = np.frombuffer(b"".join(X_recs), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in y_recs], dtype=np.int64)

    split = int(n * 0.8)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    def count_ngrams(Xd, yd):
        cnt = [{} for _ in range(K + 1)]
        for xi, yi in zip(Xd, yd):
            for k in range(1, K + 1):
                key = pack_ctx(xi[d - k:d], k)
                arr = cnt[k].get(key)
                if arr is None:
                    arr = np.zeros(C, np.int32)
                    cnt[k][key] = arr
                arr[yi] += 1
        return cnt

    cnt_tr = count_ngrams(X_tr, y_tr)
    probs_tr = [{} for _ in range(K + 1)]
    for k in range(1, K + 1):
        for key, arr in cnt_tr[k].items():
            tot = arr.sum()
            probs_tr[k][key] = ((arr + ALPHA) / (tot + ALPHA * C)).astype(np.float32)

    uniform = np.float32(1.0 / C)
    best_acc, best_lmb = -1.0, None
    steps = int(1.0 / LAMBDA_STEP) + 1
    for i in range(steps):
        l1 = i * LAMBDA_STEP
        for j in range(steps - i):
            l2 = j * LAMBDA_STEP
            l3 = 1.0 - l1 - l2
            lmb = [l1, l2, l3]
            correct = 0
            for xi, yi in zip(X_val, y_val):
                score = np.zeros(C, np.float32)
                for k in range(1, K + 1):
                    key = pack_ctx(xi[d - k:d], k)
                    p = probs_tr[k].get(key, uniform)
                    score += lmb[k - 1] * p
                if np.argmax(score) == yi:
                    correct += 1
            acc = correct / len(y_val)
            if acc > best_acc:
                best_acc, best_lmb = acc, lmb

    print(f"tuned lambdas={best_lmb} val_acc={best_acc:.4f}", file=sys.stderr)

    cnt_all = count_ngrams(X, y)
    probs_all = [{} for _ in range(K + 1)]
    for k in range(1, K + 1):
        for key, arr in cnt_all[k].items():
            tot = arr.sum()
            probs_all[k][key] = ((arr + ALPHA) / (tot + ALPHA * C)).astype(np.float32)

    sorted_items = []
    for k in range(1, K + 1):
        sorted_items.append(sorted(probs_all[k].items()))

    out = bytearray()
    out += struct.pack("<IBB2x", 0x4E47524D, K, C)
    out += struct.pack("<3f", *best_lmb)
    for k in range(1, K + 1):
        items = sorted_items[k - 1]
        out += struct.pack("<I", len(items))
        for key, p in items:
            out += bytes(unpack_ctx(key, k))
            out += p.tobytes()
    open(sys.argv[4], "wb").write(out)

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

static int32_t binsearch(const uint32_t *arr, uint32_t n, uint32_t key) {
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t mid = (lo + hi) >> 1;
        if (arr[mid] < key) lo = mid + 1;
        else hi = mid;
    }
    return (lo < n && arr[lo] == key) ? (int32_t)lo : -1;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    if (nw < 20) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    uint32_t magic = wb[0] | wb[1]<<8 | wb[2]<<16 | wb[3]<<24;
    if (magic != 0x4E47524D) { fprintf(stderr, "bad weights: magic\n"); return 3; }
    uint8_t K = wb[4];
    uint8_t C = wb[5];
    if (C != 65 || K < 1 || K > 5) { fprintf(stderr, "bad weights: C=%d K=%d\n", C, K); return 3; }

    float *lambdas = (float *)(wb + 8);
    uint8_t *ptr = wb + 8 + 3 * sizeof(float);

    uint32_t *nctx = malloc(K * sizeof(uint32_t));
    uint32_t **keys = malloc(K * sizeof(uint32_t*));
    float **probs = malloc(K * sizeof(float*));

    for (int k = 1; k <= K; k++) {
        if ((long)(ptr - wb) + 4 > nw) { fprintf(stderr, "truncated weights\n"); return 3; }
        nctx[k-1] = *(uint32_t*)ptr; ptr += 4;
        uint32_t n = nctx[k-1];
        keys[k-1] = malloc(n * sizeof(uint32_t));
        for (uint32_t i = 0; i < n; i++) {
            uint32_t key = 0;
            for (int b = 0; b < k; b++) key = (key << 8) | ptr[b];
            keys[k-1][i] = key;
            ptr += k;
        }
        size_t lp_bytes = (size_t)n * C * sizeof(float);
        if ((long)(ptr - wb) + lp_bytes > nw) { fprintf(stderr, "truncated weights\n"); return 3; }
        probs[k-1] = (float*)ptr;
        ptr += lp_bytes;
    }
    if ((long)(ptr - wb) != nw) { fprintf(stderr, "weights size mismatch\n"); return 3; }

    float uniform = 1.0f / C;
    float *score = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }
        memset(score, 0, C * sizeof(float));
        for (int ord = 1; ord <= K; ord++) {
            uint32_t key = 0;
            for (int b = 16 - ord; b < 16; b++) key = (key << 8) | xi[b];
            int32_t idx = binsearch(keys[ord-1], nctx[ord-1], key);
            float lambda = lambdas[ord-1];
            if (idx >= 0) {
                const float *p = probs[ord-1] + (size_t)idx * C;
                for (int c = 0; c < C; c++) score[c] += lambda * p[c];
            } else {
                for (int c = 0; c < C; c++) score[c] += lambda * uniform;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```