```hypothesis
Family: label-onehot. Treat each of the 64 pixels (values 0..16) as a categorical feature via one-hot encoding (1088 features) and train a regularized logistic regression (Adam, L2=1e-3). This captures non-linear per-pixel responses that the ordinal z-scoring of label-linear cannot, while keeping inference at ~640 additions/item (similar cost). If dev accuracy does not exceed 96.2% (label-linear is 96.13%), the approach fails to justify the extra parameters.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-3
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = records(sys.argv[2])
    y_bytes = records(sys.argv[3])
    X = np.frombuffer(b"".join(X_bytes), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in y_bytes], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    max_val = 16
    feature_dim = d * (max_val + 1)

    n = X.shape[0]
    Z = np.zeros((n, feature_dim), dtype=np.float32)
    for j in range(d):
        base = j * (max_val + 1)
        Z[np.arange(n), base + X[:, j]] = 1.0

    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (feature_dim, C)) * math.sqrt(6.0 / feature_dim)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    beta1, beta2 = 0.9, 0.999

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            z = Z[idx]
            yy = y[idx]
            logits = z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad = probs / len(yy)
            gW = z.T @ grad + L2 * W
            gb = grad.sum(axis=0)
            t += 1
            mW = beta1 * mW + (1 - beta1) * gW
            vW = beta2 * vW + (1 - beta2) * gW * gW
            mb = beta1 * mb + (1 - beta1) * gb
            vb = beta2 * vb + (1 - beta2) * gb * gb
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            mb_hat = mb / (1 - beta1 ** t)
            vb_hat = vb / (1 - beta2 ** t)
            W -= LR * mW_hat / (np.sqrt(vW_hat) + 1e-8)
            b -= LR * mb_hat / (np.sqrt(vb_hat) + 1e-8)

    magic = 0x4F4E4548
    header = struct.pack("<4i", magic, d, C, max_val)
    out = header + W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
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

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x4F4E4548) { fprintf(stderr, "bad weights: magic mismatch\n"); return 3; }
    int d = hd[1];
    int C = hd[2];
    int max_val = hd[3];
    if (d <= 0 || C <= 0 || C > 256 || max_val < 0 || max_val > 255) {
        fprintf(stderr, "bad weights: invalid header values\n"); return 3;
    }
    int feature_dim = d * (max_val + 1);
    long expected = 16L + 4L * ((long)feature_dim * C + C);
    if (nw != expected) {
        fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected);
        return 3;
    }
    const float *W = (const float *)(wb + 16);
    const float *b = (const float *)(wb + 16 + 4L * feature_dim * C);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            if (v > max_val) v = max_val;
            const float *w = W + (long)(j * (max_val + 1) + v) * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```