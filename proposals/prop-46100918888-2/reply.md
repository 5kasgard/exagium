```hypothesis
Family: label-ngram (compositional n-gram with unigram + bigram features).
Why: Character-level language modeling relies heavily on local bigram statistics (e.g., "th", "he", "in"). The label-linear model uses only unigram (per-position) features, missing these interactions. This model adds compositional bigram features (T2a[pos][byte_i] + T2b[pos][byte_{i+1}]) with the same width as unigrams, capturing pairwise interactions at only ~3× the inference cost of label-linear (still far below GBDT). The parameters are learned end-to-end with Adam.
Failure signal: If dev accuracy ≤ 0.34 (label-linear's 0.3375) or instructions/item > 25k, the added complexity doesn't pay off.
```
```python
import json, math, struct, sys

import numpy as np

# Hyperparameters chosen on tuning split (last 20% of train.in)
H = 64          # hidden size (0 = linear)
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 256
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
    d = card["input"]["bytes"]
    C = card["classes"]
    n = len(X_bytes)
    assert len(y_bytes) == n
    X = np.frombuffer(b"".join(X_bytes), dtype=np.uint8).reshape(n, d)
    y = np.array([r[0] for r in y_bytes], dtype=np.int64)

    rng = np.random.default_rng(SEED)
    W = H if H else C

    # Xavier/He initialization matching label-linear scaling
    fan_in_uni = d * 256
    fan_in_bi = (d - 1) * 256
    a_uni = math.sqrt(6.0 / fan_in_uni)
    a_bi = math.sqrt(6.0 / fan_in_bi)

    T1 = rng.uniform(-a_uni, a_uni, (d, 256, W)).astype(np.float32)
    T2a = rng.uniform(-a_bi, a_bi, (d - 1, 256, W)).astype(np.float32)
    T2b = rng.uniform(-a_bi, a_bi, (d - 1, 256, W)).astype(np.float32)
    b1 = np.zeros(W, np.float32)

    if H:
        W2 = rng.normal(0, math.sqrt(2.0 / H), (H, C)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [T1, T2a, T2b, b1, W2, b2]
    else:
        params = [T1, T2a, T2b, b1]

    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    idx_all = np.arange(n)

    for _ in range(max_epochs):
        rng.shuffle(idx_all)
        for s in range(0, n, BATCH):
            idx = idx_all[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            B = len(idx)

            # Forward
            acc = b1[None, :]  # (B, W) via broadcast
            for j in range(d):
                acc += T1[j, xb[:, j], :]
            for i in range(d - 1):
                acc += T2a[i, xb[:, i], :]
                acc += T2b[i, xb[:, i + 1], :]

            if H:
                h = np.maximum(acc, 0, dtype=np.float32)
                logits = h @ W2 + b2
            else:
                logits = acc

            # Softmax gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = probs / B

            # Backward
            if H:
                grad_h = grad_logits @ W2.T
                grad_acc = grad_h * (acc > 0)
                grad_W2 = h.T @ grad_logits
                grad_b2 = grad_logits.sum(axis=0)
                grads = [None, None, None, None, grad_W2, grad_b2]
            else:
                grad_acc = grad_logits
                grads = [None, None, None, None]

            grad_b1 = grad_acc.sum(axis=0).astype(np.float32)
            grad_T1 = np.zeros_like(T1)
            grad_T2a = np.zeros_like(T2a)
            grad_T2b = np.zeros_like(T2b)
            for j in range(d):
                np.add.at(grad_T1[j], xb[:, j], grad_acc)
            for i in range(d - 1):
                np.add.at(grad_T2a[i], xb[:, i], grad_acc)
                np.add.at(grad_T2b[i], xb[:, i + 1], grad_acc)

            grads[0] = grad_T1.astype(np.float32)
            grads[1] = grad_T2a.astype(np.float32)
            grads[2] = grad_T2b.astype(np.float32)
            grads[3] = grad_b1

            if L2:
                grads[0] += L2 * T1
                grads[1] += L2 * T2a
                grads[2] += L2 * T2b
                grads[3] += L2 * b1
                if H:
                    grads[4] += L2 * W2
                    grads[5] += L2 * b2

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights: magic 0x4E473231 ('12GN'), d, H, C
    magic = 0x4E473231
    out = struct.pack("<4i", magic, d, H, C)
    out += T1.astype("<f4").tobytes()
    out += T2a.astype("<f4").tobytes()
    out += T2b.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H:
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
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
    uint32_t k = buf[*pos] | (buf[*pos + 1] << 8) | (buf[*pos + 2] << 16) | ((uint32_t)buf[*pos + 3] << 24);
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
    if (nw < 16 || hd[0] != 0x4E473231) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }

    long expected = 16;
    expected += 4L * d * 256 * W;
    expected += 4L * (d - 1) * 256 * W;
    expected += 4L * (d - 1) * 256 * W;
    expected += 4L * W;
    if (H) {
        expected += 4L * H * C;
        expected += 4L * C;
    }
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *T1 = p; p += (long)d * 256 * W;
    const float *T2a = p; p += (long)(d - 1) * 256 * W;
    const float *T2b = p; p += (long)(d - 1) * 256 * W;
    const float *b1 = p; p += W;
    const float *W2 = H ? p : NULL; if (H) p += (long)H * C;
    const float *b2 = H ? p : NULL;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = (float *)malloc(sizeof(float) * W);
    float *h = H ? (float *)malloc(sizeof(float) * H) : NULL;
    float *sc = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(acc, b1, sizeof(float) * W);

        for (int j = 0; j < d; j++) {
            const float *t = T1 + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }

        for (int i2 = 0; i2 < d - 1; i2++) {
            const float *ta = T2a + ((long)i2 * 256 + xi[i2]) * W;
            const float *tb = T2b + ((long)i2 * 256 + xi[i2 + 1]) * W;
            for (int w = 0; w < W; w++) acc[w] += ta[w] + tb[w];
        }

        if (H) {
            for (int u = 0; u < H; u++) h[u] = acc[u] > 0.0f ? acc[u] : 0.0f;
            for (int c = 0; c < C; c++) sc[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float hu = h[u];
                if (hu == 0.0f) continue;
                const float *w2u = W2 + (long)u * C;
                for (int c = 0; c < C; c++) sc[c] += hu * w2u[c];
            }
        } else {
            for (int c = 0; c < C; c++) sc[c] = acc[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(acc); if (h) free(h); free(sc);
    return 0;
}
```