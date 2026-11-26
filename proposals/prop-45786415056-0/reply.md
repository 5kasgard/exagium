```hypothesis
Family: label-emb (factorized linear with per-column embeddings). Learns a D-dimensional embedding for each of the 256 possible values per column (D=4), sums embeddings across 14 columns, then applies a linear layer to 2 classes. Shares statistical strength across categories, regularizing rare values; inference costs ~14*D + D*C = 64 FLOPs vs linear's ~k*C (k≈150 expanded dim). Expected to beat label-linear (0.8524) on accuracy at lower CPU cost. Failure: dev accuracy ≤ 0.8524 or Ir/item ≥ 719.
```
```python
import json, math, struct, sys
import numpy as np

D = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 100

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
    C = card["classes"]
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)

    E = [rng.normal(0, 0.01, (256, D)).astype(np.float32) for _ in range(d)]
    W = rng.normal(0, 0.01, (D, C)).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = E + [W, b]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            B = len(yb)

            acc = np.zeros((B, D), np.float32)
            for j in range(d):
                acc += E[j][xb[:, j]]
            logits = acc @ W + b

            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = probs / B

            grad_W = acc.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0) + L2 * b
            grad_acc = grad_logits @ W.T

            grad_E = [np.zeros_like(e) for e in E]
            for j in range(d):
                np.add.at(grad_E[j], xb[:, j], grad_acc)
                grad_E[j] += L2 * E[j]

            grads = grad_E + [grad_W, grad_b]
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * g * g
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    out = struct.pack("<4i", 0x454D4231, d, D, C)
    for j in range(d):
        out += E[j].astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x454D4231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 14) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *E = (const float *)(hd + 4);
    const float *W = E + (long)d * 256 * D;
    const float *b = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * D);
    float *sc = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memset(acc, 0, sizeof(float) * D);
        for (int j = 0; j < d; j++) {
            const float *emb = E + ((long)j * 256 + xi[j]) * D;
            for (int f = 0; f < D; f++) acc[f] += emb[f];
        }
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int f = 0; f < D; f++) sum += acc[f] * W[(long)f * C + c];
            sc[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```