```hypothesis
Family: label-gradlinear. Augments the 64 pixels with 56 horizontal and 56 vertical gradients (176 features), trains logistic regression on z-scored features, then folds gradient weights into effective per-pixel weights for inference. Captures stroke/edge structure critical for digits at identical inference cost to label-linear. Fails if dev accuracy ≤ 0.965 (does not beat label-linear's 0.9613).
```
```python
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


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
    assert d == 64

    # Reshape to images for gradient computation
    X_img = X.reshape(-1, 8, 8).astype(np.float64)
    n = X.shape[0]

    # Build 176 features: 64 pixels + 56 hgrad + 56 vgrad
    F = np.zeros((n, 176), dtype=np.float64)
    F[:, :64] = X
    hgrad = X_img[:, :, 1:] - X_img[:, :, :-1]          # (n, 8, 7)
    F[:, 64:120] = hgrad.reshape(n, 56)
    vgrad = X_img[:, 1:, :] - X_img[:, :-1, :]          # (n, 7, 8)
    F[:, 120:176] = vgrad.reshape(n, 56)

    # Z-score each feature
    mu = F.mean(axis=0)
    sd = F.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (F - mu) / sd

    # Logistic regression with Adam (linear, 176 -> 10)
    rng = np.random.default_rng(SEED)
    k = 176
    W = np.zeros((k, C), np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            logits = z @ W + b
            g = softmax_grad(logits, yy)
            grad_W = z.T @ g + L2 * W
            grad_b = g.sum(axis=0)
            grads = [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Fold gradient weights into effective per-pixel weights for 64 inputs
    coeff = np.zeros((64, C), dtype=np.float32)
    const = np.zeros(C, dtype=np.float32)

    for c in range(C):
        w_pix = W[:64, c]; sd_pix = sd[:64].astype(np.float32); mu_pix = mu[:64].astype(np.float32)
        w_h = W[64:120, c]; sd_h = sd[64:120].astype(np.float32); mu_h = mu[64:120].astype(np.float32)
        w_v = W[120:176, c]; sd_v = sd[120:176].astype(np.float32); mu_v = mu[120:176].astype(np.float32)

        coeff_pix = w_pix / sd_pix
        coeff_h = w_h / sd_h
        coeff_v = w_v / sd_v

        c_coeff = coeff_pix.copy().reshape(8, 8)
        coeff_h_img = coeff_h.reshape(8, 7)
        coeff_v_img = coeff_v.reshape(7, 8)

        # Horizontal gradients: h[r,c] = x[r,c+1] - x[r,c]
        for r in range(8):
            for c_ in range(8):
                val = 0.0
                if c_ < 7:
                    val -= coeff_h_img[r, c_]
                if c_ > 0:
                    val += coeff_h_img[r, c_ - 1]
                c_coeff[r, c_] += val

        # Vertical gradients: v[r,c] = x[r+1,c] - x[r,c]
        for r in range(8):
            for c_ in range(8):
                val = 0.0
                if r < 7:
                    val -= coeff_v_img[r, c_]
                if r > 0:
                    val += coeff_v_img[r - 1, c_]
                c_coeff[r, c_] += val

        coeff[:, c] = c_coeff.ravel()

        const_c = b[c]
        const_c -= np.sum(mu_pix * coeff_pix)
        const_c -= np.sum(mu_h * coeff_h)
        const_c -= np.sum(mu_v * coeff_v)
        const[c] = const_c

    # Build lookup table T[64][256][10] = v * coeff
    v_vals = np.arange(256, dtype=np.float32)[:, None]  # (256,1)
    T = (v_vals * coeff[None, :, :]).transpose(1, 0, 2)  # (64,256,10)

    magic = 0x54424C31
    out = struct.pack("<4i", magic, 64, 0, C)
    out += T.astype("<f4").tobytes()
    out += const.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Table family (linear, or one ReLU layer) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). Format: see train.py. */
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

/* next record of a record file: 4-byte little-endian length, then the bytes. */
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
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0))) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4), *b1 = T + (long)d * 256 * W, *W2 = b1 + W, *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W), *s = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        const float *sc = acc;
        if (H) {
            for (int c = 0; c < C; c++) s[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) s[c] += h * W2[(long)u * C + c];
            }
            sc = s;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```