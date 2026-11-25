```hypothesis
Family: label-linear-mlp (linear model with a small residual MLP correction).
Why: The linear model (label-linear) captures first-order statistics but misses pairwise character interactions. Adding a tiny MLP (32 hidden units) on top of the linear logits with a residual connection allows learning non-linear corrections without harming the linear baseline (initialized to zero). This adds only ~4k parameters and ~5k multiply-adds per item, far cheaper than GBDT.
Failure: If dev accuracy does not exceed label-linear's 0.3375, or if inference cost exceeds 50k Ir/item.
```
```python
import json, math, struct, sys

import numpy as np

H = 32
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
    rng = np.random.default_rng(SEED)

    # Parameters
    W_lin = np.zeros((d, 256, C), np.float32)
    b_lin = np.zeros(C, np.float32)
    W1 = np.zeros((C, H), np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = np.zeros((H, C), np.float32)
    b2 = np.zeros(C, np.float32)

    params = [W_lin, b_lin, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            B = len(yb)

            # Forward
            lin = np.tile(b_lin, (B, 1))
            for j in range(d):
                lin += W_lin[j, xb[:, j]]

            h_pre = lin @ W1 + b1
            h = np.maximum(h_pre, 0)
            out = h @ W2 + b2
            logits = lin + out

            # Loss gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = probs / B

            # Backward
            grad_lin = grad_logits.copy()
            grad_out = grad_logits.copy()

            grad_W2 = h.T @ grad_out
            grad_b2 = grad_out.sum(axis=0)
            grad_h = grad_out @ W2.T

            grad_h_pre = grad_h * (h_pre > 0)

            grad_W1 = lin.T @ grad_h_pre
            grad_b1 = grad_h_pre.sum(axis=0)
            grad_lin += grad_h_pre @ W1.T

            grad_b_lin = grad_lin.sum(axis=0)
            grad_W_lin = np.zeros_like(W_lin)
            for j in range(d):
                np.add.at(grad_W_lin[j], xb[:, j], grad_lin)

            grads = [grad_W_lin, grad_b_lin, grad_W1, grad_b1, grad_W2, grad_b2]

            # L2 regularization on weights (not biases)
            grads[0] += L2 * W_lin
            grads[2] += L2 * W1
            grads[4] += L2 * W2

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x4C4D5052  # "LPMR"
    out = struct.pack("<4i", magic, d, H, C)
    out += W_lin.astype("<f4").tobytes()
    out += b_lin.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x4C4D5052) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (d != 16 || C != 65) { fprintf(stderr, "unexpected dimensions\n"); return 3; }

    long expected = 16
        + (long)d * 256 * C * 4
        + (long)C * 4
        + (long)C * H * 4
        + (long)H * 4
        + (long)H * C * 4
        + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W_lin = (const float *)(hd + 4);
    const float *b_lin = W_lin + (long)d * 256 * C;
    const float *W1 = b_lin + C;
    const float *b1 = W1 + (long)C * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *lin = malloc(sizeof(float) * C);
    float *h = malloc(sizeof(float) * H);
    float *out = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(lin, b_lin, C * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *w = W_lin + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) lin[c] += w[c];
        }

        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int c = 0; c < C; c++) sum += lin[c] * W1[(long)c * H + u];
            h[u] = sum > 0.0f ? sum : 0.0f;
        }

        memcpy(out, b2, C * sizeof(float));
        for (int u = 0; u < H; u++) {
            float hu = h[u];
            if (hu == 0.0f) continue;
            const float *w = W2 + (long)u * C;
            for (int c = 0; c < C; c++) out[c] += hu * w[c];
        }

        for (int c = 0; c < C; c++) lin[c] += out[c];

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (lin[c] > lin[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(lin); free(h); free(out);
    return 0;
}
```