```hypothesis
Family: label-cnn (tiny 1-conv-layer CNN). Why: 8x8 digit images have local spatial structure; weight-sharing convolution captures translation-invariant features (strokes, loops) more parameter-efficiently than per-pixel linear or fully-connected MLP. A single 3x3 conv (16 filters) + global average pooling + linear head has ~330 params, trains fast, and should exceed label-linear (0.961) while costing far less than label-gbdt. Failure if dev accuracy ≤ 0.961 or instructions/item > 20000.
```

```python
"""label-cnn: tiny CNN for 8x8 ordinal pixel inputs (0..16).
train.py card.json train.in train.key weights.bin
Architecture: Input(1,8,8) -> Conv2d(16,3x3,pad=1) -> ReLU -> GlobalAvgPool -> Linear(10) -> Softmax.
Training: Adam lr=1e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)), seed=1.
Weights format (little-endian): magic(0x434E4E31), n_filters, kernel_size, conv_w[n_filters,3,3], conv_b[n_filters], fc_w[n_filters,10], fc_b[10] all float32.
"""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-3, 1e-4, 64, 600000, 100
MAGIC = 0x434E4E31  # 'CNN1'
N_FILTERS, K = 16, 3  # 3x3 kernel


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def im2col(x, kh, kw, pad):
    """x: (N, C, H, W) -> (N, C*kh*kw, H_out*W_out)"""
    N, C, H, W = x.shape
    H_out = H + 2 * pad - kh + 1
    W_out = W + 2 * pad - kw + 1
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    col = np.lib.stride_tricks.as_strided(
        x, shape=(N, C, kh, kw, H_out, W_out),
        strides=(x.strides[0], x.strides[1], x.strides[2], x.strides[3], x.strides[2], x.strides[3])
    )
    return col.reshape(N, C * kh * kw, H_out * W_out)


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n = X_bytes.shape[0]
    assert card["input"]["bytes"] == 64

    # Normalize to [0,1] (max pixel value is 16)
    X = (X_bytes.astype(np.float32) / 16.0).reshape(n, 1, 8, 8)

    rng = np.random.default_rng(SEED)
    # He init for conv
    fan_in = 1 * K * K
    conv_w = (rng.normal(0, np.sqrt(2.0 / fan_in), (N_FILTERS, 1, K, K))).astype(np.float32)
    conv_b = np.zeros(N_FILTERS, np.float32)
    # Xavier for fc
    fc_w = (rng.normal(0, np.sqrt(2.0 / (N_FILTERS + C)), (N_FILTERS, C))).astype(np.float32)
    fc_b = np.zeros(C, np.float32)

    params = [conv_w, conv_b, fc_w, fc_b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x, yy = X[idx], y[idx]
            bsz = x.shape[0]

            # Forward
            col = im2col(x, K, K, 1)                 # (bsz, 9, 64)
            conv_out = col.transpose(0, 2, 1) @ conv_w.reshape(N_FILTERS, -1).T  # (bsz, 64, N_FILTERS)
            conv_out += conv_b
            conv_out = conv_out.transpose(0, 2, 1).reshape(bsz, N_FILTERS, 8, 8)
            relu_out = np.maximum(conv_out, 0)
            pool_out = relu_out.mean(axis=(2, 3))    # (bsz, N_FILTERS)
            logits = pool_out @ fc_w + fc_b          # (bsz, C)

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yy] -= 1.0
            dlogits = (probs / bsz).astype(np.float32)

            # Backward
            dfc_w = pool_out.T @ dlogits + L2 * fc_w
            dfc_b = dlogits.sum(axis=0)
            dpool = dlogits @ fc_w.T                 # (bsz, N_FILTERS)
            drelu = dpool[:, :, None, None] / 64.0   # (bsz, N_FILTERS, 1, 1) broadcast to (bsz, N_FILTERS, 8, 8)
            drelu = drelu * (conv_out > 0)           # (bsz, N_FILTERS, 8, 8)
            dconv = drelu.reshape(bsz, N_FILTERS, 64).transpose(0, 2, 1)  # (bsz, 64, N_FILTERS)
            dcol = dconv @ conv_w.reshape(N_FILTERS, -1)  # (bsz, 64, 9)
            dcol = dcol.transpose(0, 2, 1)           # (bsz, 9, 64)

            # conv_w grad: (N_FILTERS, 9) = sum over batch of (dcol @ x_col.T) / bsz? Actually:
            # conv_w shape (N_FILTERS, 9). dcol (bsz, 9, 64), col (bsz, 9, 64)
            # For each filter f: dconv_w[f] = sum_b sum_pos dcol[b, :, pos] * col[b, :, pos]? Wait.
            # Better: conv_w grad = sum_b (dcol[b].T @ col[b])? No.
            # col: (bsz, 9, 64), dcol: (bsz, 9, 64). We want (N_FILTERS, 9).
            # conv_out[b, pos, f] = col[b, :, pos] @ conv_w[f]
            # dL/dconv_w[f] = sum_b sum_pos dconv[b, pos, f] * col[b, :, pos]
            # So dconv_w = sum_b (dconv[b].T @ col[b].T)? dconv[b] is (64, N_FILTERS), col[b] is (9, 64).T -> (64, 9)
            # dconv[b].T @ col[b].T = (N_FILTERS, 64) @ (64, 9) = (N_FILTERS, 9). Yes.
            dconv_w = np.zeros_like(conv_w)
            for b in range(bsz):
                dconv_w += dconv[b].T @ col[b].T
            dconv_w = dconv_w.reshape(N_FILTERS, 1, K, K) + L2 * conv_w
            dconv_b = dconv.sum(axis=(0, 1))

            grads = [dconv_w, dconv_b, dfc_w, dfc_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    out = struct.pack("<3i", MAGIC, N_FILTERS, K)
    out += conv_w.astype("<f4").tobytes()
    out += conv_b.astype("<f4").tobytes()
    out += fc_w.astype("<f4").tobytes()
    out += fc_b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-cnn inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (8x8 pixels 0..16). Output: 1 byte class 0..9.
 * Weights: magic(0x434E4E31), n_filters, k, conv_w[F][1][k][k], conv_b[F], fc_w[F][10], fc_b[10] (float32 LE).
 */
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
    if (nw < 12) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x434E4E31) { fprintf(stderr, "bad magic\n"); return 3; }
    int F = hd[1], K = hd[2];
    if (F != 16 || K != 3) { fprintf(stderr, "unexpected arch\n"); return 3; }
    size_t expected = 12
        + (size_t)F * 1 * K * K * 4
        + (size_t)F * 4
        + (size_t)F * 10 * 4
        + 10 * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 3);
    const float *conv_w = p; p += F * 9;
    const float *conv_b = p; p += F;
    const float *fc_w = p; p += F * 10;
    const float *fc_b = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *conv_out = malloc(F * 8 * 8 * sizeof(float));
    float *pool_out = malloc(F * sizeof(float));
    float *logits = malloc(10 * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }

        // Normalize input to [0,1] (divide by 16)
        float input[8][8];
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 8; c++)
                input[r][c] = xi[r * 8 + c] / 16.0f;

        // Conv2d: 16 filters, 3x3, pad=1, stride=1 -> 8x8
        for (int f = 0; f < F; f++) {
            const float *w = conv_w + f * 9;
            float b = conv_b[f];
            for (int r = 0; r < 8; r++) {
                for (int c = 0; c < 8; c++) {
                    float sum = b;
                    for (int dr = -1; dr <= 1; dr++) {
                        for (int dc = -1; dc <= 1; dc++) {
                            int rr = r + dr, cc = c + dc;
                            if (rr >= 0 && rr < 8 && cc >= 0 && cc < 8) {
                                sum += input[rr][cc] * w[(dr + 1) * 3 + (dc + 1)];
                            }
                        }
                    }
                    // ReLU
                    conv_out[f * 64 + r * 8 + c] = sum > 0.0f ? sum : 0.0f;
                }
            }
        }

        // Global average pooling
        for (int f = 0; f < F; f++) {
            float sum = 0.0f;
            for (int pidx = 0; pidx < 64; pidx++) sum += conv_out[f * 64 + pidx];
            pool_out[f] = sum / 64.0f;
        }

        // Linear layer
        for (int c = 0; c < 10; c++) {
            float sum = fc_b[c];
            for (int f = 0; f < F; f++) sum += pool_out[f] * fc_w[f * 10 + c];
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < 10; c++)
            if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```